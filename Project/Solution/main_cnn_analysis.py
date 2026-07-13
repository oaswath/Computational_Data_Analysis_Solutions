import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from PIL import Image

# ==========================================
# IMPORT YOUR CUSTOM DATA CURATOR CLASS
# ==========================================
from biodiversity_screening_data_curation import PlantNetDataCurator

# --- 1. Dataset Definition ---
class PlantNetDataset(Dataset):
    def __init__(self, dataframe, image_dir, label_map=None, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
       # Determine labels. If a map is provided, use it. Otherwise build dynamically.
        if label_map is not None:
            self.label_map = label_map
        else:
            self.species_list = self.dataframe['species_name'].unique()
            self.label_map = {species: idx for idx, species in enumerate(self.species_list)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # DYNAMIC PATH CONSTRUCTION:
        split_folder = str(row.get('split', '')) 
        species_folder = str(row.get('species_id', ''))
        
        # Use the actual physical filename we found on disk!
        image_filename = str(row['actual_filename']) 
        
        img_path = os.path.join(self.image_dir, split_folder, species_folder, image_filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image {img_path} not found. Using a black placeholder.")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        #label = self.label_map[row['species_name']]
        #handle species it doesn't recognize. Instead of crashing, we can assign all "Unknown" species a default label (like -1). Since we are only extracting embeddings at this stage and not calculating training loss, -1 works perfectly.
        label = self.label_map.get(row['species_name'], -1)
        return image, label

# --- 2. Model with Feature Extraction Hook ---
class OpenSetResNet(nn.Module):
    def __init__(self, num_known_classes):
        super(OpenSetResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_known_classes)
        self.embeddings = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.embeddings = output.squeeze()
        self.backbone.avgpool.register_forward_hook(hook)

    def forward(self, x):
        logits = self.backbone(x)
        return logits, self.embeddings

# --- 3. Splitting & Disk Syncing ---
def sync_dataframe_with_disk(df, base_image_dir, max_per_class=300):
    """
    Ignores metadata image_ids. Instead, looks at the curated species, goes 
    into the physical folders on the hard drive, grabs ALL actual .jpg files, 
    and builds a new DataFrame based on reality.
    """
    print(f"Scanning disk to find actual images for our curated species...")
    valid_rows = []
    
    # Get unique combinations of split, species_id, and species_name
    unique_groups = df[['split', 'species_id', 'species_name']].drop_duplicates()
    print(f"Found {len(unique_groups)} unique species/split combinations to check on disk.")
    
    for _, row in unique_groups.iterrows():
        split_folder = str(row['split'])
        species_folder = str(row['species_id'])
        species_name = str(row['species_name'])
        
        target_dir = os.path.join(base_image_dir, split_folder, species_folder)
        
        if os.path.exists(target_dir):
            # Grab all actual .jpg files in this specific folder
            for filename in os.listdir(target_dir):
                if filename.lower().endswith('.jpg'):
                    valid_rows.append({
                        'actual_filename': filename, # The real file on your hard drive
                        'split': split_folder,
                        'species_id': species_folder,
                        'species_name': species_name
                    })
                    
    actual_df = pd.DataFrame(valid_rows)
    print(f"Disk Scan Complete: Found {len(actual_df)} total physical images.")
    
    # CORRECTED RE-BALANCING LOGIC
    print(f"Re-balancing: Capping 'train' split at {max_per_class}, keeping all val/test...")
    balanced_chunks = []
    
    for species in actual_df['species_name'].unique():
        species_df = actual_df[actual_df['species_name'] == species]
        
        # Isolate the splits
        train_df = species_df[species_df['split'] == 'train']
        val_test_df = species_df[species_df['split'] != 'train']
        
        # Cap ONLY the training images
        n_samples = min(len(train_df), max_per_class)
        balanced_chunks.append(train_df.sample(n=n_samples, random_state=42))
        
        # Append all the validation and test images untouched
        balanced_chunks.append(val_test_df)
        
    final_df = pd.concat(balanced_chunks, ignore_index=True)
    
    # Print the exact split distribution
    print(f"\nFinal usable dataset size: {len(final_df)} images.")
    print("--- Exact Split Breakdown ---")
    print(final_df['split'].value_counts().to_string())
    print("-----------------------------\n")
    
    return final_df

def create_open_set_splits(df, unknown_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    all_species = df['species_name'].unique()
    
    num_unknown = int(len(all_species) * unknown_ratio)
    unknown_species = np.random.choice(all_species, num_unknown, replace=False)
    known_species = np.setdiff1d(all_species, unknown_species)
    
    df_known = df[df['species_name'].isin(known_species)].copy()
    df_unknown = df[df['species_name'].isin(unknown_species)].copy()
    
    df_known['is_known'] = 1
    df_unknown['is_known'] = 0
    
    return df_known, df_unknown

# --- 4. The Training Loop (NEW) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Standard PyTorch training loop to fine-tune the ResNet18 on the known classes.
    """
    print("\n--- Starting Training Loop ---")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for images, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}"):
                images = images.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass & optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation performance so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# --- 5. Feature Extraction Loop ---
def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Embeddings"):
            images = images.to(device)
            _, embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.vstack(all_embeddings), np.concatenate(all_labels)

# --- 6. Analysis Functions ---
def analyze_mutual_information(embeddings, binary_labels):
    print("Calculating Mutual Information for 512 dimensions...")
    mi_scores = mutual_info_classif(embeddings, binary_labels, random_state=42)
    mi_series = pd.Series(mi_scores, name="MI Score").sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mi_series.head(20).index, y=mi_series.head(20).values, color='teal')
    plt.title("Top 20 ResNet18 {Embedding} Dimensions (Highest MI w/ Familiarity)")
    plt.xlabel("Embedding Dimension Index (0-511)")
    plt.ylabel("Mutual Information Score")
    plt.show()
    return mi_scores

def plot_pca_embeddings(embeddings, binary_labels, num_samples=2000):
    print("Fitting PCA on embeddings...")
    idx = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
    emb_subset = embeddings[idx]
    label_subset = binary_labels[idx]
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(emb_subset)
    
    df_pca = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Class': ['Known' if l == 1 else 'Unknown' for l in label_subset]
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Class', 
        palette={'Known': 'blue', 'Unknown': 'red'},
        data=df_pca, alpha=0.6, s=40
    )
    plt.title("PCA Projection of ResNet18 {Embeddings}")
    plt.grid(True)
    plt.show()

def plot_multiple_embeddings_as_grid(embeddings, num_to_plot=16):
    """
    Plots a grid of embedding 'images' for multiple samples.
    """
    plt.figure(figsize=(16, 8))
    
    # Calculate grid dimensions (e.g., 4x4 for 16 images)
    grid_size = int(np.sqrt(num_to_plot))
    
    for i in range(num_to_plot):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Reshape the i-th embedding (512-D -> 16x32)
        reshaped = embeddings[i].reshape(16, 32)
        
        plt.imshow(reshaped, cmap='viridis', aspect='auto')
        plt.axis('off')
        plt.title(f"Idx: {i}")
        
    plt.suptitle("Activation Fingerprints {Embeddings} for First 16 Images")
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    base_image_directory = 'Data/plantnet_300K/images'
    
    # 1. Initialize the Curator and fetch the DataFrame
    print("--- Step 1: Curating Data (from metadata) ---")
    curator = PlantNetDataCurator(
        metadata_path='Data/plantnet_300K/plantnet300K_metadata_formatted.json',
        species_map_path='Data/plantnet_300K/plantnet300K_species_id_2_name.json',
        min_support=150,
        max_support=300,
        random_state=42
    )
    
    plant300_df_curated = curator.get_curated_data(target_split='train', verbose=True)
    
    # 2. SYNC WITH DISK: Replace metadata logic with physical files
    print("\n--- Step 1.5: Syncing DataFrame with actual local files ---")
    plant300_df_curated = sync_dataframe_with_disk(plant300_df_curated, base_image_directory, max_per_class=300)
    
    # 3. Setup Data Splits for Open-Set Recognition
    print("\n--- Step 2: Splitting Known/Unknown ---")
    df_known, df_unknown = create_open_set_splits(plant300_df_curated, unknown_ratio=0.2)
   
    # Create the authoritative mapping for Known Classes (e.g. 0 to 144)
    known_species_list = df_known['species_name'].unique()
    num_known_classes = len(known_species_list)
    known_label_map = {species: idx for idx, species in enumerate(known_species_list)}

    print(f"Training on {num_known_classes} Known Species. Holding out Unknowns.")
    
    # 4. Setup PyTorch Environment
    print("\n--- Step 3: Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = OpenSetResNet(num_known_classes=num_known_classes).to(device)
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Safely sample based on the remaining valid files
    sample_size = min(2000, len(df_known), len(df_unknown))
    test_df = pd.concat([df_known.sample(sample_size, random_state=42), df_unknown.sample(sample_size, random_state=42)])
    
    test_dataset = PlantNetDataset(test_df, image_dir=base_image_directory, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 5. Extract Features and Analyze before training on known classes for this image300k dataset
    print(f"\n--- Step 4: Extracting Features using {device} ---")
    embeddings, binary_labels = extract_embeddings(pretrained_model, test_loader, device)
    
    print("\n--- Step 5: Running Analysis ---")
    analyze_mutual_information(embeddings, test_df['is_known'].values)
    plot_pca_embeddings(embeddings, test_df['is_known'].values)
    plot_multiple_embeddings_as_grid(embeddings, num_to_plot=16)

    # Define augmentations for training and standard transforms for evaluation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==============================================================
    # --- TRAINING THE MODEL ON KNOWN CLASSES (NEW) --------
    # ==============================================================
    print("\n--- Step 4: Setting up Training Datasets ---")
    
    # Isolate training and validation sets strictly from the 'Known' species pool
    train_df = df_known[df_known['split'] == 'train']
    val_df = df_known[df_known['split'] == 'val']
    
    train_dataset = PlantNetDataset(train_df, image_dir=base_image_directory, label_map=known_label_map, transform=train_transform)
    val_dataset = PlantNetDataset(val_df, image_dir=base_image_directory, label_map=known_label_map, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Using a small learning rate (1e-4) to safely fine-tune the pre-trained weights
    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=1e-4)
    
    # ==============================================================
    # --- ADDED: SAVE / LOAD LOGIC HERE --------
    # ==============================================================
    weights_path = 'resnet18_finetuned_knowns.pth'
    
    if os.path.exists(weights_path):
        print(f"\n--- Loading pre-trained weights from {weights_path} ---")
        pretrained_model.load_state_dict(torch.load(weights_path, map_location=device))
        pretrained_model.eval()  # Put it into evaluation mode
        trained_model = pretrained_model # Assign for the next steps
        print("Skipping training phase and proceeding to evaluation.")
    else:
        print("\n--- No pre-trained weights found. Starting training... ---")
        # Execute Training
        epochs = 5  # You can increase this for better performance
        trained_model = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
        
        # Save the trained weights to skip this step in the future
        torch.save(trained_model.state_dict(), weights_path)
        print(f"Model weights saved to '{weights_path}'")
    # ============================================================
    ''''
    # Execute Training
    epochs = 5  # You can increase this for better performance
    trained_model = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
    
    # Save the trained weights to skip this step in the future
    torch.save(trained_model.state_dict(), 'resnet18_finetuned_knowns.pth')
    print("Model weights saved to 'resnet18_finetuned_knowns.pth'")
    '''
    # Safely sample based on the remaining valid files
    sample_size = min(2000, len(df_known), len(df_unknown))
    test_df = pd.concat([df_known.sample(sample_size, random_state=42), df_unknown.sample(sample_size, random_state=42)])
    
    test_dataset = PlantNetDataset(test_df, image_dir=base_image_directory, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
   # ==============================================================
    # ---  EXTRACT FEATURES AND EVALUATE OPEN-SET ON TRIANED MODEL wih OPEN300K -----------
    # ==============================================================
    print(f"\n--- Step 5: Extracting Features using {device} ---")
    
    # Safely sample based on the remaining valid files
    sample_size = min(2000, len(df_known), len(df_unknown))
    test_df = pd.concat([df_known.sample(sample_size, random_state=42), df_unknown.sample(sample_size, random_state=42)])
    
    # Pass the known_label_map so the dataset doesn't crash when encountering an unknown species
    test_dataset = PlantNetDataset(test_df, image_dir=base_image_directory, label_map=known_label_map, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    embeddings_resnet18, binary_labels_resnet18 = extract_embeddings(trained_model, test_loader, device)
    
    print("\n--- Step 6: Running Analysis ---")
    analyze_mutual_information(embeddings_resnet18, test_df['is_known'].values)
    plot_pca_embeddings(embeddings_resnet18, test_df['is_known'].values)
    plot_multiple_embeddings_as_grid(embeddings_resnet18, num_to_plot=16)