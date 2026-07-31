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


# --- 4. The Training Loop (ENHANCED FOR HISTORY TRACKING & RESUMING) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, history=None, start_epoch=0):
    """
    Standard PyTorch training loop to fine-tune the ResNet18 on the known classes.
    Now includes tracking for accuracy history and resuming capabilities.
    """
    print("\n--- Starting Training Loop ---")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Initialize a new history dictionary if starting from scratch, or use the passed one
    if history is None:
        history = {'train_acc': [], 'val_acc': []}
        best_acc = 0.0
    else:
        # If resuming, fetch the highest accuracy achieved in previous epochs
        best_acc = max(history['val_acc']) if history['val_acc'] else 0.0

    # Total display epochs for the console output
    total_target_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_target_epochs):
        print(f'Epoch {epoch+1}/{total_target_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                dataloader = train_loader
            else:
                model.eval()   
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for images, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}"):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store the accuracy in our history dictionary
            if phase == 'train':
                history['train_acc'].append(epoch_acc)
            else:
                history['val_acc'].append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history
'''

# --- 4. The Training Loop (NEW) ---
# --- 4. The Training Loop (ENHANCED FOR HISTORY TRACKING) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Standard PyTorch training loop to fine-tune the ResNet18 on the known classes.
    Now includes tracking for accuracy history.
    """
    print("\n--- Starting Training Loop ---")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Dictionary to keep track of accuracies
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                dataloader = train_loader
            else:
                model.eval()   
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for images, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}"):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            # Extract float value for tracking
            epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store the accuracy in our history dictionary
            if phase == 'train':
                history['train_acc'].append(epoch_acc)
            else:
                history['val_acc'].append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history
'''
'''
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
'''
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

def plot_training_history(history):
    """
    Plots the training and validation accuracy across epochs.
    """
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
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
    # --- ENHANCED CHECKPOINT LOGIC (RESUME, HISTORY & .PTH EXTRACTION) ---
    # ==============================================================
    checkpoint_path_5_epochs = 'resnet18_checkpoint_5epochs.pt'
    weights_path_5_epochs = 'resnet18_weights_5epochs.pth'
    
    checkpoint_path_10_epochs = 'resnet18_checkpoint_10epochs.pt'
    weights_path_10_epochs = 'resnet18_weights_10epochs.pth'
    
    # 1. Check if we already finished the full 10 epochs
    if os.path.exists(checkpoint_path_10_epochs):
        print(f"\n--- Loading fully trained checkpoint from {checkpoint_path_10_epochs} ---")
        checkpoint = torch.load(checkpoint_path_10_epochs, map_location=device)
        
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint['history']
        
        # Extract and save the .pth file if it doesn't exist yet
        if not os.path.exists(weights_path_10_epochs):
            print(f"--- Extracting weights from checkpoint and saving to '{weights_path_10_epochs}' ---")
            torch.save(pretrained_model.state_dict(), weights_path_10_epochs)
        
        pretrained_model.eval() 
        trained_model = pretrained_model 
        print("Skipping training phase and proceeding to evaluation.")
        
        # Plot the complete 10-epoch history
        plot_training_history(history)
        
    # 2. Check if we have the 5-epoch save file, and RESUME training
    elif os.path.exists(checkpoint_path_5_epochs):
        print(f"\n--- Loading pre-trained checkpoint from {checkpoint_path_5_epochs} ---")
        checkpoint = torch.load(checkpoint_path_5_epochs, map_location=device)
        
        # Load weights, history, AND the optimizer state to retain momentum
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        
        # Extract and save the 5-epoch .pth file if it doesn't exist yet
        if not os.path.exists(weights_path_5_epochs):
            print(f"--- Extracting weights from 5-epoch checkpoint and saving to '{weights_path_5_epochs}' ---")
            torch.save(pretrained_model.state_dict(), weights_path_5_epochs)
        
        print("\n--- Resuming training for 5 additional epochs (Epochs 6-10)... ---")
        additional_epochs = 5 
        
        # Pass the pre-loaded model, history, and starting epoch back into the training loop
        trained_model, history = train_model(
            pretrained_model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=additional_epochs, device=device, history=history, start_epoch=5
        )
        
        # Plot the newly concatenated 10-epoch history
        plot_training_history(history)
        
        # Save the full 10-epoch checkpoint (.pt) and pure weights (.pth)
        checkpoint_10 = {
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(checkpoint_10, checkpoint_path_10_epochs)
        torch.save(trained_model.state_dict(), weights_path_10_epochs)
        print(f"Model checkpoint and standalone weights saved for 10 epochs.")
        
    # 3. If neither exist, start completely from scratch
    else:
        print("\n--- No pre-trained weights found. Starting training from scratch (Epochs 1-5)... ---")
        epochs = 5 
        
        # Unpack both the model and the freshly generated history dictionary
        trained_model, history = train_model(
            pretrained_model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=epochs, device=device, start_epoch=0
        )
        
        # Plot the first 5 epochs
        plot_training_history(history)
        
        # Save the 5-epoch checkpoint (.pt) and pure weights (.pth)
        checkpoint_5 = {
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(checkpoint_5, checkpoint_path_5_epochs)
        torch.save(trained_model.state_dict(), weights_path_5_epochs)
        print(f"Model checkpoint and standalone weights saved for 5 epochs.")
    # ==============================================================
    '''
    # ==============================================================
    # --- ADDED: SAVE / LOAD LOGIC HERE (ENHANCED FOR HISTORY) --------
    # ==============================================================
    checkpoint_path = 'resnet18_finetuned_knowns_checkpoint.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"\n--- Loading checkpoint from {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load both the model weights and the history
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint['history']
        
        pretrained_model.eval()  
        trained_model = pretrained_model 
        print("Skipping training phase and proceeding to evaluation.")
        
        # Plot the history loaded from disk
        plot_training_history(history)
        
    else:
        print("\n--- No checkpoint found. Starting training... ---")
        epochs = 5  
        
        # Unpack both the model and the history dictionary
        trained_model, history = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
        
        # Plot the live history just generated
        plot_training_history(history)
        
        # Save both weights and history into a single checkpoint dictionary
        checkpoint = {
            'model_state_dict': trained_model.state_dict(),
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model and training history saved to '{checkpoint_path}'")
    # ============================================================
    '''
    '''
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
    '''
    '''
    # Execute Training
    epochs = 5  # You can increase this for better performance
    trained_model = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
    
    # Save the trained weights to skip this step in the future
    torch.save(trained_model.state_dict(), 'resnet18_finetuned_knowns.pth')
    print("Model weights saved to 'resnet18_finetuned_knowns.pth'")
    '''

    # ==============================================================
    # --- Extract and save embeddings from known training images ---
    # ==============================================================

    print("\n--- Extracting Known Training Embeddings ---")

    train_embedding_dataset = PlantNetDataset(
        train_df,
        image_dir=base_image_directory,
        label_map=known_label_map,
        transform=eval_transform
    )

    train_embedding_loader = DataLoader(
        train_embedding_dataset,
        batch_size=64,
        shuffle=False
    )

    train_embeddings, train_class_labels = extract_embeddings(
        trained_model,
        train_embedding_loader,
        device
    )

    output_dir = "open_set_outputs"
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "known_train_embeddings.npz"),
        embeddings=train_embeddings,
        class_labels=train_class_labels,
        species_name=train_df["species_name"].to_numpy()
    )

    print(
        "Saved known training embeddings to:",
        os.path.join(output_dir, "known_train_embeddings.npz")
    )

    
   # ==============================================================
    # --- Extract features and evaluate the trained model on the open-set test data ---
    # ==============================================================
    print(f"\n--- Step 5: Extracting Features using {device} ---")
    
    # Safely sample based on the remaining valid files
    #sample_size = min(2000, len(df_known), len(df_unknown))
    #test_df = pd.concat([df_known.sample(sample_size, random_state=42), df_unknown.sample(sample_size, random_state=42)]) 
    # Use only held-out test images for open-set evaluation
    known_test_df = df_known[df_known["split"] == "test"].copy()
    unknown_test_df = df_unknown[df_unknown["split"] == "test"].copy()

    sample_size = min(
        2000,
        len(known_test_df),
        len(unknown_test_df)
    )

    known_test_sample = known_test_df.sample(
        sample_size,
        random_state=42
    )

    unknown_test_sample = unknown_test_df.sample(
        sample_size,
        random_state=42
    )

    test_df = pd.concat(
        [known_test_sample, unknown_test_sample],
        ignore_index=True
    )

    test_df = test_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nOpen-set test composition:")
    print(test_df["is_known"].value_counts())
    print("Known test species:", known_test_sample["species_name"].nunique())
    print("Unknown test species:", unknown_test_sample["species_name"].nunique())
    
    
    # Pass the known_label_map so the dataset doesn't crash when encountering an unknown species
    test_dataset = PlantNetDataset(test_df, image_dir=base_image_directory, label_map=known_label_map, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    embeddings_resnet18, binary_labels_resnet18 = extract_embeddings(trained_model, test_loader, device)

    output_dir = "open_set_outputs"
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "open_set_test_embeddings.npz"),
        embeddings=embeddings_resnet18,
        class_labels=binary_labels_resnet18,
        is_known=test_df["is_known"].to_numpy(),
        species_name=test_df["species_name"].to_numpy()
    )

    print(
        "Saved open-set test embeddings to:",
        os.path.join(output_dir, "open_set_test_embeddings.npz")
    )
    
    print("\n--- Step 6: Running Analysis ---")
    analyze_mutual_information(embeddings_resnet18, test_df['is_known'].values)
    plot_pca_embeddings(embeddings_resnet18, test_df['is_known'].values)
    plot_multiple_embeddings_as_grid(embeddings_resnet18, num_to_plot=16)