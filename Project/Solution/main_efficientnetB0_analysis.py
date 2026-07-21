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

        split_folder = str(row.get('split', '')) 
        species_folder = str(row.get('species_id', ''))
        image_filename = str(row['actual_filename']) 
        
        img_path = os.path.join(self.image_dir, split_folder, species_folder, image_filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        # Safe lookup for unknown species (-1 default)
        label = self.label_map.get(row['species_name'], -1)
        return image, label

# --- 2. Model with Feature Extraction Hook ---
class EfficientNetOpenSet(nn.Module):
    def __init__(self, num_known_classes):
        super(EfficientNetOpenSet, self).__init__()
        # Initialize with ImageNet V1 weights
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # EfficientNet classifier is a Sequential block; we replace the final Linear layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_known_classes)
        
        self.embeddings = None
        self._register_hook()

    def _register_hook(self):
        # Captures the 1280-D vector from the avgpool layer
        def hook(module, input, output):
            self.embeddings = output.squeeze()
        self.backbone.avgpool.register_forward_hook(hook)

    def forward(self, x):
        logits = self.backbone(x)
        return logits, self.embeddings

# --- 3. Splitting & Disk Syncing ---
def sync_dataframe_with_disk(df, base_image_dir, max_per_class=300):
    print(f"Scanning disk to find actual images for our curated species...")
    valid_rows = []
    
    unique_groups = df[['split', 'species_id', 'species_name']].drop_duplicates()
    
    for _, row in unique_groups.iterrows():
        split_folder = str(row['split'])
        species_folder = str(row['species_id'])
        species_name = str(row['species_name'])
        
        target_dir = os.path.join(base_image_dir, split_folder, species_folder)
        
        if os.path.exists(target_dir):
            for filename in os.listdir(target_dir):
                if filename.lower().endswith('.jpg'):
                    valid_rows.append({
                        'actual_filename': filename,
                        'split': split_folder,
                        'species_id': species_folder,
                        'species_name': species_name
                    })
                    
    actual_df = pd.DataFrame(valid_rows)
    print(f"Disk Scan Complete: Found {len(actual_df)} total physical images.")
    
    balanced_chunks = []
    for species in actual_df['species_name'].unique():
        species_df = actual_df[actual_df['species_name'] == species]
        
        train_df = species_df[species_df['split'] == 'train']
        val_test_df = species_df[species_df['split'] != 'train']
        
        n_samples = min(len(train_df), max_per_class)
        balanced_chunks.append(train_df.sample(n=n_samples, random_state=42))
        balanced_chunks.append(val_test_df)
        
    return pd.concat(balanced_chunks, ignore_index=True)

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

# --- 4. The Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("\n--- Starting Training Loop ---")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- 5. Feature Extraction ---
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
    print("Calculating Mutual Information for 1280 dimensions...")
    mi_scores = mutual_info_classif(embeddings, binary_labels, random_state=42)
    mi_series = pd.Series(mi_scores, name="MI Score").sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mi_series.head(20).index, y=mi_series.head(20).values, color='teal')
    plt.title("Top 20 EfficientNet-B0 Embedding Dimensions (Highest MI)")
    plt.xlabel("Embedding Dimension Index (0-1279)")
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
    plt.title("PCA Projection of EfficientNet-B0 Embeddings")
    plt.grid(True)
    plt.show()

def plot_multiple_embeddings_as_grid(embeddings, num_to_plot=16):
    plt.figure(figsize=(16, 8))
    grid_size = int(np.sqrt(num_to_plot))
    
    for i in range(num_to_plot):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # CHANGED FOR EFFICIENTNET: Reshape 1280-D vector into 32x40
        reshaped = embeddings[i].reshape(32, 40)
        
        plt.imshow(reshaped, cmap='viridis', aspect='auto')
        plt.axis('off')
        plt.title(f"Idx: {i}")
        
    plt.suptitle("Activation Fingerprints (1280-D Embeddings) for First 16 Images")
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    base_image_directory = 'Data/plantnet_300K/images'
    
    print("--- Step 1: Curating Data ---")
    curator = PlantNetDataCurator(
        metadata_path='Data/plantnet_300K/plantnet300K_metadata_formatted.json',
        species_map_path='Data/plantnet_300K/plantnet300K_species_id_2_name.json',
        min_support=150,
        max_support=300,
        random_state=42
    )
    
    plant300_df_curated = curator.get_curated_data(target_split='train', verbose=True)
    plant300_df_curated = sync_dataframe_with_disk(plant300_df_curated, base_image_directory, max_per_class=300)
    
    print("\n--- Step 2: Splitting Known/Unknown ---")
    df_known, df_unknown = create_open_set_splits(plant300_df_curated, unknown_ratio=0.2)
   
    known_species_list = df_known['species_name'].unique()
    num_known_classes = len(known_species_list)
    known_label_map = {species: idx for idx, species in enumerate(known_species_list)}

    print(f"Training on {num_known_classes} Known Species. Holding out Unknowns.")
    
    print("\n--- Step 3: Initializing EfficientNet-B0 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = EfficientNetOpenSet(num_known_classes=num_known_classes).to(device)
    
    # Standard EfficientNet/ImageNet evaluation transforms
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    known_test_df = df_known[df_known['split'] == 'test'].copy()
    unknown_test_df = df_unknown[df_unknown['split'] == 'test'].copy()

    sample_size = min(2000, len(known_test_df), len(unknown_test_df))

    test_df = pd.concat([known_test_df.sample(n=sample_size, random_state=42),unknown_test_df.sample(n=sample_size, random_state=42)], ignore_index=True)

    print(f"Open-set held-out test set: "
          f"{sample_size} known and {sample_size} unknown images."
        )
    
    test_dataset = PlantNetDataset(test_df, image_dir=base_image_directory, label_map=known_label_map, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\n--- Step 4: Extracting Pre-Trained Features using {device} ---")
    embeddings, binary_labels = extract_embeddings(pretrained_model, test_loader, device)
    
    print("\n--- Step 5: Running Pre-Training Analysis ---")
    analyze_mutual_information(embeddings, test_df['is_known'].values)
    plot_pca_embeddings(embeddings, test_df['is_known'].values)
    plot_multiple_embeddings_as_grid(embeddings, num_to_plot=16)

    # Standard EfficientNet training augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- Step 6: Setting up Training Datasets ---")
    train_df = df_known[df_known['split'] == 'train']
    val_df = df_known[df_known['split'] == 'val']
    
    train_dataset = PlantNetDataset(train_df, image_dir=base_image_directory, label_map=known_label_map, transform=train_transform)
    val_dataset = PlantNetDataset(val_df, image_dir=base_image_directory, label_map=known_label_map, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=1e-4)
    
    # ==============================================================
    # --- ENHANCED CHECKPOINT LOGIC (RESUME TRAINING) --------
    # ==============================================================
    weights_path_5_epochs = 'efficientnet_b0_finetuned_knowns.pth'
    weights_path_10_epochs = 'efficientnet_b0_finetuned_knowns_10epochs.pth'
    
    # 1. Check if we already finished the full 10 epochs
    if os.path.exists(weights_path_10_epochs):
        print(f"\n--- Loading fully trained weights from {weights_path_10_epochs} ---")
        pretrained_model.load_state_dict(torch.load(weights_path_10_epochs, map_location=device))
        pretrained_model.eval() 
        trained_model = pretrained_model 
        print("Skipping training phase and proceeding to evaluation.")
        
    # 2. Check if we have the 5-epoch save file, and RESUME training
    elif os.path.exists(weights_path_5_epochs):
        print(f"\n--- Loading pre-trained weights from {weights_path_5_epochs} ---")
        # Load the progress from the first 5 epochs
        pretrained_model.load_state_dict(torch.load(weights_path_5_epochs, map_location=device))
        
        print("\n--- Resuming training for 5 additional epochs... ---")
        additional_epochs = 5 
        # Pass the pre-loaded model back into the training loop
        trained_model = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=additional_epochs, device=device)
        
        # Save the new 10-epoch model so we don't have to do it again
        torch.save(trained_model.state_dict(), weights_path_10_epochs)
        print(f"Model weights saved to '{weights_path_10_epochs}'")
        
    # 3. If neither exist, start completely from scratch
    else:
        print("\n--- No pre-trained weights found. Starting training from scratch... ---")
        epochs = 5 
        trained_model = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
        torch.save(trained_model.state_dict(), weights_path_5_epochs)
        print(f"Model weights saved to '{weights_path_5_epochs}'")
    # ==============================================================
    # ==============================================================

    print(f"\n--- Step 7: Extracting Fine-Tuned Features using {device} ---")
    
    # Re-instantiating the test loader ensures a fresh evaluation pass
    test_dataset = PlantNetDataset(test_df, image_dir=base_image_directory, label_map=known_label_map, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    embeddings_effnet, binary_labels_effnet = extract_embeddings(trained_model, test_loader, device)
    
    # Save embeddings for downstream open-set recognition experiments
    output_dir = "open_set_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Known training embeddings
    known_train_dataset = PlantNetDataset(
        train_df,
        image_dir=base_image_directory,
        label_map=known_label_map,
        transform=eval_transform
    )

    known_train_loader = DataLoader(
        known_train_dataset,
        batch_size=64,
        shuffle=False
    )

    known_train_embeddings, known_train_labels = extract_embeddings(
        trained_model,
        known_train_loader,
        device
    )

    np.savez_compressed(
        os.path.join(output_dir, "known_train_embeddings_effnet.npz"),
        embeddings=known_train_embeddings,
        class_labels=known_train_labels
    )

    # Open-set test embeddings
    np.savez_compressed(
        os.path.join(output_dir, "open_set_test_embeddings_effnet.npz"),
        embeddings=embeddings_effnet,
        class_labels=binary_labels_effnet,
        is_known=test_df["is_known"].to_numpy(),
        species_name=test_df["species_name"].to_numpy()
    )

    print(f"Saved EfficientNet embeddings to '{output_dir}'")

    print("\n--- Step 8: Running Post-Training Analysis ---")
    analyze_mutual_information(embeddings_effnet, test_df['is_known'].values)
    plot_pca_embeddings(embeddings_effnet, test_df['is_known'].values)
    plot_multiple_embeddings_as_grid(embeddings_effnet, num_to_plot=16)