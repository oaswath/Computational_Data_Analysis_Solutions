import json
try:
    import pandas as pd  # type: ignore[import]
except ImportError:
    pd = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import numpy as np  # type: ignore[import]
except ImportError:
    np = None
try:
    import seaborn as sns  # type: ignore[import]
except ImportError:
    sns = None

# defining file paths
METADATA_PATH = 'Data/plantnet_300K/plantnet300K_metadata_formatted.json'
SPECIES_MAP_PATH = 'Data/plantnet_300K/plantnet300K_species_id_2_name.json'

def load_and_merge_data():
    # Load species mapping
    with open(SPECIES_MAP_PATH, 'r') as f:
        species_map = json.load(f)

    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Convert metadata to a pandas DataFrame
    # Assumes standard Pl@ntNet JSON structure: { "image_id": {"species_id": "...", "split": "..."} }
    df = pd.DataFrame.from_dict(metadata, orient='index').reset_index()
    df.rename(columns={'index': 'image_id'}, inplace=True)

    # Map species names using the ID
    df['species_name'] = df['species_id'].astype(str).map(species_map)
    
    # Drop records where species name mapping failed
    df = df.dropna(subset=['species_name'])
    
    return df

def analyze_data_spread(df, title_suffix="Original"):
    species_counts = df['species_name'].value_counts()
    
    # 1. Overall Data Spread (Distribution of Class Sizes)
    plt.figure(figsize=(10, 5))
    sns.histplot(species_counts, bins=50, kde=True, color='blue')
    plt.title(f"Data Spread: Number of Images per Species ({title_suffix})")
    plt.xlabel("Number of Images")
    plt.ylabel("Frequency (Number of Species)")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
    # 2. Highest Variability (Top 20 vs Bottom 20 Species)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 20 Species
    sns.barplot(x=species_counts.head(20).values, y=species_counts.head(20).index, hue=species_counts.head(20).index, legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title("Highest Representation (Top 20 Species)")
    axes[0].set_xlabel("Image Count")
    
    # Bottom 20 Species
    sns.barplot(x=species_counts.tail(20).values, y=species_counts.tail(20).index, hue=species_counts.tail(20).index, legend=False, ax=axes[1], palette="magma")
    axes[1].set_title("Lowest Representation (Bottom 20 Species)")
    axes[1].set_xlabel("Image Count")
    
    plt.tight_layout()
    plt.show()
    
    return species_counts

def _core_downsample_loop(df, min_images, max_images, random_state):
    """
    The safe, loop-based downsampling function to prevent Pandas KeyErrors.
    (Internal helper function)
    """
    species_counts = df['species_name'].value_counts()
    valid_species = species_counts[species_counts >= min_images].index
    
    df_valid = df[df['species_name'].isin(valid_species)].copy()
    sampled_chunks = []
    
    for species in valid_species:
        species_data = df_valid[df_valid['species_name'] == species]
        n_samples = min(len(species_data), max_images)
        sampled_chunks.append(species_data.sample(n=n_samples, random_state=random_state))
        
    return pd.concat(sampled_chunks, ignore_index=True)

def curated_balanced_dataset(df, target_split='train', min_images_per_species=150, max_images_per_species=300, random_state=42):
    """
    Curates the dataset with an option to target a specific split or the entire dataset.
    
    Parameters:
    - df: The loaded and merged Pandas DataFrame.
    - target_split: 'train', 'val', 'test', or 'all'.
    - min_images_per_species: Minimum required images to keep a species.
    - max_images_per_species: Maximum allowed images per species (downsampling cap).
    - random_state: Seed for reproducible random sampling.

    Returns:
    - A single combined DataFrame containing the curated data.
    """

    # USE CASE 1: Curate the entire dataset together
    if target_split.lower() == 'all':
        print("Curating the ENTIRE dataset (Train, Val, Test combined)...")
        return _core_downsample_loop(df, min_images_per_species, max_images_per_species, random_state)
    # USE CASE 2: Curate a specific split (e.g., 'train')
    else:
        print(f"Curating ONLY the '{target_split}' split...")
        
        # 1. Separate the target split from the rest of the data
        df_target = df[df['split'] == target_split].copy()
        df_others = df[df['split'] != target_split].copy()
        
        # 2. Curate ONLY the target split
        curated_target = _core_downsample_loop(df_target, min_images_per_species, max_images_per_species, random_state)
        
        # 3. Identify which species survived the curation
        valid_species = curated_target['species_name'].unique()
        
        # 4. Filter the rest of the data to match the surviving species
        # (Crucial: prevents evaluating the model on species it never trained on)
        filtered_others = df_others[df_others['species_name'].isin(valid_species)].copy()
        
        print(f"  -> Original {target_split} images: {len(df_target)}")
        print(f"  -> Curated {target_split} images:  {len(curated_target)}")
        print(f"  -> Unaltered other splits (filtered for valid species): {len(filtered_others)}")

        # 5. Recombine into a single dataframe
        balanced_df = pd.concat([curated_target, filtered_others], ignore_index=True)
        return balanced_df

def prove_balance(original_counts, curated_df):
    curated_counts = curated_df['species_name'].value_counts()
    
    # Statistical Proof
    print("--- Proof of Balance: Statistical Summary ---")
    print(f"Original Species Count: {len(original_counts)}")
    print(f"Curated Species Count:  {len(curated_counts)}")
    print(f"Original Standard Deviation of Class Size: {original_counts.std():.2f}")
    print(f"Curated Standard Deviation of Class Size:  {curated_counts.std():.2f}")
    print(f"Original Median Images/Species: {original_counts.median()}")
    print(f"Curated Median Images/Species:  {curated_counts.median()}")
    
    # Graphical Proof
    plt.figure(figsize=(10, 5))
    sns.histplot(curated_counts, bins=30, kde=False, color='green')
    plt.title("Curated Data Spread: Uniformity Achieved")
    plt.xlabel("Number of Images")
    plt.ylabel("Frequency (Number of Species)")
    plt.axvline(curated_counts.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {curated_counts.mean():.1f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == "__main__":
    if pd is None:
        raise ImportError("pandas is required for this script to run.")
    
    plant300_df_original = load_and_merge_data()
    print(plant300_df_original.info())
    print(plant300_df_original.all())
    print(f"Total original images: {len(plant300_df_original)}")
    
    original_counts = analyze_data_spread(plant300_df_original)

    print(f"Total species in original dataset: {original_counts.shape[0]}")
    print(original_counts.describe())

    MIN_SUPPORT = 150
    MAX_SUPPORT = 300
    RANDOM_STATE = 42

    plant300_df_curated = curated_balanced_dataset(plant300_df_original, target_split='train', min_images_per_species=MIN_SUPPORT, max_images_per_species=MAX_SUPPORT, random_state=RANDOM_STATE)
    print("columns in curated dataset:", plant300_df_curated.columns)
    print(plant300_df_curated.info())
    print(plant300_df_curated.all())
    print(f"Total curated images: {len(plant300_df_curated)}")
    
    curated_counts = analyze_data_spread(plant300_df_curated, title_suffix="Curated")

    print(f"Total species in curated dataset: {curated_counts.shape[0]}")
    print(curated_counts.describe())

    prove_balance(original_counts, plant300_df_curated)