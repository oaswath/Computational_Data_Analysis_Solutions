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

class PlantNetDataCurator:
    def __init__(self, metadata_path, species_map_path, min_support=150, max_support=300, random_state=42):
        self.metadata_path = metadata_path
        self.species_map_path = species_map_path
        self.min_support = min_support
        self.max_support = max_support
        self.random_state = random_state

    def load_and_merge_data(self):
        """Loads metadata and maps species IDs to names."""
        with open(self.species_map_path, 'r') as f:
            species_map = json.load(f)

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        df = pd.DataFrame.from_dict(metadata, orient='index').reset_index()
        df.rename(columns={'index': 'image_id'}, inplace=True)

        # Map species names using the ID
        df['species_name'] = df['species_id'].astype(str).map(species_map)
        df = df.dropna(subset=['species_name'])
        
        return df

    def _core_downsample_loop(self, df):
        """Internal helper function to downsample safely."""
        species_counts = df['species_name'].value_counts()
        valid_species = species_counts[species_counts >= self.min_support].index
        
        df_valid = df[df['species_name'].isin(valid_species)].copy()
        sampled_chunks = []
        
        for species in valid_species:
            species_data = df_valid[df_valid['species_name'] == species]
            n_samples = min(len(species_data), self.max_support)
            sampled_chunks.append(species_data.sample(n=n_samples, random_state=self.random_state))
            
        return pd.concat(sampled_chunks, ignore_index=True)

    def get_curated_data(self, target_split='train', verbose=True):
        """
        Master method to load, merge, and curate the dataset.
        Returns the finalized curated Pandas DataFrame.
        """
        if verbose: print("Loading and merging original data...")
        df_original = self.load_and_merge_data()

        if target_split.lower() == 'all':
            if verbose: print("Curating the ENTIRE dataset...")
            return self._core_downsample_loop(df_original)
        else:
            if verbose: print(f"Curating ONLY the '{target_split}' split...")
            df_target = df_original[df_original['split'] == target_split].copy()
            df_others = df_original[df_original['split'] != target_split].copy()
            
            print(f"Total initial {target_split} images: {len(df_target)}")
            curated_target = self._core_downsample_loop(df_target)
            valid_species = curated_target['species_name'].unique()
            
            # Filter the rest of the data to match the surviving species
            filtered_others = df_others[df_others['species_name'].isin(valid_species)].copy()
            balanced_df = pd.concat([curated_target, filtered_others], ignore_index=True)
            
            if verbose:
                print(f"  -> Curated {target_split} images: {len(curated_target)}")
                print(f"  -> Total curated dataset size: {len(balanced_df)}")
                
            return balanced_df

    def analyze_data_spread(self, df, title_suffix="Original"):
        """Generates plots to visualize the distribution of classes."""
        species_counts = df['species_name'].value_counts()
        
        if plt is not None and sns is not None:
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
        else:
            print("matplotlib and seaborn are required for plotting. Skipping plots.")
            
        return species_counts

    def prove_balance(self, original_counts, curated_df):
        """Prints statistical proofs and plots a histogram to show balance was achieved."""
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
        if plt is not None and sns is not None:
            plt.figure(figsize=(10, 5))
            sns.histplot(curated_counts, bins=30, kde=False, color='green')
            plt.title("Curated Data Spread: Uniformity Achieved")
            plt.xlabel("Number of Images")
            plt.ylabel("Frequency (Number of Species)")
            plt.axvline(curated_counts.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {curated_counts.mean():.1f}')
            plt.legend()
            plt.grid(axis='y', alpha=0.75)
            plt.show()


# ==========================================
# STANDALONE EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    if pd is None:
        raise ImportError("pandas is required for this script to run.")
        
    print("Initializing PlantNetDataCurator...")
    curator = PlantNetDataCurator(
        metadata_path='Data/plantnet_300K/plantnet300K_metadata_formatted.json',
        species_map_path='Data/plantnet_300K/plantnet300K_species_id_2_name.json',
        min_support=150,
        max_support=300,
        random_state=42
    )
    
    # 1. Load original data to show spread before curation
    print("\n--- Analyzing Original Dataset ---")
    plant300_df_original = curator.load_and_merge_data()
    print(f"Total original images: {len(plant300_df_original)}")
    original_counts = curator.analyze_data_spread(plant300_df_original, title_suffix="Original")
    print(original_counts.describe())
    
    # 2. Get the fully curated dataset
    print("\n--- Curating Dataset ---")
    plant300_df_curated = curator.get_curated_data(target_split='train', verbose=True)
    print(f"Total curated images: {len(plant300_df_curated)}")
    
    # 3. Analyze curated spread
    print("\n--- Analyzing Curated Dataset ---")
    curated_counts = curator.analyze_data_spread(plant300_df_curated, title_suffix="Curated")
    print(curated_counts.describe())
    
    # 4. Prove balance
    print("\n--- Final Balance Proof ---")
    curator.prove_balance(original_counts, plant300_df_curated)