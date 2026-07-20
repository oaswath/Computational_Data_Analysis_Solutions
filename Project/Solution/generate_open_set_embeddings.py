"""
Headless generator for the two open-set embedding files.

This reproduces exactly the embedding-export portion of ``main_cnn_analysis.py``
(same curator, same seeds, same known/unknown split, same label map, same
save schema) but:

  * runs on Apple MPS / CUDA / CPU (whichever is available) instead of the
    cuda-or-cpu-only device in the original script, and
  * skips the interactive MI/PCA plotting passes so it can run unattended.

It imports the dataset/model/split helpers straight from the existing modules,
so nothing about the pipeline logic is re-implemented or changed. Importing
``main_cnn_analysis`` does not execute its ``__main__`` block.

Output (written to ``open_set_outputs/``):
  known_train_embeddings.npz : embeddings (N,512), class_labels (0..180),
                               species_name
  open_set_test_embeddings.npz : embeddings (N,512), class_labels (known idx or
                                 -1), is_known (1/0), species_name

Run:
    python3 generate_open_set_embeddings.py
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from biodiversity_screening_data_curation import PlantNetDataCurator
from main_cnn_analysis import (
    OpenSetResNet,
    PlantNetDataset,
    create_open_set_splits,
    extract_embeddings,
    sync_dataframe_with_disk,
)

BASE_IMAGE_DIR = "Data/plantnet_300K/images"
METADATA_PATH = "Data/plantnet_300K/plantnet300K_metadata_formatted.json"
SPECIES_MAP_PATH = "Data/plantnet_300K/plantnet300K_species_id_2_name.json"
WEIGHTS_PATH = "resnet18_finetuned_knowns.pth"
OUTPUT_DIR = "open_set_outputs"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = pick_device()
    print(f"Using device: {device}")

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # --- Curate + sync with disk (identical to main_cnn_analysis.py) ---
    print("\n--- Curating data ---")
    curator = PlantNetDataCurator(
        metadata_path=METADATA_PATH,
        species_map_path=SPECIES_MAP_PATH,
        min_support=150,
        max_support=300,
        random_state=42,
    )
    df = curator.get_curated_data(target_split="train", verbose=True)

    print("\n--- Syncing with disk ---")
    df = sync_dataframe_with_disk(df, BASE_IMAGE_DIR, max_per_class=300)

    print("\n--- Known/unknown split ---")
    df_known, df_unknown = create_open_set_splits(df, unknown_ratio=0.2)

    known_species_list = df_known["species_name"].unique()
    num_known_classes = len(known_species_list)
    known_label_map = {sp: idx for idx, sp in enumerate(known_species_list)}
    print(f"Known classes: {num_known_classes}")

    # --- Build model and load the fine-tuned checkpoint ---
    print("\n--- Loading model ---")
    model = OpenSetResNet(num_known_classes=num_known_classes).to(device)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Known training embeddings ---
    print("\n--- Extracting known training embeddings ---")
    train_df = df_known[df_known["split"] == "train"]
    train_ds = PlantNetDataset(
        train_df,
        image_dir=BASE_IMAGE_DIR,
        label_map=known_label_map,
        transform=eval_transform,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    train_emb, train_labels = extract_embeddings(model, train_loader, device)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "known_train_embeddings.npz"),
        embeddings=train_emb,
        class_labels=train_labels,
        species_name=train_df["species_name"].to_numpy(),
    )
    print(f"Saved known_train_embeddings.npz  {train_emb.shape}")

    # --- Open-set test embeddings (held-out test split only) ---
    print("\n--- Extracting open-set test embeddings ---")
    known_test_df = df_known[df_known["split"] == "test"].copy()
    unknown_test_df = df_unknown[df_unknown["split"] == "test"].copy()

    sample_size = min(2000, len(known_test_df), len(unknown_test_df))
    known_test_sample = known_test_df.sample(sample_size, random_state=42)
    unknown_test_sample = unknown_test_df.sample(sample_size, random_state=42)

    test_df = pd.concat(
        [known_test_sample, unknown_test_sample], ignore_index=True
    )
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Open-set test composition:")
    print(test_df["is_known"].value_counts())

    test_ds = PlantNetDataset(
        test_df,
        image_dir=BASE_IMAGE_DIR,
        label_map=known_label_map,
        transform=eval_transform,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "open_set_test_embeddings.npz"),
        embeddings=test_emb,
        class_labels=test_labels,
        is_known=test_df["is_known"].to_numpy(),
        species_name=test_df["species_name"].to_numpy(),
    )
    print(f"Saved open_set_test_embeddings.npz  {test_emb.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
