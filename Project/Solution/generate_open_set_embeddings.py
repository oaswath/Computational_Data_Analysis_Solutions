"""
Headless open-set embedding generator.

Mirrors the training scripts' export (same curation, seeds, split, and schema)
but runs on MPS/CUDA/CPU without the interactive plots. Both backbones share the
curation/split, so their known/unknown sets match and compare fairly:

    python3 generate_open_set_embeddings.py --model resnet18|efficientnet

Writes to open_set_outputs/: known_train_*.npz (embeddings, class_labels,
species_name) and open_set_test_*.npz (+ is_known).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from biodiversity_screening_data_curation import PlantNetDataCurator

BASE_IMAGE_DIR = "Data/plantnet_300K/images"
METADATA_PATH = "Data/plantnet_300K/plantnet300K_metadata_formatted.json"
SPECIES_MAP_PATH = "Data/plantnet_300K/plantnet300K_species_id_2_name.json"
OUTPUT_DIR = "open_set_outputs"

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# resnet18 keeps un-suffixed names; efficientnet uses the merged pipeline's _effnet.
MODELS = {
    "resnet18": {
        "weights": "resnet18_finetuned_knowns.pth",
        "transform": transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE]
        ),
        "train_file": "known_train_embeddings.npz",
        "test_file": "open_set_test_embeddings.npz",
    },
    "efficientnet": {
        "weights": "efficientnet_b0_finetuned_knowns_10epochs.pth",
        "transform": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        ),
        "train_file": "known_train_embeddings_effnet.npz",
        "test_file": "open_set_test_embeddings_effnet.npz",
    },
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def import_backbone(model_name):
    """Return (ModelClass, helpers) for a backbone without building a model
    (so helpers are cheap; the model is constructed only when needed)."""
    if model_name == "resnet18":
        from main_cnn_analysis import (
            OpenSetResNet as ModelClass,
            PlantNetDataset,
            create_open_set_splits,
            extract_embeddings,
            sync_dataframe_with_disk,
        )
    else:
        from main_efficientnetB0_analysis import (
            EfficientNetOpenSet as ModelClass,
            PlantNetDataset,
            create_open_set_splits,
            extract_embeddings,
            sync_dataframe_with_disk,
        )

    helpers = (
        PlantNetDataset,
        create_open_set_splits,
        extract_embeddings,
        sync_dataframe_with_disk,
    )
    return ModelClass, helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="resnet18",
        help="Backbone to extract embeddings from (default: resnet18).",
    )
    args = parser.parse_args()
    cfg = MODELS[args.model]

    device = pick_device()
    print(f"Model: {args.model} | device: {device}")

    eval_transform = cfg["transform"]

    # Import helpers now; build the model later, once we know the class count.
    ModelClass, helpers = import_backbone(args.model)
    PlantNetDataset, create_open_set_splits, extract_embeddings, sync_disk = helpers

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
    df = sync_disk(df, BASE_IMAGE_DIR, max_per_class=300)

    print("\n--- Known/unknown split ---")
    df_known, df_unknown = create_open_set_splits(df, unknown_ratio=0.2)

    known_species_list = df_known["species_name"].unique()
    num_known_classes = len(known_species_list)
    known_label_map = {sp: idx for idx, sp in enumerate(known_species_list)}
    print(f"Known classes: {num_known_classes}")

    print("\n--- Loading model ---")
    model = ModelClass(num_known_classes=num_known_classes).to(device)
    model.load_state_dict(torch.load(cfg["weights"], map_location=device))
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
        os.path.join(OUTPUT_DIR, cfg["train_file"]),
        embeddings=train_emb,
        class_labels=train_labels,
        species_name=train_df["species_name"].to_numpy(),
    )
    print(f"Saved {cfg['train_file']}  {train_emb.shape}")

    # --- Open-set test embeddings (held-out test split) ---
    print("\n--- Extracting open-set test embeddings ---")
    known_test_df = df_known[df_known["split"] == "test"].copy()
    unknown_test_df = df_unknown[df_unknown["split"] == "test"].copy()

    # Same per-class sampling and concat order as the merged pipeline (no shuffle).
    sample_size = min(2000, len(known_test_df), len(unknown_test_df))
    test_df = pd.concat(
        [
            known_test_df.sample(n=sample_size, random_state=42),
            unknown_test_df.sample(n=sample_size, random_state=42),
        ],
        ignore_index=True,
    )

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
        os.path.join(OUTPUT_DIR, cfg["test_file"]),
        embeddings=test_emb,
        class_labels=test_labels,
        is_known=test_df["is_known"].to_numpy(),
        species_name=test_df["species_name"].to_numpy(),
    )
    print(f"Saved {cfg['test_file']}  {test_emb.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
