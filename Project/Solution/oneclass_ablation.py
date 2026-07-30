"""
Ablation testing the diagnosis of the one-class failure.

``open_set_scoring.py`` finds One-Class SVM and Isolation Forest performing at
or below chance, and the norm analysis attributes this to embedding magnitude:
unknown-species images produce weaker activations, so they sit nearer the centre
of the standardized feature cloud, exactly where a density estimator calls a
point typical. Magnitude, not novelty, is what these scorers end up ranking.

That diagnosis makes a falsifiable prediction. Removing magnitude by projecting
embeddings onto the unit sphere before fitting should rescue both methods. This
script fits each one twice on identical data -- once on standardized embeddings
(the original setup) and once on L2-normalized embeddings -- so the only thing
that changes is whether magnitude information survives.

Results are cached to ``open_set_outputs/oneclass_ablation_<model>.npz``.

Run: python3 oneclass_ablation.py [--model resnet18|efficientnet|both]
"""

import argparse
import os

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import open_set_scoring as oss

TARGET_TPR = 0.95
DISPLAY_NAMES = {"resnet18": "ResNet-18", "efficientnet": "EfficientNet-B0"}


def fpr_at_tpr(y_true, scores, target=TARGET_TPR):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = min(np.searchsorted(tpr, target, side="left"), len(tpr) - 1)
    return fpr[idx]


def centroid_comparison(train_emb, train_labels, test_emb, y_true):
    """Isolate class-conditioning as the cause, holding geometry fixed.

    If the normalization ablation fails to rescue the one-class methods, the
    remaining structural difference from the scorers that do work is that
    Mahalanobis and kNN model each known class separately while OC-SVM and
    Isolation Forest pool all classes into one region. This compares two scorers
    that are identical in every respect -- Euclidean distance on L2-normalized
    embeddings -- except for how many reference points they use: a single global
    centroid versus the nearest of the C class centroids.
    """
    train_l2 = oss._l2_normalize(train_emb)
    test_l2 = oss._l2_normalize(test_emb)

    classes = np.unique(train_labels)
    class_means = np.vstack([train_l2[train_labels == c].mean(axis=0)
                             for c in classes])
    global_mean = train_l2.mean(axis=0)[None, :]

    d_global = cdist(test_l2, global_mean).ravel()
    d_class = cdist(test_l2, class_means).min(axis=1)

    return {
        "n_classes": len(classes),
        "global": roc_auc_score(y_true, d_global),
        "class_conditional": roc_auc_score(y_true, d_class),
    }


def print_centroid(name, c):
    print(f"--- Class-conditioning isolation ({name}) ---")
    print("Euclidean distance on L2-normalized embeddings; the only difference "
          "is the number of reference points.")
    nearest_label = f"Nearest of {c['n_classes']} class centroids"
    print(f"{'Reference':<34}{'ROC AUC':>10}")
    print(f"{'Single global centroid':<34}{c['global']:>10.4f}")
    print(f"{nearest_label:<34}{c['class_conditional']:>10.4f}")
    print(f"{'Difference':<34}{c['class_conditional'] - c['global']:>+10.4f}\n")


def run_model(model, output_dir):
    preset = oss.MODEL_PRESETS[model]
    weights = os.path.join(oss.SCRIPT_DIR, preset["weights"])
    cache = os.path.join(output_dir, f"oneclass_ablation_{model}.npz")

    print(f"\n{'=' * 66}\n=== {DISPLAY_NAMES[model]} ===\n{'=' * 66}")

    data = oss.load_real_data(
        output_dir, weights, preset["train_file"], preset["test_file"]
    )
    y_true = 1 - data["test_is_known"]
    train_emb, test_emb = data["train_emb"], data["test_emb"]

    if os.path.exists(cache):
        print("Loading cached ablation scores.")
        npz = np.load(cache, allow_pickle=True)
        scores = {k: npz[k] for k in npz.files}
    else:
        scaler = StandardScaler().fit(train_emb)
        train_std, test_std = scaler.transform(train_emb), scaler.transform(test_emb)
        # Magnitude removed; only direction survives.
        train_l2 = oss._l2_normalize(train_emb)
        test_l2 = oss._l2_normalize(test_emb)

        scores = {}
        print("Fitting One-Class SVM on standardized embeddings (slow) ...",
              flush=True)
        scores["ocsvm_standardized"] = oss.ocsvm_scores(train_std, test_std)
        print("Fitting One-Class SVM on L2-normalized embeddings (slow) ...",
              flush=True)
        scores["ocsvm_l2"] = oss.ocsvm_scores(train_l2, test_l2)
        print("Fitting Isolation Forest on standardized embeddings ...", flush=True)
        scores["iforest_standardized"] = oss.isolation_forest_scores(
            train_std, test_std
        )
        print("Fitting Isolation Forest on L2-normalized embeddings ...", flush=True)
        scores["iforest_l2"] = oss.isolation_forest_scores(train_l2, test_l2)

        np.savez_compressed(cache, **scores)
        print(f"Cached to {os.path.basename(cache)}")

    rows = [
        ("One-Class SVM", "ocsvm_standardized", "ocsvm_l2"),
        ("Isolation Forest", "iforest_standardized", "iforest_l2"),
    ]

    print(f"\n{'Method':<20}{'Input':<16}{'ROC AUC':>10}{'FPR@95':>10}")
    print("-" * 56)
    out = {}
    for label, std_key, l2_key in rows:
        for tag, key in (("standardized", std_key), ("L2-normalized", l2_key)):
            s = np.asarray(scores[key])
            auc = roc_auc_score(y_true, s)
            fpr = fpr_at_tpr(y_true, s)
            out[(label, tag)] = (auc, fpr)
            print(f"{label:<20}{tag:<16}{auc:>10.4f}{fpr:>10.4f}")
        delta = out[(label, "L2-normalized")][0] - out[(label, "standardized")][0]
        print(f"{'':<20}{'change':<16}{delta:>+10.4f}\n")

    cent = centroid_comparison(
        train_emb, data["train_labels"], test_emb, y_true
    )
    print_centroid(DISPLAY_NAMES[model], cent)

    return out, cent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet18", "efficientnet", "both"],
                        default="both")
    parser.add_argument("--output-dir", default=oss.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    models = (["resnet18", "efficientnet"] if args.model == "both"
              else [args.model])
    for model in models:
        run_model(model, args.output_dir)


if __name__ == "__main__":
    main()
