"""
Open-set recognition scoring for the PlantNet biodiversity screening project.

Consumes the embedding files written by ``generate_open_set_embeddings.py`` into
``open_set_outputs/`` plus the matching classifier head (.pth), then builds
several "is this an unfamiliar species?" scorers (MSP, Energy, Mahalanobis, kNN,
One-Class SVM, Isolation Forest) and compares them. The CNN is never retrained.
Both backbones are supported via ``--model`` (resnet18 or efficientnet); each
preset points at the right weights/embeddings/output files.

Convention: y_true = 1 - is_known, so 1 == "unfamiliar" is the positive class,
and every scorer is oriented so a HIGHER score means MORE unfamiliar.

Run: python open_set_scoring.py [--model resnet18|efficientnet]
"""

import argparse
import os
import warnings

import numpy as np

# Silence spurious macOS BLAS matmul FP warnings (not real numeric issues here).
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

TARGET_TPR = 0.95

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "open_set_outputs")

# Per-model defaults so `--model efficientnet` picks the right files/weights.
# resnet18 keeps the original (un-suffixed) filenames; efficientnet is suffixed.
MODEL_PRESETS = {
    "resnet18": {
        "weights": "resnet18_finetuned_knowns.pth",
        "train_file": "known_train_embeddings.npz",
        "test_file": "open_set_test_embeddings.npz",
        "roc_file": "roc_curves.png",
    },
    "efficientnet": {
        # Canonical _effnet filenames match the merged EfficientNet pipeline
        # (main_efficientnetB0_analysis.py) so both consume the same embeddings.
        "weights": "efficientnet_b0_finetuned_knowns_10epochs.pth",
        "train_file": "known_train_embeddings_effnet.npz",
        "test_file": "open_set_test_embeddings_effnet.npz",
        "roc_file": "roc_curves_effnet.png",
    },
}


# ==========================================================================
# Data loading
# ==========================================================================
def load_real_data(output_dir, weights_path, train_file, test_file):
    """Load both npz embedding files and the classifier head (W, b) from disk."""
    train_path = os.path.join(output_dir, train_file)
    test_path = os.path.join(output_dir, test_file)

    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected embedding file not found: {path}\n"
                "Generate the embeddings first (generate_open_set_embeddings.py)."
            )

    train_npz = np.load(train_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    W, b = _load_classifier_head(weights_path)

    return {
        "train_emb": train_npz["embeddings"].astype(np.float64),
        "train_labels": train_npz["class_labels"].astype(int),
        "test_emb": test_npz["embeddings"].astype(np.float64),
        "test_is_known": test_npz["is_known"].astype(int),
        "W": W,
        "b": b,
    }


# Classifier-head keys per backbone: ResNet18 uses backbone.fc, EfficientNet-B0
# uses backbone.classifier.1 (the Linear inside its Sequential classifier).
_HEAD_KEYS = [
    ("backbone.fc.weight", "backbone.fc.bias"),
    ("backbone.classifier.1.weight", "backbone.classifier.1.bias"),
]


def _load_classifier_head(weights_path):
    """Rebuild the linear head (W, b) from the .pth so logits can be recomputed.

    Logits are not saved at runtime. Torch is imported lazily so it is only
    needed when this actually runs.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Classifier weights not found: {weights_path}\n"
            "The MSP baseline needs the final Linear layer's weight/bias."
        )

    import torch

    state_dict = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    for w_key, b_key in _HEAD_KEYS:
        if w_key in state_dict and b_key in state_dict:
            W = state_dict[w_key].cpu().numpy().astype(np.float64)
            b = state_dict[b_key].cpu().numpy().astype(np.float64)
            return W, b

    raise KeyError(
        f"No known classifier-head keys found in {weights_path}. "
        f"Tried: {_HEAD_KEYS}"
    )


# ==========================================================================
# Scorers  (all oriented so higher == more unfamiliar)
# ==========================================================================
def _l2_normalize(x):
    """Row-wise L2 normalization (cosine geometry); helps distance-based scorers."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _class_means(train_emb, train_labels):
    """Per-class mean vectors, one row per sorted class label."""
    classes = np.unique(train_labels)
    return np.vstack([train_emb[train_labels == c].mean(axis=0) for c in classes])


def msp_scores(test_emb, W, b):
    """MSP baseline: 1 - max softmax over the reconstructed logits."""
    logits = test_emb @ W.T + b
    return 1.0 - softmax(logits, axis=1).max(axis=1)


def energy_scores(test_emb, W, b):
    """Energy score: -logsumexp(logits); uses the whole logit vector, not just max."""
    return -logsumexp(test_emb @ W.T + b, axis=1)


def mahalanobis_scores(train_emb, train_labels, test_emb, normalize=False):
    """Min Mahalanobis distance to a class mean, using one shared Ledoit-Wolf
    covariance (per-class covariances are singular in these high-D embedding
    spaces). ``normalize=True`` L2-normalizes first for cosine geometry.
    """
    if normalize:
        train_emb, test_emb = _l2_normalize(train_emb), _l2_normalize(test_emb)

    means = _class_means(train_emb, train_labels)
    residuals = train_emb - means[np.searchsorted(np.unique(train_labels), train_labels)]
    precision = LedoitWolf().fit(residuals).precision_

    return cdist(test_emb, means, metric="mahalanobis", VI=precision).min(axis=1)


def knn_scores(train_emb, test_emb, k=5, normalize=False):
    """Mean distance to the k nearest known-training embeddings.

    ``normalize=True`` L2-normalizes first (cosine geometry); ``run()`` enables
    it. Default matches ``mahalanobis_scores`` for a consistent API.
    """
    if normalize:
        train_emb, test_emb = _l2_normalize(train_emb), _l2_normalize(test_emb)

    nn = NearestNeighbors(n_neighbors=k).fit(train_emb)
    return nn.kneighbors(test_emb)[0].mean(axis=1)


def ocsvm_scores(train_scaled, test_scaled):
    """One-Class SVM (rbf); negate decision_function so higher == more unfamiliar."""
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    clf.fit(train_scaled)
    return -clf.decision_function(test_scaled)


def isolation_forest_scores(train_scaled, test_scaled, seed=42):
    """Isolation Forest; negate score_samples so higher == more unfamiliar."""
    clf = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    clf.fit(train_scaled)
    return -clf.score_samples(test_scaled)


# ==========================================================================
# Metrics
# ==========================================================================
def evaluate_scorer(y_true, scores):
    """ROC AUC, FPR@95%TPR, and precision/recall/F1 at the 95%-TPR threshold."""
    auc = roc_auc_score(y_true, scores)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # First operating point reaching the target TPR.
    idx = np.searchsorted(tpr, TARGET_TPR, side="left")
    idx = min(idx, len(tpr) - 1)
    fpr_at_tpr = fpr[idx]
    threshold = thresholds[idx]

    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    return {
        "auc": auc,
        "fpr_at_95tpr": fpr_at_tpr,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Table row order (MSP first as the baseline).
SCORER_ORDER = [
    "MSP",
    "Energy",
    "Mahalanobis",
    "Mahalanobis (L2-norm)",
    "kNN (L2-norm)",
    "One-Class SVM",
    "Isolation Forest",
]


def print_comparison_table(results):
    """Print one table with MSP as the baseline (first) row."""
    order = SCORER_ORDER

    label_width = max(len(n) for n in order) + len(" (baseline)") + 1
    header = (
        f"{'Scorer':<{label_width}}{'ROC AUC':>10}{'FPR@95TPR':>12}"
        f"{'Precision':>11}{'Recall':>9}{'F1':>8}"
    )
    print("\n" + "=" * len(header))
    print("Open-Set Detection Comparison  (positive class = 'unfamiliar')")
    print("Precision / Recall / F1 reported at the threshold giving 95% TPR")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in order:
        if name not in results:
            continue
        r = results[name]
        label = name + (" (baseline)" if name == "MSP" else "")
        print(
            f"{label:<{label_width}}{r['auc']:>10.4f}{r['fpr_at_95tpr']:>12.4f}"
            f"{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}"
        )
    print("=" * len(header))

    flipped = [name for name, r in results.items() if r["auc"] < 0.5]
    if flipped:
        print(
            "\n[WARNING] AUC < 0.5 for: "
            + ", ".join(flipped)
            + "\n          This means the score orientation is flipped for these "
            "scorers\n          (lower scores are landing on the 'unfamiliar' "
            "class). Check the\n          sign convention -- every score must be "
            "higher = more unfamiliar."
        )


# ==========================================================================
# Main
# ==========================================================================
def run(data, roc_path=None):
    train_emb = data["train_emb"]
    train_labels = data["train_labels"]
    test_emb = data["test_emb"]
    test_is_known = data["test_is_known"]
    W = data["W"]
    b = data["b"]

    y_true = 1 - test_is_known  # 1 == unfamiliar

    print(
        f"Known train embeddings: {train_emb.shape} across "
        f"{len(np.unique(train_labels))} classes"
    )
    print(
        f"Open-set test embeddings: {test_emb.shape}  "
        f"(unfamiliar: {int(y_true.sum())}, familiar: {int((1 - y_true).sum())})"
    )

    # Standardize (used by SVM + Isolation Forest only).
    scaler = StandardScaler().fit(train_emb)
    train_scaled = scaler.transform(train_emb)
    test_scaled = scaler.transform(test_emb)

    scores = {}  # raw per-sample scores, kept for ROC plotting

    print("\nScoring: MSP baseline ...")
    scores["MSP"] = msp_scores(test_emb, W, b)

    print("Scoring: Energy (-logsumexp logits) ...")
    scores["Energy"] = energy_scores(test_emb, W, b)

    print("Scoring: Mahalanobis (shared Ledoit-Wolf covariance) ...")
    scores["Mahalanobis"] = mahalanobis_scores(train_emb, train_labels, test_emb)

    print("Scoring: Mahalanobis on L2-normalized embeddings ...")
    scores["Mahalanobis (L2-norm)"] = mahalanobis_scores(
        train_emb, train_labels, test_emb, normalize=True
    )

    print("Scoring: kNN distance on L2-normalized embeddings ...")
    scores["kNN (L2-norm)"] = knn_scores(train_emb, test_emb, k=5, normalize=True)

    print("Scoring: One-Class SVM ...")
    scores["One-Class SVM"] = ocsvm_scores(train_scaled, test_scaled)

    print("Scoring: Isolation Forest ...")
    scores["Isolation Forest"] = isolation_forest_scores(train_scaled, test_scaled)

    results = {name: evaluate_scorer(y_true, s) for name, s in scores.items()}

    print_comparison_table(results)

    if roc_path is None:
        roc_path = os.path.join(DEFAULT_OUTPUT_DIR, "roc_curves.png")
    plot_roc_curves(y_true, scores, results, roc_path)

    return results


def plot_roc_curves(y_true, scores, results, out_path):
    """Save one ROC-curve figure with every scorer, ordered as in the table."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping ROC plot.")
        return

    plt.figure(figsize=(7, 7))
    for name in SCORER_ORDER:
        if name not in scores:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores[name])
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC {results[name]['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Open-Set Detection ROC Curves\n(positive class = 'unfamiliar')")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Open-set recognition scoring for PlantNet embeddings."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS),
        default="resnet18",
        help="Which backbone's embeddings/weights to score (default: resnet18).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory with the .npz embedding files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--weights",
        help="Override path to the fine-tuned checkpoint (.pth).",
    )
    parser.add_argument(
        "--train-file", help="Override known-train .npz filename."
    )
    parser.add_argument(
        "--test-file", help="Override open-set test .npz filename."
    )
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    weights = args.weights or os.path.join(SCRIPT_DIR, preset["weights"])
    train_file = args.train_file or preset["train_file"]
    test_file = args.test_file or preset["test_file"]

    print(f"=== Scoring {args.model} embeddings ===")
    data = load_real_data(args.output_dir, weights, train_file, test_file)
    run(data, roc_path=os.path.join(args.output_dir, preset["roc_file"]))


if __name__ == "__main__":
    main()
