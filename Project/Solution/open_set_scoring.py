"""
Open-set recognition scoring for the PlantNet biodiversity screening project.

Consumes the embedding files that ``main_cnn_analysis.py`` writes into
``open_set_outputs/`` plus the classifier head in ``resnet18_finetuned_knowns.pth``,
then builds several "is this an unfamiliar species?" scorers (MSP, Energy,
Mahalanobis, kNN, One-Class SVM, Isolation Forest) and compares them. The CNN is
never retrained.

Convention: y_true = 1 - is_known, so 1 == "unfamiliar" is the positive class,
and every scorer is oriented so a HIGHER score means MORE unfamiliar.

Run: python open_set_scoring.py
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
DEFAULT_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "resnet18_finetuned_knowns.pth")


# ==========================================================================
# Data loading
# ==========================================================================
def load_real_data(output_dir, weights_path):
    """Load both npz embedding files and the classifier head (W, b) from disk."""
    train_path = os.path.join(output_dir, "known_train_embeddings.npz")
    test_path = os.path.join(output_dir, "open_set_test_embeddings.npz")

    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected embedding file not found: {path}\n"
                "Run main_cnn_analysis.py first to generate the embeddings."
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


def _load_classifier_head(weights_path):
    """Rebuild the linear head (W, b) from the .pth so logits can be recomputed.

    Logits are not saved at runtime. Torch is imported lazily so it is only
    needed when this actually runs.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Classifier weights not found: {weights_path}\n"
            "The MSP baseline needs backbone.fc.weight / backbone.fc.bias."
        )

    import torch

    state_dict = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    W = state_dict["backbone.fc.weight"].cpu().numpy().astype(np.float64)
    b = state_dict["backbone.fc.bias"].cpu().numpy().astype(np.float64)
    return W, b


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
    covariance (per-class covariances are singular in 512-D). ``normalize=True``
    L2-normalizes first for cosine geometry.
    """
    if normalize:
        train_emb, test_emb = _l2_normalize(train_emb), _l2_normalize(test_emb)

    means = _class_means(train_emb, train_labels)
    residuals = train_emb - means[np.searchsorted(np.unique(train_labels), train_labels)]
    precision = LedoitWolf().fit(residuals).precision_

    return cdist(test_emb, means, metric="mahalanobis", VI=precision).min(axis=1)


def knn_scores(train_emb, test_emb, k=5, normalize=True):
    """Mean distance to the k nearest known-training embeddings (L2-norm by default)."""
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
def run(data):
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing the .npz embedding files "
        f"(default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to resnet18_finetuned_knowns.pth "
        f"(default: {DEFAULT_WEIGHTS_PATH}).",
    )
    args = parser.parse_args()

    print("=== Loading embeddings from disk ===")
    data = load_real_data(args.output_dir, args.weights)
    run(data)


if __name__ == "__main__":
    main()
