"""
Extended evaluation for the PlantNet open-set project.

Covers the evaluation commitments from the project proposal that the headline
comparison in ``open_set_scoring.py`` does not address:

  A. Closed-set metrics on the held-out known species: top-1 / top-5 accuracy
     and micro- vs macro-averaged precision / recall / F1. The micro-macro gap
     is the long-tail diagnostic promised in the proposal.
  B. Detection difficulty vs taxonomic distance: unknown species are split by
     whether their genus was seen during training, testing the proposal's
     hypothesis that same-genus novelties are the hard cases.
  C. Bootstrap confidence intervals for ROC AUC, resampling whole species
     rather than images so the interval reflects "which species happened to be
     held out" rather than only image-level noise.
  D. Test-time openness sweep: AUC as a function of how many distinct unknown
     species appear in the evaluation stream.

Scoring the seven detectors is expensive (the One-Class SVM fit dominates), so
per-sample scores are computed once and cached to
``open_set_outputs/scores_<model>.npz``. Every analysis above is then a cheap
re-slicing of that cache. Delete the cache to force recomputation.

Run: python3 extended_evaluation.py [--model resnet18|efficientnet|both]
"""

import argparse
import os

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

import open_set_scoring as oss

BOOTSTRAP_ROUNDS = 2000
OPENNESS_DRAWS = 200
OPENNESS_GRID = [5, 10, 20, 30, 45]
RNG_SEED = 42


# ==========================================================================
# Score cache
# ==========================================================================
def compute_all_scores(data):
    """Compute every scorer's per-sample unfamiliarity scores (higher == more
    unfamiliar). Mirrors ``open_set_scoring.run`` so the numbers agree."""
    train_emb = data["train_emb"]
    train_labels = data["train_labels"]
    test_emb = data["test_emb"]

    scaler = StandardScaler().fit(train_emb)
    train_scaled = scaler.transform(train_emb)
    test_scaled = scaler.transform(test_emb)

    scores = {}
    print("  MSP ...", flush=True)
    scores["MSP"] = oss.msp_scores(test_emb, data["W"], data["b"])
    print("  Energy ...", flush=True)
    scores["Energy"] = oss.energy_scores(test_emb, data["W"], data["b"])
    print("  Mahalanobis ...", flush=True)
    scores["Mahalanobis"] = oss.mahalanobis_scores(train_emb, train_labels, test_emb)
    print("  Mahalanobis (L2-norm) ...", flush=True)
    scores["Mahalanobis (L2-norm)"] = oss.mahalanobis_scores(
        train_emb, train_labels, test_emb, normalize=True
    )
    print("  kNN (L2-norm) ...", flush=True)
    scores["kNN (L2-norm)"] = oss.knn_scores(train_emb, test_emb, k=5, normalize=True)
    print("  One-Class SVM (slow) ...", flush=True)
    scores["One-Class SVM"] = oss.ocsvm_scores(train_scaled, test_scaled)
    print("  Isolation Forest ...", flush=True)
    scores["Isolation Forest"] = oss.isolation_forest_scores(train_scaled, test_scaled)
    return scores


def load_or_compute_scores(data, cache_path):
    """Return the score dict, reading the cache when it exists."""
    if os.path.exists(cache_path):
        print(f"Loading cached scores from {os.path.basename(cache_path)}")
        npz = np.load(cache_path, allow_pickle=True)
        return {name: npz[name] for name in npz.files}

    print("No score cache found; computing scores (this takes a while):")
    scores = compute_all_scores(data)
    np.savez_compressed(cache_path, **scores)
    print(f"Cached scores to {os.path.basename(cache_path)}")
    return scores


# ==========================================================================
# A. Closed-set metrics on known test images
# ==========================================================================
def closed_set_metrics(test_emb, class_labels, is_known, W, b):
    """Top-1 / top-5 accuracy and micro/macro P/R/F1 over the known species.

    Unknown-species rows carry a sentinel label of -1 and are excluded, since
    closed-set accuracy is only defined for images of the known classes.
    """
    mask = is_known == 1
    logits = test_emb[mask] @ W.T + b
    y = class_labels[mask]
    n_classes = W.shape[0]

    top1 = top_k_accuracy_score(y, logits, k=1, labels=np.arange(n_classes))
    top5 = top_k_accuracy_score(y, logits, k=5, labels=np.arange(n_classes))

    y_pred = logits.argmax(axis=1)
    micro = precision_recall_fscore_support(
        y, y_pred, average="micro", zero_division=0
    )
    macro = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )

    return {
        "n_images": int(mask.sum()),
        "n_classes_present": int(len(np.unique(y))),
        "top1": top1,
        "top5": top5,
        "micro": micro[:3],
        "macro": macro[:3],
    }


def print_closed_set(name, m):
    print(f"\n--- Closed-set performance on held-out known species ({name}) ---")
    print(f"Images: {m['n_images']}   distinct known classes present: "
          f"{m['n_classes_present']}")
    print(f"Top-1 accuracy: {m['top1']:.4f}")
    print(f"Top-5 accuracy: {m['top5']:.4f}")
    print(f"{'Averaging':<10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for label, vals in (("Micro", m["micro"]), ("Macro", m["macro"])):
        print(f"{label:<10}{vals[0]:>11.4f}{vals[1]:>9.4f}{vals[2]:>8.4f}")
    gap = m["micro"][2] - m["macro"][2]
    print(f"Micro-macro F1 gap: {gap:+.4f}  "
          "(positive => rare classes underperform)")


# ==========================================================================
# B. Taxonomic distance
# ==========================================================================
def genus_of(species_name):
    """Genus is the first whitespace-delimited token of the scientific name."""
    return str(species_name).split()[0]


def taxonomic_breakdown(scores, is_known, test_species, known_species):
    """AUC restricted to same-genus vs novel-genus unknown species.

    Known-species images are the negative class in both cases, so the two AUCs
    are directly comparable and differ only in which novelties are scored.
    """
    known_genera = {genus_of(s) for s in np.unique(known_species)}
    shares_genus = np.array([genus_of(s) in known_genera for s in test_species])

    known_mask = is_known == 1
    shared_mask = (is_known == 0) & shares_genus
    novel_mask = (is_known == 0) & ~shares_genus

    unknown_species = np.unique(test_species[is_known == 0])
    n_shared_sp = sum(1 for s in unknown_species if genus_of(s) in known_genera)

    out = {
        "n_shared_species": n_shared_sp,
        "n_novel_species": len(unknown_species) - n_shared_sp,
        "n_shared_images": int(shared_mask.sum()),
        "n_novel_images": int(novel_mask.sum()),
        "per_scorer": {},
    }

    for name in oss.SCORER_ORDER:
        if name not in scores:
            continue
        s = np.asarray(scores[name])
        auc_shared = roc_auc_score(
            np.r_[np.zeros(known_mask.sum()), np.ones(shared_mask.sum())],
            np.r_[s[known_mask], s[shared_mask]],
        )
        auc_novel = roc_auc_score(
            np.r_[np.zeros(known_mask.sum()), np.ones(novel_mask.sum())],
            np.r_[s[known_mask], s[novel_mask]],
        )
        out["per_scorer"][name] = (auc_shared, auc_novel)
    return out


def print_taxonomic(name, t):
    print(f"\n--- Detection difficulty vs taxonomic distance ({name}) ---")
    print(f"Same-genus-as-known unknowns: {t['n_shared_species']} species, "
          f"{t['n_shared_images']} images")
    print(f"Novel-genus unknowns:         {t['n_novel_species']} species, "
          f"{t['n_novel_images']} images")
    width = max(len(n) for n in t["per_scorer"]) + 2
    print(f"{'Scorer':<{width}}{'AUC same-genus':>16}{'AUC novel-genus':>17}"
          f"{'Delta':>9}")
    for scorer, (a_shared, a_novel) in t["per_scorer"].items():
        print(f"{scorer:<{width}}{a_shared:>16.4f}{a_novel:>17.4f}"
              f"{a_novel - a_shared:>+9.4f}")


# ==========================================================================
# C. Species-level bootstrap confidence intervals
# ==========================================================================
def species_bootstrap_ci(scores, is_known, test_species, rounds=BOOTSTRAP_ROUNDS,
                         seed=RNG_SEED):
    """Percentile CI for AUC, resampling whole species with replacement.

    Image-level bootstrap would understate uncertainty here: the dominant
    source of variance is which species landed in the held-out set, and all
    images of one species share that draw. Resampling species (a cluster
    bootstrap) propagates it.
    """
    rng = np.random.default_rng(seed)

    known_sp = np.unique(test_species[is_known == 1])
    unknown_sp = np.unique(test_species[is_known == 0])
    # Row indices grouped by species, so a resampled species pulls all its images.
    idx_by_species = {sp: np.flatnonzero(test_species == sp)
                      for sp in np.r_[known_sp, unknown_sp]}

    # Draw the resamples once and reuse them for every scorer, so the resulting
    # intervals are paired and differences between scorers are not inflated by
    # using different resamples for each.
    draws = []
    for _ in range(rounds):
        k_idx = np.concatenate([
            idx_by_species[sp]
            for sp in rng.choice(known_sp, size=len(known_sp), replace=True)
        ])
        u_idx = np.concatenate([
            idx_by_species[sp]
            for sp in rng.choice(unknown_sp, size=len(unknown_sp), replace=True)
        ])
        y = np.r_[np.zeros(len(k_idx)), np.ones(len(u_idx))]
        draws.append((np.r_[k_idx, u_idx], y))

    out = {}
    for name in oss.SCORER_ORDER:
        if name not in scores:
            continue
        s = np.asarray(scores[name])
        aucs = np.array([roc_auc_score(y, s[idx]) for idx, y in draws])
        out[name] = (aucs.mean(), *np.percentile(aucs, [2.5, 97.5]))
    return out


def print_bootstrap(name, ci):
    print(f"\n--- Species-level bootstrap 95% CI for ROC AUC ({name}, "
          f"{BOOTSTRAP_ROUNDS} rounds) ---")
    width = max(len(n) for n in ci) + 2
    print(f"{'Scorer':<{width}}{'Mean AUC':>10}{'95% CI':>22}{'Width':>9}")
    for scorer, (mean, lo, hi) in ci.items():
        print(f"{scorer:<{width}}{mean:>10.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}"
              f"{hi - lo:>9.4f}")


# ==========================================================================
# D. Test-time openness sweep
# ==========================================================================
def openness_sweep(scores, is_known, test_species, scorer,
                   grid=OPENNESS_GRID, draws=OPENNESS_DRAWS, seed=RNG_SEED):
    """AUC vs the number of distinct unknown species present at test time.

    The known negatives are held fixed and unknown species are subsampled, so
    this varies openness of the *evaluation stream* only. It does not vary
    which species the network was trained on, which would require retraining.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(scores[scorer])

    known_idx = np.flatnonzero(is_known == 1)
    unknown_sp = np.unique(test_species[is_known == 0])
    idx_by_species = {sp: np.flatnonzero(test_species == sp) for sp in unknown_sp}

    results = {}
    for m in grid:
        m = min(m, len(unknown_sp))
        aucs = np.empty(draws)
        for d in range(draws):
            picked = rng.choice(unknown_sp, size=m, replace=False)
            u_idx = np.concatenate([idx_by_species[sp] for sp in picked])
            y = np.r_[np.zeros(len(known_idx)), np.ones(len(u_idx))]
            aucs[d] = roc_auc_score(y, np.r_[s[known_idx], s[u_idx]])
        results[m] = (aucs.mean(), aucs.std())
    return results


def print_openness(name, scorer, sweep):
    print(f"\n--- Test-time openness sweep ({name}, {scorer}) ---")
    print(f"{'Unknown species':>16}{'Mean AUC':>10}{'Std':>9}")
    for m, (mean, sd) in sweep.items():
        print(f"{m:>16}{mean:>10.4f}{sd:>9.4f}")


def plot_openness(sweeps, scorer, out_path):
    """One figure overlaying the openness curve for each backbone."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping openness plot.")
        return

    plt.figure(figsize=(7, 4.5))
    for label, sweep in sweeps.items():
        ms = sorted(sweep)
        means = np.array([sweep[m][0] for m in ms])
        sds = np.array([sweep[m][1] for m in ms])
        plt.errorbar(ms, means, yerr=sds, marker="o", capsize=3, lw=1.8,
                     label=label)

    plt.xlabel("Number of distinct unknown species in the evaluation set")
    plt.ylabel("ROC AUC")
    plt.title(f"Test-time openness sensitivity ({scorer})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved openness figure to {out_path}")


# ==========================================================================
# Main
# ==========================================================================
BEST_SCORER = "kNN (L2-norm)"

DISPLAY_NAMES = {"resnet18": "ResNet-18", "efficientnet": "EfficientNet-B0"}


def evaluate_model(model, output_dir):
    preset = oss.MODEL_PRESETS[model]
    weights = os.path.join(oss.SCRIPT_DIR, preset["weights"])

    print(f"\n{'=' * 70}\n=== {DISPLAY_NAMES[model]} ===\n{'=' * 70}")
    data = oss.load_real_data(
        output_dir, weights, preset["train_file"], preset["test_file"]
    )

    test_npz = np.load(os.path.join(output_dir, preset["test_file"]),
                       allow_pickle=True)
    train_npz = np.load(os.path.join(output_dir, preset["train_file"]),
                        allow_pickle=True)
    test_species = test_npz["species_name"].astype(str)
    known_species = train_npz["species_name"].astype(str)
    class_labels = test_npz["class_labels"].astype(int)
    is_known = data["test_is_known"]

    cache_path = os.path.join(output_dir, f"scores_{model}.npz")
    scores = load_or_compute_scores(data, cache_path)

    name = DISPLAY_NAMES[model]

    cs = closed_set_metrics(data["test_emb"], class_labels, is_known,
                            data["W"], data["b"])
    print_closed_set(name, cs)

    tax = taxonomic_breakdown(scores, is_known, test_species, known_species)
    print_taxonomic(name, tax)

    ci = species_bootstrap_ci(scores, is_known, test_species)
    print_bootstrap(name, ci)

    sweep = openness_sweep(scores, is_known, test_species, BEST_SCORER)
    print_openness(name, BEST_SCORER, sweep)

    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet18", "efficientnet", "both"],
                        default="both")
    parser.add_argument("--output-dir", default=oss.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    models = (["resnet18", "efficientnet"] if args.model == "both"
              else [args.model])

    sweeps = {}
    for model in models:
        sweeps[DISPLAY_NAMES[model]] = evaluate_model(model, args.output_dir)

    plot_openness(sweeps, BEST_SCORER,
                  os.path.join(args.output_dir, "openness_sweep.png"))


if __name__ == "__main__":
    main()
