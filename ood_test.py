"""
Exploratory out-of-distribution test on the Allen natural scenes.

    python ood_test.py [--splits 10] [--test-fraction 0.2]

Runs after neural_alignment.py, which saves the kept neurons' trials, the
images, and the best layer per model.

Read this before trusting any number it prints: 118 images is too few for
this test. With about 24 held-out images the standard error of a single
correlation is about 0.2, which is larger than the typical score. So this
script does the honest version of the test and reports its uncertainty:

- repeated random train/test splits (default 10), so the baseline and every
  OOD condition carry a confidence interval over image splits, which is where
  the noise actually lives;
- the best layer per model, not the final layer;
- ceiling-normalised r², median over neurons, per split;
- retention = median OOD score over median in-distribution score, per split,
  then mean and 95 percent interval over splits.

The protocol differs from Madan et al. 2024 in two ways that matter: their
shifts are natural (hold out the images whose contrast, hue, etc. is in the
top or bottom quarter of a large natural set) and their area is macaque IT.
Synthetic corruptions of grayscale mouse-V1 stimuli are a different and weaker
test, and the README says so.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import encoding
from features import extract_all_layers, load_models
from perturbations import build_conditions

OUT = Path("results")


def fit_and_score(x_train, y_train, x_test, y_test, reliability):
    scaler = StandardScaler().fit(x_train)
    model = RidgeCV(alphas=encoding.RIDGE_ALPHAS, alpha_per_target=True).fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    r = np.array([encoding.score_pair(y_test[:, j], pred[:, j]) for j in range(y_test.shape[1])])
    return float(np.nanmedian(encoding.normalised_r2(r, reliability)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    alignment = json.loads((OUT / "neural_alignment.json").read_text())
    best_layer = alignment["rdm"]["best_layer"]
    trials = np.load(OUT / "responses_trials.npy")
    images = np.load(OUT / "images.npy")
    responses = np.nanmean(trials, axis=1)
    reliability = encoding.split_half_reliability(trials, seed=args.seed)
    n_images = len(images)
    n_test = int(round(n_images * args.test_fraction))
    print(f"{n_images} images, {responses.shape[1]} neurons, {args.splits} splits of {n_images - n_test}/{n_test}")
    print("Warning: with this few test images every number below has a wide interval. Read the intervals, not the means.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    rng = np.random.RandomState(args.seed)
    conditions = build_conditions(images, rng)
    print(f"{len(conditions)} perturbation conditions")

    # features for clean and perturbed images at the best layer, computed once
    feats = {}
    for name, model in models.items():
        stage = best_layer[name]
        clean = extract_all_layers(name, model, images, device)[stage]
        proj = encoding.SparseRandomProjection(n_components=4096, random_state=args.seed).fit(clean) \
            if clean.shape[1] > 4096 else None
        reduce = (lambda f: proj.transform(f).astype(np.float32)) if proj else (lambda f: f.astype(np.float32))
        feats[name] = {"clean": reduce(clean)}
        for cond, imgs in conditions.items():
            feats[name][cond] = reduce(extract_all_layers(name, model, imgs, device)[stage])
        print(f"{name}: features at {stage} for clean + {len(conditions)} conditions")

    scores = {name: {"clean": [], **{c: [] for c in conditions}} for name in models}
    for s in range(args.splits):
        perm = np.random.RandomState(args.seed + s).permutation(n_images)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        for name in models:
            x_train = feats[name]["clean"][train_idx]
            for cond in ["clean"] + list(conditions):
                x_test = feats[name][cond][test_idx]
                scores[name][cond].append(fit_and_score(x_train, responses[train_idx], x_test, responses[test_idx], reliability))

    out = {"metadata": {"n_images": n_images, "n_test": n_test, "n_splits": args.splits,
                        "n_neurons": int(responses.shape[1]), "best_layer": best_layer},
           "conditions": {}}
    for name in models:
        base = np.array(scores[name]["clean"])
        out["metadata"][f"{name}_baseline_normalised_r2"] = {"mean": float(base.mean()), "ci": ci(base)}
    for cond in conditions:
        out["conditions"][cond] = {}
        for name in models:
            base = np.array(scores[name]["clean"])
            ood = np.array(scores[name][cond])
            retention = ood / np.where(base > 0, base, np.nan)
            out["conditions"][cond][name] = {"score_mean": float(ood.mean()), "score_ci": ci(ood),
                                             "retention_mean": float(np.nanmean(retention)), "retention_ci": ci(retention)}
    (OUT / "ood_test.json").write_text(json.dumps(out, indent=2))
    plot(out, list(models))
    print("Saved results/ood_test.json and results/ood_retention.png")


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [None, None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def plot(out, names):
    conds = list(out["conditions"])
    fig, ax = plt.subplots(figsize=(max(10, len(conds) * 0.3), 5))
    x = np.arange(len(conds))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        means = [out["conditions"][c][name]["retention_mean"] for c in conds]
        lo = [m - (out["conditions"][c][name]["retention_ci"][0] or m) for m, c in zip(means, conds)]
        hi = [(out["conditions"][c][name]["retention_ci"][1] or m) - m for m, c in zip(means, conds)]
        ax.bar(x + i * w, means, w, yerr=[lo, hi], capsize=2, label=name)
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(conds, rotation=90, fontsize=6)
    ax.set_ylabel("retention of ceiling-normalised r² (OOD / clean), 95% interval over splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ood_retention.png", dpi=200)


if __name__ == "__main__":
    main()
