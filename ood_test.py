"""
Does a flexible readout keep its score under image corruptions as well as a strict one?

    python ood_test.py [--splits 10] [--test-fraction 0.2]

Runs after neural_alignment.py. For each model, each readout (ridge on all
features, on 100, 20 and 5 components, and RSA) and each corruption, reports
the in-distribution score and the retention (corrupted score / clean score)
over ten random train/test splits. Regression readouts are scored as median
ceiling-normalised r² over neurons, RSA as Spearman rho between RDMs.

With 118 images the intervals are wide; this is a pilot of the analysis.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.random_projection import SparseRandomProjection

import encoding
from features import extract_all_layers, load_models
from perturbations import build_conditions

OUT = Path("results")


def regression_score(readout, x_train, y_train, x_test, y_test, reliability):
    pred = readout.fit(x_train, y_train).predict(x_test)
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
    print("With this few test images the intervals are wide.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    rng = np.random.RandomState(args.seed)
    conditions = build_conditions(images, rng)
    print(f"{len(conditions)} perturbation conditions")

    feats = {}
    for name, model in models.items():
        stage = best_layer[name]
        clean = extract_all_layers(name, model, images, device)[stage]
        proj = SparseRandomProjection(n_components=4096, random_state=args.seed).fit(clean) if clean.shape[1] > 4096 else None
        reduce = (lambda f: proj.transform(f).astype(np.float32)) if proj else (lambda f: f.astype(np.float32))
        feats[name] = {"clean": reduce(clean)}
        for cond, imgs in conditions.items():
            feats[name][cond] = reduce(extract_all_layers(name, model, imgs, device)[stage])
        print(f"{name}: features at {stage} for clean and {len(conditions)} conditions")

    readout_names = [r.name for r in encoding.regression_readouts()] + ["rsa"]
    cond_names = ["clean"] + list(conditions)
    scores = {m: {r: {c: [] for c in cond_names} for r in readout_names} for m in models}

    for s in range(args.splits):
        perm = np.random.RandomState(args.seed + s).permutation(n_images)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        y_train, y_test = responses[train_idx], responses[test_idx]
        for m in models:
            x_train = feats[m]["clean"][train_idx]
            for cond in cond_names:
                x_test = feats[m][cond][test_idx]
                for readout in encoding.regression_readouts():
                    scores[m][readout.name][cond].append(
                        regression_score(readout, x_train, y_train, x_test, y_test, reliability))
                scores[m]["rsa"][cond].append(encoding.rsa_score(x_test, y_test))

    out = {"metadata": {"n_images": n_images, "n_test": n_test, "n_splits": args.splits,
                        "n_neurons": int(responses.shape[1]), "best_layer": best_layer,
                        "readouts": readout_names},
           "in_distribution": {}, "retention": {}}
    for m in models:
        out["in_distribution"][m] = {}
        out["retention"][m] = {}
        for r in readout_names:
            base = np.array(scores[m][r]["clean"])
            out["in_distribution"][m][r] = {"mean": float(base.mean()), "ci": ci(base)}
            out["retention"][m][r] = {}
            for cond in conditions:
                ood = np.array(scores[m][r][cond])
                ret = ood / np.where(base > 0, base, np.nan)
                out["retention"][m][r][cond] = {"mean": float(np.nanmean(ret)), "ci": ci(ret)}
            all_ret = np.array([out["retention"][m][r][c]["mean"] for c in conditions])
            out["retention"][m][r]["median_over_conditions"] = float(np.nanmedian(all_ret))

    (OUT / "ood_test.json").write_text(json.dumps(out, indent=2))
    print_summary(out, list(models), readout_names)
    plot(out, list(models), readout_names, list(conditions))
    print("Saved results/ood_test.json and results/ood_readouts.png")


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [None, None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def print_summary(out, models, readouts):
    print("\nIn-distribution score and median retention over conditions, per model and readout")
    print(f"{'model':12s} {'readout':12s} {'ID score':>18s} {'median retention':>18s}")
    for m in models:
        for r in readouts:
            idv = out["in_distribution"][m][r]
            ret = out["retention"][m][r]["median_over_conditions"]
            print(f"{m:12s} {r:12s} {idv['mean']:8.3f} [{idv['ci'][0]:.3f}, {idv['ci'][1]:.3f}] {ret:18.2f}")
    pass


def plot(out, models, readouts, conditions):
    fig, axes = plt.subplots(len(models), 1, figsize=(max(10, len(conditions) * 0.3), 4 * len(models)), squeeze=False)
    x = np.arange(len(conditions))
    w = 0.8 / len(readouts)
    for ax, m in zip(axes[:, 0], models):
        for i, r in enumerate(readouts):
            means = [out["retention"][m][r][c]["mean"] for c in conditions]
            lo = [mu - (out["retention"][m][r][c]["ci"][0] or mu) for mu, c in zip(means, conditions)]
            hi = [(out["retention"][m][r][c]["ci"][1] or mu) - mu for mu, c in zip(means, conditions)]
            ax.bar(x + i * w, means, w, yerr=[lo, hi], capsize=1.5, label=r)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{m}: retention of in-distribution score under each corruption, by readout")
        ax.set_ylabel("OOD / clean, 95% interval over splits")
        ax.set_xticks(x + w)
        ax.set_xticklabels(conditions, rotation=90, fontsize=6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ood_readouts.png", dpi=200)


if __name__ == "__main__":
    main()
