"""
Which stage of ViT-S/16 and of ResNet-34 best predicts mouse V1, and how close to the ceiling?

    python neural_alignment.py [--reliability 0.5] [--experiment-id N]

Loads one Allen natural-scenes experiment with every trial, keeps neurons whose
split-half reliability is at least --reliability, fits cross-validated ridge from
every stage of each model (random-projected to 4096 dims), and reports per-stage
median r and ceiling-normalised r². At each model's best stage the model RDM is
compared with the neural RDM and its ceiling. Writes results/.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import encoding
from allen_data import load_experiment
from features import WEIGHTS, extract_all_layers, load_models

OUT = Path("results")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reliability", type=float, default=0.5,
                        help="minimum Spearman-Brown split-half reliability to keep a neuron")
    parser.add_argument("--experiment-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_experiment(experiment_id=args.experiment_id)
    images, trials, responses = data["images"], data["responses_trials"], data["responses"]
    print(f"Experiment {data['metadata']['experiment_id']}: {responses.shape[1]} neurons, "
          f"{responses.shape[0]} images, up to {trials.shape[1]} trials, cre line {data['metadata']['cre_line']}")

    reliability = encoding.split_half_reliability(trials, seed=args.seed)
    keep = encoding.select_reliable_neurons(reliability, args.reliability)
    print(f"Neurons with reliability >= {args.reliability}: {len(keep)} of {len(reliability)} "
          f"(median reliability among kept: {np.median(reliability[keep]):.3f})")
    if len(keep) < 10:
        raise SystemExit("Too few reliable neurons to say anything; lower --reliability or pick another experiment.")
    responses, trials, reliability = responses[:, keep], trials[:, :, keep], reliability[keep]
    np.save(OUT / "responses_trials.npy", trials)
    np.save(OUT / "images.npy", images)

    models = load_models(device)
    results = {"metadata": data["metadata"], "weights": WEIGHTS, "reliability_threshold": args.reliability,
               "n_neurons_kept": int(len(keep)), "median_ceiling": float(np.median(reliability)), "layers": {}}
    best = {}
    for name, model in models.items():
        print(f"\n{name} ({WEIGHTS[name]})")
        layer_feats = extract_all_layers(name, model, images, device)
        results["layers"][name] = {}
        for stage, feats in layer_feats.items():
            reduced = encoding.reduce_features(feats, seed=args.seed)
            scores = encoding.cv_ridge_scores(reduced, responses, seed=args.seed)
            summary = encoding.summarise(scores["r"], reliability)
            summary["n_features_raw"] = int(feats.shape[1])
            results["layers"][name][stage] = summary
            print(f"  {stage:12s} dims {feats.shape[1]:>7d}  median r {summary['median_r']:.3f}  "
                  f"median normalised r2 {summary['median_normalised_r2']:.3f}")
            np.save(OUT / f"features_{name}_{stage}.npy", reduced)
        best_stage = max(results["layers"][name], key=lambda s: results["layers"][name][s]["median_normalised_r2"])
        best[name] = best_stage
        print(f"  best layer: {best_stage}")

    # RDMs at the best layer, against the neural RDM and its ceiling
    neural_rdm = encoding.compute_rdm(responses)
    rdm_ceiling = encoding.rdm_ceiling(trials, seed=args.seed)
    results["rdm"] = {"neural_rdm_ceiling": rdm_ceiling, "best_layer": best}
    for name in models:
        feats = np.load(OUT / f"features_{name}_{best[name]}.npy")
        rho, p = encoding.compare_rdms(encoding.compute_rdm(feats), neural_rdm)
        results["rdm"][name] = {"rho": rho, "p": p, "rho_over_ceiling": rho / rdm_ceiling if rdm_ceiling > 0 else None}
        print(f"RDM {name} @ {best[name]}: rho {rho:.3f} (p {p:.2g}), ceiling {rdm_ceiling:.3f}")

    (OUT / "neural_alignment.json").write_text(json.dumps(results, indent=2, default=float))
    plot_layers(results)
    print(f"\nSaved results/neural_alignment.json and results/layer_sweep.png")


def plot_layers(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, layers in results["layers"].items():
        stages = list(layers)
        depth = np.linspace(0, 1, len(stages))
        ax.plot(depth, [layers[s]["median_normalised_r2"] for s in stages], marker="o", label=name)
    ax.set_xlabel("relative depth (0 = first stage, 1 = last)")
    ax.set_ylabel("median ceiling-normalised r²")
    ax.set_title("V1 predictivity by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "layer_sweep.png", dpi=200)


if __name__ == "__main__":
    main()
