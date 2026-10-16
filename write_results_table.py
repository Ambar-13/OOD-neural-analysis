"""
Regenerate the results tables from the committed result files, so that no
number in the README is typed by hand.

    python write_results_table.py

Reads  analysis_results/results_summary.json
       analysis_results/ood_testing_comprehensive/summary_comprehensive.json
Writes RESULTS_original_run.md
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent / "analysis_results"
OUT = Path(__file__).parent / "RESULTS_original_run.md"


def main():
    base = json.loads((ROOT / "results_summary.json").read_text())
    ood = json.loads((ROOT / "ood_testing_comprehensive" / "summary_comprehensive.json").read_text())

    lines = ["# Results of the original run (December 2025), generated from the committed JSON", ""]
    lines.append("Do not edit by hand. `python write_results_table.py` rewrites this file.")
    lines.append("")
    lines.append("These numbers come from the December 2025 pipeline (see CHANGELOG.md for how it differs from the "
                 "current one). They are kept so the record can be checked.")
    lines.append("")

    np_ = base["neural_predictivity"]
    st = base["statistical_tests"]["predictivity_comparison"]
    lines += ["## Baseline predictivity, 5-fold CV over all 118 images", "",
              "| Model | Median r | Mean r |", "|---|---|---|",
              f"| ViT-B/16 (torchvision, in1k) | {np_['vit']['median_predictivity']:.3f} | {np_['vit']['mean_predictivity']:.3f} |",
              f"| ResNet-34 (torchvision, in1k) | {np_['resnet']['median_predictivity']:.3f} | {np_['resnet']['mean_predictivity']:.3f} |",
              "",
              f"Paired t-test over neurons: p = {st['paired_t_test']['p']:.2f}; Wilcoxon p = {st['wilcoxon']['p']:.2f}; "
              f"Cohen's d = {st['cohens_d']:.2f}. The two models are not distinguishable on this data.",
              "",
              f"The file also records a 'noise ceiling' of {base['noise_ceilings']['full']['lower']:.3f}, computed by "
              "splitting neurons rather than trials; it is not a per-neuron ceiling and is not used.", ""]

    rdm = base["rdm_correlations"]
    lines += ["## RDM comparison (Spearman, upper triangles)", "",
              "| Pair | rho | p |", "|---|---|---|",
              f"| ViT vs neural | {rdm['vit_neural']['correlation']:.3f} | {rdm['vit_neural']['p_value']:.2f} |",
              f"| ResNet vs neural | {rdm['resnet_neural']['correlation']:.3f} | {rdm['resnet_neural']['p_value']:.2f} |",
              f"| ViT vs ResNet | {rdm['vit_resnet']['correlation']:.3f} | {rdm['vit_resnet']['p_value']:.1e} |",
              "", "Neither final-layer RDM is related to the V1 RDM.", ""]

    meta = ood["metadata"]
    lines += ["## Out-of-distribution run (single split)", "",
              f"Train {meta['n_train']} images, test {meta['n_test']} images, {meta['n_neurons']} neurons, "
              f"{meta['n_boot']} bootstrap resamples over neurons.", "",
              f"Baseline median r on the 24 test images: ViT {ood['baseline']['vit_r']:.4f}, ResNet {ood['baseline']['resnet_r']:.4f}.",
              "", f"{len(ood['ood_results'])} conditions. Retention = per-neuron OOD r over baseline r, for neurons with baseline r > 0.01; "
              "median over those neurons, with a bootstrap CI over neurons (not over images).", "",
              "| Condition | ViT r | ResNet r | ViT retention % [CI] | ResNet retention % [CI] | neurons |",
              "|---|---|---|---|---|---|"]
    worst = {"vit": [], "resnet": []}
    for name, v in ood["ood_results"].items():
        lines.append(f"| {name} | {v['vit_r']:.3f} | {v['resnet_r']:.3f} | {v['vit_retention']:.0f} [{v['vit_ci'][0]:.0f}, {v['vit_ci'][1]:.0f}] "
                     f"| {v['resnet_retention']:.0f} [{v['resnet_ci'][0]:.0f}, {v['resnet_ci'][1]:.0f}] | {v['n_valid']} |")
        worst["vit"].append((v["vit_retention"], name))
        worst["resnet"].append((v["resnet_retention"], name))
    vit_r = sorted(worst["vit"]); res_r = sorted(worst["resnet"])
    lines += ["", "### What the table actually shows", "",
              f"- ViT retention ranges from {vit_r[0][0]:.0f}% ({vit_r[0][1]}) to {vit_r[-1][0]:.0f}% ({vit_r[-1][1]}). "
              f"{sum(1 for r, _ in vit_r if r < 50)} conditions are below 50%.",
              f"- ResNet retention ranges from {res_r[0][0]:.0f}% ({res_r[0][1]}) to {res_r[-1][0]:.0f}% ({res_r[-1][1]}). "
              f"{sum(1 for r, _ in res_r if r < 50)} conditions are below 50%.",
              "- With 24 test images the standard error of a single r near 0.13 is about 0.2, so the retention values "
              "are ratios of two numbers within their own noise. The bootstrap intervals resample neurons and do not "
              "include image-sampling noise.", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
