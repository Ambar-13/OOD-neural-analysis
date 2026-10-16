# Mouse V1 predictivity of ViT-S/16 and ResNet-34, layer by layer

Ambar, Computer Engineering, National University of Singapore.

## Status (16 October 2026): corrected, not yet rerun

The original version of this repo (December 2025) reported that "ViT shows consistent degradation (70 to 96 percent retention) while ResNet shows high variance (19 to 103 percent)" under distribution shift, and that ViT had "59 percent higher baseline neural predictivity". Those claims do not hold against the repo's own result files, and the pipeline that produced them had design errors. The code has been rewritten; the rerun is pending (it needs the Allen SDK and a GPU). The old numbers are kept in [RESULTS_original_run.md](RESULTS_original_run.md), generated from the committed JSON by `write_results_table.py`, so anyone can check the claims below.

What was wrong, and what changed:

| Problem in the original | What the committed data shows | Fix |
|---|---|---|
| README headline: ViT retention 70 to 96 percent, never fails | ViT at 12 percent (brightness 0.1), 16 percent (heavy blur); six conditions below 50 percent. ResNet at 10 percent on motion blur. | README regenerated from the JSON. |
| "ViT 59 percent higher baseline predictivity" | Median r 0.090 vs 0.057 with paired p = 0.55 and d = 0.06; in the OOD run the baselines were 0.1296 vs 0.1291. | Reported as not distinguishable. |
| Noise ceiling computed by splitting neurons, not trials | "14 percent of ceiling" has no meaning. | Per-neuron split-half across the 50 trials, Spearman-Brown corrected, in `encoding.py`. |
| Final-layer features used to predict V1 | On this dataset the best layer is early to intermediate (Conwell et al. 2021). Final-layer features here score the same as random features under proper cross-validation. | Every stage of each model is swept; the best layer is chosen by ceiling-normalised r². |
| torchvision ViT-B/16 (86M, in1k) vs ResNet-34 (22M, in1k) | Not a matched pair. | timm `vit_small_patch16_224.augreg_in1k` vs `resnet34.a1_in1k`, both 22M and ImageNet-1k only. |
| 100 random neurons, no reliability filter | Unreliable neurons cannot be predicted by anything. | Neurons kept at split-half reliability of at least 0.5 (configurable). |
| One 94/24 split; bootstrap over neurons | Image sampling noise, the dominant noise, was never measured. | Ten random splits; intervals over splits. |
| Madan et al. 2024 described as "models retain ~20 percent" under synthetic corruptions | Their shifts are natural percentile holdouts on macaque IT; 20 percent is one worst case (a ViT, high saturation); best case 66 percent. | README and docstrings corrected. |
| 47 conditions claimed | 53 in the file; hue, saturation and grayscale are identities on grayscale stimuli. | Identity conditions are skipped automatically. |

## What the corrected pipeline asks

1. At which depth does each model best predict mouse V1, and how close to the per-neuron ceiling does it get?
2. Do the two matched models differ at their best depth, with intervals that include image-sampling noise?
3. Exploratory only: how does ceiling-normalised predictivity move under synthetic image corruptions, with intervals over repeated splits?

Question 3 is kept because the code exists, but 118 images cannot support strong OOD claims, and the script says so when it runs. For a real OOD test use Madan's MacaqueITBench (8,233 natural images, public).

## Preliminary check on the committed arrays (16 October 2026, CPU)

The corrected pipeline was run on the 118 images and 100 trial-averaged neurons saved by the original run. No trial data is committed, so there is no ceiling and no reliability filter here; the numbers are raw cross-validated median r over all 100 neurons, image-level 5-fold CV, sparse random projection to 4096 dims, ridge with one penalty per neuron. Random 4096-dim features give median r of -0.145 on this data, which is the no-signal floor for this scorer.

| ViT-S/16 (in1k) stage | median r | neurons with r > 0.2 | ResNet-34 (in1k) stage | median r | neurons with r > 0.2 |
|---|---|---|---|---|---|
| patch_embed | -0.123 | 1 | stem | -0.049 | 11 |
| block00 | 0.079 | 19 | layer1 | -0.009 | 12 |
| block01 | 0.066 | 28 | layer2 | **0.042** | **22** |
| block02 | **0.105** | **33** | layer3 | -0.012 | 9 |
| block03 | 0.104 | 29 | layer4 | -0.085 | 2 |
| block04 | 0.048 | 24 | pool (final) | -0.091 | 3 |
| block05 to block11 | -0.05 to -0.09 | 2 to 6 | | | |
| norm (final) | -0.048 | 8 | | | |

Two things are already visible. The final stages of both models, which the original analysis used, sit at the no-signal floor. The best stages are a quarter of the way into the ViT and at the second of four ResNet stages, which is what Conwell et al. 2021 found for mouse visual cortex across dozens of models. Whether the two models differ at their best stage is not answerable without the ceiling, the reliability filter, and intervals over image splits; that is what the full rerun gives.

## Running it

```bash
./install.sh                      # venv with torch, timm, allensdk (Python 3.9 to 3.11 for allensdk)
source venv/bin/activate
python neural_alignment.py        # downloads one Allen VISp experiment, sweeps layers, writes results/
python ood_test.py                # exploratory OOD on the best layer, repeated splits
python write_results_table.py     # regenerates RESULTS_original_run.md from the old JSON
```

## Files

- `encoding.py`: noise ceiling, neuron selection, sparse random projection, cross-validated ridge, RDMs. Numpy and scikit-learn only; tested without torch.
- `allen_data.py`: loads one Allen natural-scenes experiment and keeps every trial.
- `features.py`: the two matched models and per-stage feature hooks.
- `neural_alignment.py`: the main analysis.
- `perturbations.py`, `ood_test.py`: the exploratory corruption test.
- `write_results_table.py`, `RESULTS_original_run.md`: the original run's numbers, regenerated from its JSON.
- `analysis_results/`: the original run's outputs, unchanged.

## Method notes

- Ceiling: for each neuron, trials are split in half at random, each half is averaged, the two half-means are correlated across images, and the correlation is Spearman-Brown corrected (r_full = 2r / (1 + r)); averaged over 20 random splits. The normalised score is r² of the prediction over that reliability.
- Regression: features at each stage are flattened over space or tokens, projected to 4096 dimensions with a sparse random projection (Conwell et al. 2021), standardised on the training fold, and fit with ridge, one penalty per neuron by leave-one-out on the training fold. Five outer folds; predictions are gathered and scored once per neuron. Under no signal this scorer gives a slightly negative r, which is the expected sign that nothing was learned, not evidence of anti-prediction.
- Stimuli: Allen natural scenes are grayscale, 918 by 1174, resized to 224 square without preserving aspect ratio. The mouse saw them across about 120 degrees of visual field; the model sees the whole image. This is the standard compromise in the mouse literature and it limits what any model can reach.
- Response: mean dF/F over the stimulus frames shifted by two frames for the calcium lag.

## References

- Conwell, Mayo, Buice, Katz, Alvarez, Barbu. Neural regression, representational similarity, model zoology and neural taskonomy at scale in rodent visual cortex. NeurIPS 2021.
- Conwell, Prince, Kay, Alvarez, Konkle. A large-scale examination of inductive biases shaping high-level visual representation in brains and machines. Nature Communications 2024.
- Madan, Xiao, Cao, Pfister, Livingstone, Kreiman. Benchmarking out-of-distribution generalization capabilities of DNN-based encoding models for the ventral visual cortex. NeurIPS 2024.
- Allen Institute Brain Observatory, Visual Coding 2P.
