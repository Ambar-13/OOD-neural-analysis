# Mouse V1 predictivity of ViT-S/16 and ResNet-34, stage by stage

Ambar, Computer Engineering, National University of Singapore.

Which stage of a vision transformer and of a ResNet best predicts mouse primary visual cortex, and how close does each get to the noise ceiling? Both models have 22M parameters and ImageNet-1k weights, so the comparison is between architectures rather than training sets. A second script asks whether readouts of different flexibility keep their score equally well under image corruptions.

## Method

- Data: one Allen Brain Observatory natural-scenes experiment (VISp, 118 grayscale scenes, about 50 presentations each), with every trial kept.
- Ceiling: per neuron, split-half across trials, Spearman-Brown corrected, averaged over 20 splits. Neurons below 0.5 are dropped. Scores are r² of the prediction over that ceiling.
- Models: timm `vit_small_patch16_224.augreg_in1k` and `resnet34.a1_in1k`, every stage recorded through forward hooks.
- Regression: each stage's features are flattened, projected to 4096 dimensions with a sparse random projection (Conwell et al. 2021), standardised on the training fold, and fit with ridge, one penalty per neuron. Five folds; held-out predictions are scored once per neuron.
- RDMs: at each model's best stage, against the neural RDM and its own split-half ceiling.
- Corruptions: contrast, brightness, blur, noise and weather on the held-out images, ten random splits, with ridge on all features, on 100, 20 and 5 components, and RSA, side by side. With 118 images the intervals are wide; the same analysis on MacaqueITBench is in a separate repository.

## Running it

```bash
./install.sh                      # venv with torch, timm, allensdk (Python 3.9 to 3.11)
source venv/bin/activate
python neural_alignment.py        # downloads one VISp experiment, sweeps stages, writes results/
python ood_test.py                # corruption test at the best stage
```

## Results so far

Pipeline check on the images and trial-averaged responses in `analysis_results/` (100 neurons, no ceiling, image-level cross-validation, CPU). The full run with trial data is pending.

| ViT-S/16 stage | median r | ResNet-34 stage | median r |
|---|---|---|---|
| patch_embed | -0.12 | stem | -0.05 |
| block00 | 0.08 | layer1 | -0.01 |
| block02 | **0.11** | layer2 | **0.04** |
| block03 | 0.10 | layer3 | -0.01 |
| block05 to block11 | -0.05 to -0.09 | layer4 | -0.09 |
| norm (final) | -0.05 | pool (final) | -0.09 |

Random 4096-dimensional features give median r of -0.15 on this data, the floor for this scorer. Both final stages sit near it; the best stages are a quarter of the way into the ViT and at the second ResNet stage, as in Conwell et al. 2021.

## Files

- `encoding.py`: ceiling, neuron selection, random projection, ridge, readouts, RDMs. Numpy and scikit-learn only.
- `allen_data.py`: one experiment with every trial.
- `features.py`: the two models and per-stage hooks.
- `neural_alignment.py`: the stage sweep.
- `perturbations.py`, `ood_test.py`: the corruption test.
- `write_results_table.py`: regenerates `RESULTS_original_run.md` from the December 2025 result files in `analysis_results/`.

## Notes

- Allen natural scenes are grayscale, 918 by 1174, resized to 224 square without preserving aspect ratio. The mouse saw them across about 120 degrees of visual field; the model sees the whole image. This is the usual compromise in the mouse literature.
- With uninformative features the gathered cross-validated r is slightly negative, since each fold regresses toward its own training mean.
- A response is the mean dF/F over the stimulus frames, shifted two frames for the calcium lag.

October 2026: the pipeline was rebuilt after a code review; the December 2025 numbers are kept in [RESULTS_original_run.md](RESULTS_original_run.md) and the changes are listed in [CHANGELOG.md](CHANGELOG.md).

## References

- Conwell, Mayo, Buice, Katz, Alvarez, Barbu. Neural regression, representational similarity, model zoology and neural taskonomy at scale in rodent visual cortex. NeurIPS 2021.
- Conwell, Prince, Kay, Alvarez, Konkle. A large-scale examination of inductive biases shaping high-level visual representation in brains and machines. Nature Communications 2024.
- Madan, Xiao, Cao, Pfister, Livingstone, Kreiman. Benchmarking out-of-distribution generalization capabilities of DNN-based encoding models for the ventral visual cortex. NeurIPS 2024.
- Allen Institute Brain Observatory, Visual Coding 2P.
