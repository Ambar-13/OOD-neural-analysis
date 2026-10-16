# Mouse V1 predictivity of ViT-S/16 and ResNet-34, layer by layer

Ambar, Computer Engineering, National University of Singapore.

Which depth of a vision transformer and of a ResNet best predicts mouse primary visual cortex, and how close does each get to the per-neuron noise ceiling? Both models are 22M parameters with ImageNet-1k weights, so the comparison is between architectures, not between training sets. An exploratory second script measures how predictivity moves under synthetic image corruptions.

## Method

- **Data.** One Allen Brain Observatory natural-scenes experiment (VISp, 118 grayscale scenes, about 50 presentations each), loaded with every trial kept.
- **Ceiling.** Per neuron, split-half across trials, Spearman-Brown corrected, averaged over 20 random splits. Neurons below a reliability of 0.5 are dropped. Scores are reported as r² of the prediction over that reliability.
- **Models.** timm `vit_small_patch16_224.augreg_in1k` and `resnet34.a1_in1k`. Every stage of each model is recorded through forward hooks.
- **Regression.** Features at each stage are flattened, projected to 4096 dimensions with a sparse random projection (Conwell et al. 2021), standardised on the training fold, and fit with ridge, one penalty per neuron. Five outer folds; predictions are gathered and scored once per neuron.
- **RDMs.** At each model's best stage, the model RDM is compared with the neural RDM and with the neural RDM's own split-half ceiling.
- **Corruptions (exploratory).** Contrast, brightness, blur, noise and weather conditions on the held-out images, with ten random train/test splits so that every number carries an interval over image splits. With 118 images these intervals are wide, and the script says so; for a real out-of-distribution test use Madan et al.'s MacaqueITBench.

## Running it

```bash
./install.sh                      # venv with torch, timm, allensdk (Python 3.9 to 3.11)
source venv/bin/activate
python neural_alignment.py        # downloads one VISp experiment, sweeps layers, writes results/
python ood_test.py                # exploratory corruption test at the best layer
```

## Results

The full run with trial data is scheduled. A check of the pipeline on the images and trial-averaged responses already in `analysis_results/` (100 neurons, no ceiling, image-level cross-validation, CPU) gives:

| ViT-S/16 stage | median r | ResNet-34 stage | median r |
|---|---|---|---|
| patch_embed | -0.12 | stem | -0.05 |
| block00 | 0.08 | layer1 | -0.01 |
| block02 | **0.11** | layer2 | **0.04** |
| block03 | 0.10 | layer3 | -0.01 |
| block05 to block11 | -0.05 to -0.09 | layer4 | -0.09 |
| norm (final) | -0.05 | pool (final) | -0.09 |

Random 4096-dimensional features give a median r of -0.15 on this data, the no-signal floor for this scorer. The final stages of both models sit near it; the best stages are a quarter of the way into the ViT and at the second ResNet stage, in line with Conwell et al. 2021. Whether the two models differ at their best stage needs the ceiling, the reliability filter and the intervals from the full run.

## Files

- `encoding.py`: ceiling, neuron selection, random projection, cross-validated ridge, RDMs. Numpy and scikit-learn only.
- `allen_data.py`: loads one experiment with every trial.
- `features.py`: the two models and the per-stage hooks.
- `neural_alignment.py`: the main analysis.
- `perturbations.py`, `ood_test.py`: the corruption test.
- `write_results_table.py`: regenerates `RESULTS_original_run.md` from the December 2025 result files.
- `analysis_results/`: outputs of the December 2025 run, unchanged.

## Notes

- Allen natural scenes are grayscale, 918 by 1174, resized to 224 square; aspect ratio is not preserved. The mouse saw them across about 120 degrees of visual field while the model sees the whole image at once. This is the usual compromise in the mouse literature and it caps what any model can reach.
- Under no signal the gathered-CV scorer gives a slightly negative r; that is the expected sign that nothing was learned.
- Response per presentation is the mean dF/F over the stimulus frames, shifted two frames for the calcium lag.

## Revision note

October 2026: the pipeline was rebuilt after a code review. The December 2025 version computed its noise ceiling across neurons rather than trials, used final-layer features for V1, tested on a single 24-image split, and compared a torchvision ViT-B/16 with a ResNet-34. Its README also summarised the retention results more favourably than its own result files support. The earlier numbers are kept in [RESULTS_original_run.md](RESULTS_original_run.md), regenerated from those files; the full list of changes is in [CHANGELOG.md](CHANGELOG.md).

## References

- Conwell, Mayo, Buice, Katz, Alvarez, Barbu. Neural regression, representational similarity, model zoology and neural taskonomy at scale in rodent visual cortex. NeurIPS 2021.
- Conwell, Prince, Kay, Alvarez, Konkle. A large-scale examination of inductive biases shaping high-level visual representation in brains and machines. Nature Communications 2024.
- Madan, Xiao, Cao, Pfister, Livingstone, Kreiman. Benchmarking out-of-distribution generalization capabilities of DNN-based encoding models for the ventral visual cortex. NeurIPS 2024.
- Allen Institute Brain Observatory, Visual Coding 2P.
