# Changelog

## October 2026

Pipeline rebuilt after a code review. The December 2025 result files are kept in `analysis_results/` and summarised in `RESULTS_original_run.md`, which `write_results_table.py` regenerates from them.

- Noise ceiling is now per neuron, split-half across the roughly 50 trials of each image, Spearman-Brown corrected. The earlier ceiling split the neuron population in half, which measures population consistency rather than per-neuron predictability.
- Trial-level responses are kept on load (`allen_data.py`); the earlier loader averaged trials away.
- Neurons are filtered by reliability (default 0.5). Earlier, 100 neurons were drawn at random.
- Every stage of each model is swept and the best is chosen by ceiling-normalised r². Earlier, only final-layer features were used, which is the weakest choice for V1.
- Models are a matched pair, `vit_small_patch16_224.augreg_in1k` and `resnet34.a1_in1k` (22M parameters each, ImageNet-1k). Earlier, torchvision ViT-B/16 (86M) was compared with ResNet-34.
- Regression uses a sparse random projection to 4096 dimensions and ridge with one penalty per neuron, scored once per neuron on gathered cross-validated predictions.
- The corruption test uses ten random splits with intervals over splits, instead of one 94/24 split with a bootstrap over neurons. Identity conditions on grayscale stimuli (hue, saturation, grayscale) are skipped.
- README: the December 2025 README described ViT retention as 70 to 96 percent across conditions and ResNet as 19 to 103 percent; the result files have ViT as low as 12 percent and ResNet as low as 10 percent, with 53 conditions rather than 47. The baseline difference between the models (median r 0.090 vs 0.057) had p = 0.55. The description of Madan et al. 2024 has been corrected: their shifts are natural percentile holdouts on macaque IT, and the 20 percent figure is one worst case.
- Old scripts (`base_neural_analysis.py`, `ood_testing_comprehensive_corrected.py`, `run_comprehensive_ood.py`) replaced by the modules listed in the README.

## December 2025

First version.
