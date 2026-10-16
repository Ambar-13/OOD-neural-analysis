# Results of the original run (December 2025), generated from the committed JSON

Do not edit by hand. `python write_results_table.py` rewrites this file.

These numbers come from the December 2025 pipeline (see CHANGELOG.md for how it differs from the current one). They are kept so the record can be checked.

## Baseline predictivity, 5-fold CV over all 118 images

| Model | Median r | Mean r |
|---|---|---|
| ViT-B/16 (torchvision, in1k) | 0.090 | 0.076 |
| ResNet-34 (torchvision, in1k) | 0.057 | 0.067 |

Paired t-test over neurons: p = 0.55; Wilcoxon p = 0.49; Cohen's d = 0.06. The two models are not distinguishable on this data.

The file also records a 'noise ceiling' of 0.636, computed by splitting neurons rather than trials; it is not a per-neuron ceiling and is not used.

## RDM comparison (Spearman, upper triangles)

| Pair | rho | p |
|---|---|---|
| ViT vs neural | -0.021 | 0.08 |
| ResNet vs neural | 0.001 | 0.93 |
| ViT vs ResNet | 0.536 | 0.0e+00 |

Neither final-layer RDM is related to the V1 RDM.

## Out-of-distribution run (single split)

Train 94 images, test 24 images, 100 neurons, 500 bootstrap resamples over neurons.

Baseline median r on the 24 test images: ViT 0.1296, ResNet 0.1291.

52 conditions. Retention = per-neuron OOD r over baseline r, for neurons with baseline r > 0.01; median over those neurons, with a bootstrap CI over neurons (not over images).

| Condition | ViT r | ResNet r | ViT retention % [CI] | ResNet retention % [CI] | neurons |
|---|---|---|---|---|---|
| contrast_0.1 | 0.073 | 0.050 | 63 [48, 74] | 33 [6, 67] | 72 |
| contrast_0.3 | 0.105 | 0.031 | 81 [72, 89] | 58 [21, 81] | 72 |
| contrast_0.5 | 0.127 | 0.067 | 96 [92, 105] | 76 [56, 91] | 72 |
| contrast_1.5 | 0.108 | 0.109 | 83 [76, 91] | 90 [84, 105] | 72 |
| contrast_2.5 | 0.108 | 0.116 | 80 [70, 88] | 88 [73, 105] | 72 |
| contrast_4.0 | 0.099 | 0.144 | 72 [56, 95] | 93 [79, 103] | 72 |
| brightness_0.1 | 0.016 | 0.129 | 12 [-14, 46] | 76 [58, 100] | 72 |
| brightness_0.3 | 0.090 | 0.162 | 84 [68, 97] | 103 [78, 115] | 72 |
| brightness_0.5 | 0.107 | 0.133 | 91 [83, 102] | 97 [91, 105] | 72 |
| brightness_1.5 | 0.103 | 0.089 | 90 [78, 96] | 93 [81, 107] | 72 |
| brightness_2.5 | 0.111 | 0.057 | 71 [60, 99] | 64 [43, 77] | 72 |
| brightness_4.0 | 0.086 | 0.025 | 48 [27, 70] | 20 [12, 40] | 72 |
| blur_gaussian_s1 | 0.080 | 0.076 | 75 [56, 90] | 85 [69, 98] | 72 |
| blur_gaussian_s2 | 0.110 | 0.077 | 79 [48, 99] | 73 [36, 99] | 72 |
| blur_gaussian_s4 | 0.108 | 0.076 | 83 [54, 113] | 64 [30, 93] | 72 |
| blur_gaussian_s6 | 0.047 | 0.088 | 38 [12, 75] | 49 [17, 67] | 72 |
| blur_gaussian_s10 | 0.024 | 0.051 | 16 [-8, 46] | 37 [13, 71] | 72 |
| blur_defocus_r2 | 0.057 | 0.080 | 63 [48, 79] | 85 [65, 95] | 72 |
| blur_defocus_r4 | 0.131 | 0.076 | 79 [55, 101] | 60 [38, 90] | 72 |
| blur_defocus_r6 | 0.096 | 0.067 | 69 [41, 87] | 53 [40, 106] | 72 |
| blur_motion_s5_a0 | 0.078 | 0.092 | 64 [46, 83] | 74 [64, 82] | 72 |
| blur_motion_s5_a45 | 0.086 | 0.093 | 75 [60, 89] | 77 [63, 94] | 72 |
| blur_motion_s10_a0 | 0.092 | 0.082 | 73 [50, 111] | 68 [39, 88] | 72 |
| blur_motion_s10_a45 | 0.106 | -0.012 | 72 [56, 92] | 10 [-24, 52] | 72 |
| blur_motion_s15_a0 | 0.118 | -0.024 | 63 [46, 103] | 15 [-23, 53] | 72 |
| blur_motion_s15_a45 | 0.124 | 0.002 | 75 [51, 93] | 28 [-6, 61] | 72 |
| blur_zoom_1.1 | 0.106 | 0.098 | 76 [62, 102] | 80 [44, 96] | 72 |
| blur_zoom_1.2 | 0.098 | 0.048 | 76 [46, 122] | 47 [16, 73] | 72 |
| blur_zoom_1.3 | 0.105 | 0.072 | 73 [38, 96] | 48 [10, 81] | 72 |
| noise_gaussian_0.02 | 0.114 | 0.111 | 94 [82, 97] | 93 [84, 97] | 72 |
| noise_gaussian_0.05 | 0.106 | 0.117 | 83 [76, 95] | 85 [72, 103] | 72 |
| noise_gaussian_0.10 | 0.098 | 0.064 | 76 [67, 89] | 74 [60, 90] | 72 |
| noise_gaussian_0.15 | 0.025 | 0.039 | 38 [27, 63] | 39 [12, 58] | 72 |
| noise_gaussian_0.25 | 0.043 | 0.040 | 45 [23, 63] | 46 [0, 81] | 72 |
| noise_shot_0.5 | 0.081 | 0.096 | 82 [70, 90] | 74 [61, 107] | 72 |
| noise_shot_1.0 | 0.125 | 0.109 | 82 [73, 91] | 93 [78, 112] | 72 |
| noise_shot_2.0 | 0.110 | 0.115 | 87 [80, 93] | 90 [72, 101] | 72 |
| noise_impulse_0.01 | 0.147 | 0.092 | 95 [84, 119] | 81 [60, 88] | 72 |
| noise_impulse_0.03 | 0.126 | 0.035 | 86 [70, 107] | 54 [24, 83] | 72 |
| noise_impulse_0.05 | 0.112 | 0.072 | 82 [66, 110] | 45 [41, 77] | 72 |
| noise_speckle_0.05 | 0.130 | 0.136 | 95 [88, 99] | 97 [83, 104] | 72 |
| noise_speckle_0.10 | 0.088 | 0.112 | 70 [62, 87] | 99 [77, 110] | 72 |
| noise_speckle_0.20 | 0.042 | 0.076 | 50 [29, 84] | 87 [73, 106] | 72 |
| weather_snow_0.5 | 0.100 | 0.073 | 80 [72, 89] | 74 [62, 85] | 72 |
| weather_snow_1.0 | 0.126 | 0.083 | 85 [75, 92] | 86 [67, 93] | 72 |
| weather_snow_2.0 | 0.077 | 0.087 | 65 [47, 83] | 63 [45, 83] | 72 |
| weather_frost_0.3 | 0.120 | 0.083 | 96 [93, 99] | 92 [85, 100] | 72 |
| weather_frost_0.5 | 0.111 | 0.090 | 93 [88, 97] | 81 [70, 94] | 72 |
| weather_frost_0.8 | 0.121 | 0.088 | 92 [83, 103] | 74 [57, 82] | 72 |
| weather_fog_0.3 | 0.117 | 0.075 | 100 [95, 103] | 83 [68, 97] | 72 |
| weather_fog_0.5 | 0.110 | 0.047 | 89 [82, 96] | 73 [33, 86] | 72 |
| weather_fog_0.7 | 0.103 | 0.016 | 62 [53, 76] | 53 [4, 78] | 72 |

### What the table actually shows

- ViT retention ranges from 12% (brightness_0.1) to 100% (weather_fog_0.3). 6 conditions are below 50%.
- ResNet retention ranges from 10% (blur_motion_s10_a45) to 103% (brightness_0.3). 12 conditions are below 50%.
- With 24 test images the standard error of a single r near 0.13 is about 0.2, so the retention values are ratios of two numbers within their own noise. The bootstrap intervals resample neurons and do not include image-sampling noise.
