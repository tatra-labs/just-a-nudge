## Main table: shares of the RL-tuned model's gain on held-out prompts (95% bootstrap CI)

| | match | upright |
|---|---|---|
| base reward | 0.9104 | -0.1738 |
| RL-tuned (LoRA r=32) reward | 0.9562 | -0.0647 |
| RL gain | +0.0458 [+0.0341, +0.0583] | +0.1090 [+0.1000, +0.1187] |
| s_mean (mean difference) share % | 61.9 [55.7, 70.3] | 10.2 [9.0, 11.5] |
| **s_fit (velocity matching) share %** | **83.0 [76.9, 89.1]** | **64.7 [62.6, 66.8]** |
| same-size LoRA (rank 2, out-proj), fitted, share % | 91.8 [86.4, 97.3] | 84.4 [82.9, 85.8] |
| **s_RL (vectors trained by Flow-GRPO from zero) share %** | **82.9 [76.8, 90.0]** | **47.9 [45.6, 50.2]** |
| random vectors, same norms, share % | -291.3 [-394.0, -223.6] | 45.1 [37.9, 51.5] |
| best CFG scale on the base, share % | 5.0 (CFG 6) | 2.1 (CFG 8) |
| steering parameters | 5,632 (44 vectors) | 5,632 (44 vectors) |
| RL LoRA parameters | 286,720 (51x) | 286,720 (51x) |
| velocity gap closed by s_fit (cond / uncond) | 0.62 / 0.73 | 0.64 / 0.68 |
| velocity gap closed by s_mean (cond / uncond) | 0.25 / 0.40 | 0.11 / 0.14 |
| velocity gap closed by same-size LoRA | 0.79 / 0.84 | 0.83 / 0.80 |

## Where the fitted vectors act (share % of the RL gain when only these vectors are kept)

| kept | match | upright |
|---|---|---|
| abl_cond_only | 46.7 [21.5, 63.5] | 77.3 [75.5, 78.9] |
| abl_uncond_only | -223.7 [-279.1, -185.2] | -13.4 [-15.3, -11.6] |
| abl_early_only | 66.9 [60.3, 75.9] | 62.1 [59.9, 64.3] |
| abl_late_only | 55.5 [50.2, 62.7] | 3.4 [3.1, 3.7] |
| abl_text_only | 82.8 [75.9, 89.8] | 44.6 [41.6, 47.3] |
| abl_image_only | 6.3 [0.3, 12.8] | 23.5 [22.0, 24.9] |
| abl_cond_text_early | 57.5 [51.1, 65.9] | 48.8 [46.0, 51.4] |
| abl_blocks_0-2 | 77.7 [69.9, 84.9] | 51.1 [48.0, 53.9] |
| abl_blocks_3-5 | -29.5 [-40.7, -21.3] | 20.1 [18.1, 22.2] |
| kvreset_s_fit | 17.5 [11.7, 23.9] | 23.9 [22.4, 25.4] |
| kvreset_rl_lora | 83.8 [77.2, 93.0] | 65.2 [63.6, 66.8] |

## Side effects: gains on every metric (held-out prompts)


**match**

| policy | match | exact (classifier B + weights) | weight in tercile | p(digit) | both digits right (B) | upright | ink |
|---|---|---|---|---|---|---|---|
| base (absolute) | 0.9104 | 0.9083 | 0.9533 | 0.9998 | 0.9992 | -0.1738 | 1.1387 |
| rl_lora | +0.0458 | +0.0767 | +0.0396 | -0.0001 | +0.0000 | -0.0002 | -0.0429 |
| s_fit | +0.0380 | +0.0675 | +0.0354 | -0.0002 | -0.0008 | +0.0006 | -0.0309 |
| s_rl | +0.0379 | +0.0692 | +0.0358 | -0.0004 | +0.0000 | +0.0008 | -0.0280 |

**upright**

| policy | match | exact (classifier B + weights) | weight in tercile | p(digit) | both digits right (B) | upright | ink |
|---|---|---|---|---|---|---|---|
| base (absolute) | 0.9104 | 0.9083 | 0.9533 | 0.9998 | 0.9992 | -0.1738 | 1.1387 |
| rl_lora | +0.0062 | +0.0175 | +0.0096 | -0.0002 | +0.0008 | +0.1090 | -0.0041 |
| s_fit | +0.0078 | +0.0142 | +0.0075 | -0.0000 | +0.0008 | +0.0705 | -0.0018 |
| s_rl | -0.0059 | -0.0175 | -0.0092 | +0.0001 | +0.0008 | +0.0523 | +0.0075 |

## CFG sweep on the base (gain on the run's reward)

| CFG | match | upright |
|---|---|---|
| 1 | -0.2133 (-466%) | -0.0286 (-26%) |
| 2 | -0.0269 (-59%) | -0.0102 (-9%) |
| 4 | -0.0022 (-5%) | -0.0011 (-1%) |
| 5 | +0.0006 (1%) | +0.0008 (1%) |
| 6 | +0.0023 (5%) | +0.0016 (1%) |
| 8 | +0.0006 (1%) | +0.0023 (2%) |
