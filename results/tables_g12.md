
### exp1

| Condition | Discrimination % [95% CI] | Order-cancelled % [95% CI] | n trials | '1' rate % | Detection (order-cancelled) % | Story words | Turn-1 tokens | Literal mention % |
|---|---|---|---|---|---|---|---|---|
| no plan (anchor) (`dont_reveal`) | 84.1 [75.0, 92.6] | 87.1 [78.1, 95.3] | 1440 | 48 | 84.2 [66.1, 97.4] | 524 | 0 | 0.8 |
| decoy word (`decoy`) | 83.1 [74.8, 91.4] | 87.1 [77.3, 95.1] | 960 | 48 | n/a | 522 | 0 | 0.0 |
| self-written outline (`plan_self`) | 81.9 [74.5, 89.5] | 84.2 [74.5, 92.6] | 960 | 54 | n/a | 631 | 500 | 0.8 |
| secret-blind outline (`plan_blind`) | 49.7 [45.1, 54.1] | 46.7 [40.7, 53.2] | 960 | 44 | 85.0 [69.1, 96.8] | 680 | 500 | 0.0 |
| secret-blind premise (brief) (`plan_brief`) | 54.7 [46.3, 61.3] | 54.8 [43.8, 64.3] | 960 | 54 | n/a | 655 | 52 | 0.8 |
| filler: craft advice (300 tok) (`filler_craft`) | 79.5 [69.9, 89.5] | 83.5 [73.4, 93.5] | 960 | 53 | 82.5 [65.0, 96.2] | 537 | 301 | 0.0 |
| filler: unrelated text (300 tok) (`filler_irrelevant`) | 55.8 [45.9, 64.7] | 54.8 [42.8, 65.6] | 960 | 42 | n/a | 525 | 302 | 0.0 |
| filler: unrelated text (300 tok), neutral request (`filler_irrelevant_plain`) | 78.3 [70.3, 86.4] | 78.8 [69.4, 87.9] | 960 | 51 | n/a | 518 | 302 | 0.0 |
| filler: craft advice (500 tok) (`filler_craft_long`) | 75.1 [64.4, 84.9] | 81.0 [71.4, 90.4] | 960 | 55 | n/a | 554 | 500 | 0.0 |

### exp1_contrasts

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| plan_blind - dont_reveal | -34.4 [-44.9, -24.1] | 0.0002 | 0.0028 |
| plan_self - dont_reveal | -2.2 [-9.1, +6.8] | 0.5036 | 1.0000 |
| plan_brief - dont_reveal | -29.4 [-42.3, -19.4] | 0.0002 | 0.0028 |
| filler_craft - dont_reveal | -4.6 [-9.6, +0.2] | 0.0616 | 0.3696 |
| filler_irrelevant - dont_reveal | -28.3 [-40.9, -17.9] | 0.0002 | 0.0028 |
| decoy - dont_reveal | -1.0 [-10.4, +9.8] | 0.8452 | 1.0000 |
| plan_blind - filler_craft | -29.8 [-41.7, -18.9] | 0.0002 | 0.0028 |
| plan_blind - filler_irrelevant | -6.1 [-16.7, +4.6] | 0.2764 | 0.8292 |
| filler_irrelevant_plain - dont_reveal | -5.8 [-14.0, +2.6] | 0.1484 | 0.5936 |
| filler_irrelevant_plain - filler_irrelevant | +22.5 [+14.2, +30.7] | 0.0002 | 0.0028 |
| filler_craft_long - dont_reveal | -9.0 [-17.3, -1.7] | 0.0156 | 0.1092 |
| plan_blind - filler_craft_long | -25.4 [-36.6, -12.3] | 0.0002 | 0.0028 |
| plan_self - filler_craft_long | +6.8 [-0.8, +15.2] | 0.0756 | 0.3780 |
| plan_self - plan_blind | +32.2 [+22.3, +40.8] | 0.0002 | 0.0028 |

### exp1_contrasts_oc

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| plan_blind - dont_reveal | -40.4 [-48.7, -31.4] | 0.0002 | 0.0028 |
| plan_self - dont_reveal | -2.9 [-12.1, +7.0] | 0.5180 | 1.0000 |
| plan_brief - dont_reveal | -32.3 [-47.3, -20.4] | 0.0002 | 0.0028 |
| filler_craft - dont_reveal | -3.5 [-9.4, +2.0] | 0.2088 | 1.0000 |
| filler_irrelevant - dont_reveal | -32.3 [-45.7, -20.6] | 0.0002 | 0.0028 |
| decoy - dont_reveal | +0.0 [-10.7, +11.6] | 0.9776 | 1.0000 |
| plan_blind - filler_craft | -36.9 [-47.0, -26.7] | 0.0002 | 0.0028 |
| plan_blind - filler_irrelevant | -8.1 [-20.9, +6.1] | 0.2288 | 1.0000 |
| filler_irrelevant_plain - dont_reveal | -8.3 [-18.2, +1.0] | 0.0716 | 0.5012 |
| filler_irrelevant_plain - filler_irrelevant | +24.0 [+15.5, +33.3] | 0.0002 | 0.0028 |
| filler_craft_long - dont_reveal | -6.0 [-14.0, +1.9] | 0.1124 | 0.6744 |
| plan_blind - filler_craft_long | -34.4 [-44.0, -24.4] | 0.0002 | 0.0028 |
| plan_self - filler_craft_long | +3.1 [-5.1, +12.1] | 0.4824 | 1.0000 |
| plan_self - plan_blind | +37.5 [+27.2, +47.0] | 0.0002 | 0.0028 |

outline leak: 83.8 [77.9, 89.7]

### exp2

| Condition | Discrimination % [95% CI] | Order-cancelled % [95% CI] | n trials | '1' rate % | Detection (order-cancelled) % | Story words |
|---|---|---|---|---|---|---|
| no plan (anchor) (`base`) | 55.0 [51.6, 59.3] | 60.4 [53.1, 67.4] | 1152 | 92 | 31.9 [23.6, 40.3] | 476 |
| self outline, whole story (`plan_self_full`) | 55.6 [53.4, 58.2] | 64.6 [60.1, 69.4] | 1152 | 89 | 45.1 [37.5, 53.5] | 457 |
| self outline, opening (`plan_self_open`) | 53.0 [50.9, 55.4] | 58.0 [55.0, 61.6] | 1152 | 90 | 35.4 [22.9, 49.3] | 493 |
| secret-blind outline (`plan_blind`) | 51.3 [50.4, 52.3] | 53.8 [51.7, 55.9] | 1152 | 96 | 50.7 [38.2, 63.2] | 525 |
| filler: craft advice (300 tok) (`filler_craft`) | 55.6 [53.0, 58.7] | 59.9 [52.3, 67.4] | 1152 | 89 | 38.2 [27.1, 49.3] | 445 |
| no hide instruction, no plan (`open_base`) | 60.6 [56.2, 65.6] | 68.1 [61.8, 74.3] | 576 | 89 | n/a | 495 |
| no hide instruction, twist-blind outline (`open_plan_blind`) | 51.9 [49.1, 54.9] | 55.6 [47.6, 63.2] | 576 | 95 | n/a | 523 |

### exp2_contrasts

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| plan_blind - base | -3.7 [-7.3, -0.7] | 0.0156 | 0.0780 |
| plan_self_full - base | +0.5 [-2.6, +3.6] | 0.7492 | 1.0000 |
| plan_self_open - base | -2.1 [-5.1, +0.3] | 0.1300 | 0.5200 |
| filler_craft - base | +0.5 [-1.0, +1.9] | 0.5144 | 1.0000 |
| plan_blind - filler_craft | -4.3 [-6.9, -1.9] | 0.0002 | 0.0016 |
| plan_self_open - plan_blind | +1.6 [-0.3, +4.2] | 0.1536 | 0.5200 |
| open_base - base | +5.6 [+1.9, +9.8] | 0.0004 | 0.0024 |
| open_plan_blind - open_base | -8.7 [-13.9, -4.3] | 0.0002 | 0.0016 |

### exp2_contrasts_oc

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| plan_blind - base | -6.6 [-13.2, +0.2] | 0.0604 | 0.3408 |
| plan_self_full - base | +4.2 [-2.6, +10.8] | 0.2272 | 0.6816 |
| plan_self_open - base | -2.4 [-8.9, +3.8] | 0.4816 | 0.9632 |
| filler_craft - base | -0.5 [-7.3, +5.7] | 0.9192 | 0.9632 |
| plan_blind - filler_craft | -6.1 [-13.9, +2.3] | 0.1492 | 0.5968 |
| plan_self_open - plan_blind | +4.2 [+1.6, +6.9] | 0.0012 | 0.0096 |
| open_base - base | +7.6 [+0.0, +15.1] | 0.0568 | 0.3408 |
| open_plan_blind - open_base | -12.5 [-22.2, -1.4] | 0.0292 | 0.2044 |

### exp2_contrasts_det

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| plan_blind - base | +18.8 [+9.7, +27.8] | 0.0002 | 0.0008 |
| filler_craft - base | +6.2 [-2.8, +15.3] | 0.2144 | 0.2144 |
| plan_blind - filler_craft | +12.5 [+2.1, +23.6] | 0.0276 | 0.0552 |
| plan_self_full - base | +13.2 [+6.2, +20.8] | 0.0004 | 0.0012 |

### exp3

| Condition | Discrimination % [95% CI] | Order-cancelled % [95% CI] | n trials | '1' rate % | Detection (order-cancelled) % | Story words | Rating (1-9) | NLL/token | Distinct-2 | Recall /15 | Literal mention % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no intervention (`abl_none`) | 80.8 [70.3, 90.7] | 84.4 [73.7, 94.4] | 360 | 49 | n/a | 525 | 8.04 ± 0.01 | 0.643 ± 0.007 | 0.952 | 15 | 1.1 |
| subtract own word's signature (`sub_own`) | 75.6 [62.9, 87.1] | 75.6 [60.4, 89.5] | 360 | 52 | n/a | 531 | 8.05 ± 0.01 | 0.652 ± 0.007 | 0.950 | 15 | 0.0 |
| subtract 2x own signature (`sub_own_k2`) | 71.4 [57.1, 84.4] | 71.7 [58.1, 84.5] | 360 | 50 | n/a | 526 | 8.05 ± 0.02 | 0.760 ± 0.011 | 0.948 | 15 | 1.1 |
| subtract other word's signature (`sub_other`) | 85.6 [75.3, 94.1] | 87.8 [77.3, 96.8] | 360 | 50 | n/a | 517 | 8.04 ± 0.01 | 0.680 ± 0.008 | 0.952 | 15 | 0.0 |
| subtract random vector (same norm) (`sub_random`) | 84.7 [74.7, 93.9] | 85.6 [74.1, 96.2] | 360 | 49 | n/a | 529 | 8.05 ± 0.02 | 0.675 ± 0.008 | 0.950 | 15 | 0.0 |
| no secret + add signature (`add_own_k1`) | 53.1 [46.8, 59.0] | 54.4 [44.1, 65.3] | 360 | 42 | n/a | 603 | 8.38 ± 0.04 | 0.572 ± 0.005 | 0.935 | n/a | 2.2 |
| no secret + add 3x signature (`add_own_k3`) | 54.4 [45.1, 63.2] | 62.2 [50.0, 74.8] | 360 | 43 | n/a | 605 | 8.26 ± 0.03 | 0.818 ± 0.014 | 0.929 | n/a | 0.0 |
| secret in context for first 30 tokens only (`swap_early30`) | 52.2 [44.2, 61.3] | 56.1 [41.0, 70.0] | 360 | 43 | n/a | 599 | n/a | n/a | n/a | n/a | 5.6 |
| secret for first 100 tokens only (`swap_early100`) | 68.1 [59.0, 77.2] | 68.3 [54.7, 81.4] | 360 | 50 | n/a | 595 | n/a | n/a | n/a | n/a | 20.0 |
| secret only after first 30 tokens (`swap_late30`) | 80.6 [71.6, 89.2] | 81.7 [71.2, 91.1] | 360 | 49 | n/a | 540 | n/a | n/a | n/a | n/a | 6.7 |
| secret only after first 100 tokens (`swap_late100`) | 77.5 [67.8, 86.6] | 81.7 [68.4, 92.4] | 360 | 46 | n/a | 548 | n/a | n/a | n/a | n/a | 6.7 |

### exp3_contrasts

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| sub_own - abl_none | -5.3 [-16.5, +5.8] | 0.3184 | 1.0000 |
| sub_own_k2 - abl_none | -9.4 [-20.8, +0.5] | 0.0652 | 0.5216 |
| sub_other - abl_none | +4.7 [-5.6, +15.2] | 0.3448 | 1.0000 |
| sub_random - abl_none | +3.9 [-4.8, +13.4] | 0.3920 | 1.0000 |
| sub_own - sub_other | -10.0 [-17.6, -3.4] | 0.0036 | 0.0396 |
| sub_own - sub_random | -9.2 [-19.4, -0.5] | 0.0440 | 0.3960 |
| add_own_k3 - add_own_k1 | +1.4 [-9.0, +10.2] | 0.7864 | 1.0000 |
| swap_early30 - abl_none | -28.6 [-39.1, -17.4] | 0.0002 | 0.0024 |
| swap_early100 - abl_none | -12.8 [-22.0, -3.7] | 0.0108 | 0.1080 |
| swap_late30 - abl_none | -0.3 [-9.6, +10.5] | 0.9816 | 1.0000 |
| swap_late100 - abl_none | -3.3 [-13.3, +6.1] | 0.4808 | 1.0000 |
| swap_early100 - swap_late100 | -9.4 [-21.0, +1.3] | 0.0848 | 0.5936 |

### exp3_contrasts_oc

| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |
|---|---|---|---|
| sub_own - abl_none | -8.9 [-23.3, +4.3] | 0.2040 | 1.0000 |
| sub_own_k2 - abl_none | -12.8 [-25.4, -3.0] | 0.0104 | 0.1040 |
| sub_other - abl_none | +3.3 [-6.3, +12.6] | 0.4744 | 1.0000 |
| sub_random - abl_none | +1.1 [-8.8, +11.5] | 0.8272 | 1.0000 |
| sub_own - sub_other | -12.2 [-23.0, -2.5] | 0.0164 | 0.1476 |
| sub_own - sub_random | -10.0 [-26.8, +4.9] | 0.2060 | 1.0000 |
| add_own_k3 - add_own_k1 | +7.8 [-5.1, +20.5] | 0.2376 | 1.0000 |
| swap_early30 - abl_none | -28.3 [-44.8, -14.5] | 0.0002 | 0.0024 |
| swap_early100 - abl_none | -16.1 [-27.2, -5.6] | 0.0064 | 0.0704 |
| swap_late30 - abl_none | -2.8 [-13.8, +8.1] | 0.6820 | 1.0000 |
| swap_late100 - abl_none | -2.8 [-13.3, +7.1] | 0.6232 | 1.0000 |
| swap_early100 - swap_late100 | -13.3 [-25.9, -1.1] | 0.0404 | 0.3232 |
