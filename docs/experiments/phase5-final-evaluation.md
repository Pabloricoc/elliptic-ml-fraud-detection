# Phase 5 final evaluation

One authorised evaluation of steps 40–49 under the [approved pre-registration](preregistration.md). Access started 2026-11-28T14:07:53.368204+01:00 on executable revision `b46a7f5f9c947b6d8989f3d854c3d700cad1380f`. The source release SHA-256 is `903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043`. All S2 results are conditional on A1. The deployed S1 model remains the legacy XGBoost.

Frozen shortlist SHA-256 `75460d2e2de778df60db2f6a4511f6c86160c39e6f220cf695827b2702706ed1`. Approval revision `52d2915f3a5b85986a7618ee89d572a53630ac8f`.

## Frozen arms

| Arm | Frozen identity | Model SHA-256 |
| --- | --- | --- |
| S1 legacy XGBoost | model-4e919f2711a948c78cd080cb6f0e3417 | b66ee19643d20a1af4e7a393d9d9194f163a04869a2997a04d7d7ebfba9e4282 |
| S1 T0 XGBoost refit | frozen-3716ebddbff64811a5fce150b45c7e9f/s1_refit | 1993dc4ccae68c7fa0fe151055857e1a270d2baff715c62fe24864485be4d519 |
| S2 Random Forest | frozen-3716ebddbff64811a5fce150b45c7e9f/s2_rf | 25c98710d29aa1803953e363673d2b693ec901eb07590abed991f0e29df3c431 |
| S2 LightGBM (5 seeds) | frozen-3716ebddbff64811a5fce150b45c7e9f/s2_lgbm | a7c6508b2955f47fcdb9c7ade556eb4f2d8779149397e81fcb4e421361b8320c |

The LightGBM model digest binds the five member digests in seed order.

## Paired contrasts for mean step AP

Each interval uses the same 1,000 paired whole-step bootstrap draws with seed 42. The two primary intervals are nominal 95% intervals without multiplicity adjustment. The third interval is descriptive.

| Role | Contrast | AP difference | 95% CI | Paired steps | Valid draws | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| Primary | S2 LightGBM − S2 Random Forest | 0.043190 | [0.008068, 0.085532] | 10 | 1000 | S2 LightGBM has an advantage in this period. |
| Primary | S2 LightGBM − S1 legacy XGBoost | 0.329241 | [0.166398, 0.500979] | 10 | 1000 | S2 LightGBM has an advantage in this period. |
| Descriptive | S2 LightGBM − S1 T0 XGBoost refit | 0.330575 | [0.167224, 0.502495] | 10 | 1000 | Descriptive interval entirely above zero. |

## Average precision and probability metrics

| Arm | Mean step AP | 95% CI | Defined steps | Valid draws | Step AP SD | Pooled AP | ROC-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 legacy XGBoost | 0.116532 | [0.062393, 0.171766] | 10 | 1000 | 0.093125 | 0.088235 | 0.625770 | 0.056860 |
| S1 T0 XGBoost refit | 0.115198 | [0.060101, 0.170617] | 10 | 1000 | 0.093182 | 0.086827 | 0.622535 | 0.056285 |
| S2 Random Forest | 0.402582 | [0.194164, 0.628172] | 10 | 1000 | 0.372247 | 0.709176 | 0.923753 | 0.024384 |
| S2 LightGBM (5 seeds) | 0.445772 | [0.240940, 0.662389] | 10 | 1000 | 0.359711 | 0.751160 | 0.929317 | 0.023300 |

## Labelled queue precision and recall

| Arm | K | Mean step precision | 95% precision CI | Mean step recall |
| --- | --- | --- | --- | --- |
| S1 legacy XGBoost | 10 | 0.100000 | [0.040000, 0.160000] | 0.063477 |
| S1 legacy XGBoost | 25 | 0.096000 | [0.024000, 0.176000] | 0.073420 |
| S1 legacy XGBoost | 50 | 0.096000 | [0.019950, 0.180000] | 0.096397 |
| S1 T0 XGBoost refit | 10 | 0.080000 | [0.030000, 0.150000] | 0.062292 |
| S1 T0 XGBoost refit | 25 | 0.080000 | [0.028000, 0.144100] | 0.079243 |
| S1 T0 XGBoost refit | 50 | 0.096000 | [0.027950, 0.178000] | 0.103924 |
| S2 Random Forest | 10 | 0.360000 | [0.120000, 0.650000] | 0.085027 |
| S2 Random Forest | 25 | 0.384000 | [0.144000, 0.664200] | 0.152133 |
| S2 Random Forest | 50 | 0.382000 | [0.149950, 0.666100] | 0.261787 |
| S2 LightGBM (5 seeds) | 10 | 0.460000 | [0.230000, 0.720000] | 0.116142 |
| S2 LightGBM (5 seeds) | 25 | 0.444000 | [0.212000, 0.708100] | 0.206444 |
| S2 LightGBM (5 seeds) | 50 | 0.436000 | [0.192000, 0.714000] | 0.339727 |

## All-transaction queue and uniform random baseline

K=50 is the registered operating capacity. K=25 and K=100 are supporting diagnostics.

| Arm | K | Reviewed | Illicit | Licit | Unlabelled | Unlabelled share | Known-illicit recall | Expected random hits | Observed / expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 legacy XGBoost | 25 | 250 | 2 | 22 | 226 | 0.904000 | 0.003145 | 3.262781 | 0.612974 |
| S1 legacy XGBoost | 50 | 500 | 5 | 39 | 456 | 0.912000 | 0.007862 | 6.525561 | 0.766218 |
| S1 legacy XGBoost | 100 | 1000 | 14 | 97 | 889 | 0.889000 | 0.022013 | 13.051122 | 1.072705 |
| S1 T0 XGBoost refit | 25 | 250 | 2 | 28 | 220 | 0.880000 | 0.003145 | 3.262781 | 0.612974 |
| S1 T0 XGBoost refit | 50 | 500 | 4 | 54 | 442 | 0.884000 | 0.006289 | 6.525561 | 0.612974 |
| S1 T0 XGBoost refit | 100 | 1000 | 7 | 115 | 878 | 0.878000 | 0.011006 | 13.051122 | 0.536352 |
| S2 Random Forest | 25 | 250 | 40 | 0 | 210 | 0.840000 | 0.062893 | 3.262781 | 12.259482 |
| S2 Random Forest | 50 | 500 | 81 | 0 | 419 | 0.838000 | 0.127358 | 6.525561 | 12.412726 |
| S2 Random Forest | 100 | 1000 | 159 | 4 | 837 | 0.837000 | 0.250000 | 13.051122 | 12.182860 |
| S2 LightGBM (5 seeds) | 25 | 250 | 52 | 0 | 198 | 0.792000 | 0.081761 | 3.262781 | 15.937327 |
| S2 LightGBM (5 seeds) | 50 | 500 | 98 | 0 | 402 | 0.804000 | 0.154088 | 6.525561 | 15.017865 |
| S2 LightGBM (5 seeds) | 100 | 1000 | 168 | 3 | 829 | 0.829000 | 0.264151 | 13.051122 | 12.872456 |

## Cutoff ties at K=50

| Arm | Mean tied at cut | Steps with at most five tied | Steps with ties spanning cut |
| --- | --- | --- | --- |
| S1 legacy XGBoost | 688 | 0/10 | 10 |
| S1 T0 XGBoost refit | 368.700000 | 0/10 | 10 |
| S2 Random Forest | 3.300000 | 9/10 | 1 |
| S2 LightGBM (5 seeds) | 1 | 10/10 | 0 |

## Descriptive split at step 43

| Arm | 40–42 mean step AP | 40–42 pooled AP | 43–49 mean step AP | 43–49 pooled AP |
| --- | --- | --- | --- | --- |
| S1 legacy XGBoost | 0.200948 | 0.199342 | 0.080354 | 0.028190 |
| S1 T0 XGBoost refit | 0.200191 | 0.195383 | 0.078772 | 0.027806 |
| S2 Random Forest | 0.882994 | 0.890976 | 0.196692 | 0.078440 |
| S2 LightGBM (5 seeds) | 0.880567 | 0.887891 | 0.259432 | 0.168229 |

## Operating points at the frozen F4 thresholds

| Arm | Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 legacy XGBoost | 0.16880415380001068 | 328 | 3227 | 308 | 7321 | 0.092264 | 0.515723 | 0.156526 |
| S1 T0 XGBoost refit | 0.16374672949314117 | 181 | 1460 | 455 | 9088 | 0.110299 | 0.284591 | 0.158981 |
| S2 Random Forest | 0.5608666666666665 | 366 | 5 | 270 | 10543 | 0.986523 | 0.575472 | 0.726912 |
| S2 LightGBM (5 seeds) | 0.800860653329169 | 357 | 12 | 279 | 10536 | 0.967480 | 0.561321 | 0.710448 |

## Per-step support

The eligible rows and known outcomes are shared across all four arms. Illicit prevalence uses known labels as the denominator.

| Step | Eligible | Labelled | Illicit | Licit | Unlabelled | Illicit prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | 4481 | 1211 | 112 | 1099 | 3270 | 0.092486 |
| 41 | 5342 | 1132 | 116 | 1016 | 4210 | 0.102473 |
| 42 | 7140 | 2154 | 239 | 1915 | 4986 | 0.110956 |
| 43 | 5063 | 1370 | 24 | 1346 | 3693 | 0.017518 |
| 44 | 4975 | 1591 | 24 | 1567 | 3384 | 0.015085 |
| 45 | 5598 | 1221 | 5 | 1216 | 4377 | 0.004095 |
| 46 | 3519 | 712 | 2 | 710 | 2807 | 0.002809 |
| 47 | 5121 | 846 | 22 | 824 | 4275 | 0.026005 |
| 48 | 2954 | 471 | 36 | 435 | 2483 | 0.076433 |
| 49 | 2454 | 476 | 56 | 420 | 1978 | 0.117647 |

## Per-step AP and Q-lab precision / recall

| Step | Arm | AP | P / R at 10 | P / R at 25 | P / R at 50 |
| --- | --- | --- | --- | --- | --- |
| 40 | S1 legacy XGBoost | 0.160165 | 0.200000 / 0.017857 | 0.200000 / 0.044643 | 0.280000 / 0.125000 |
| 40 | S1 T0 XGBoost refit | 0.168466 | 0.300000 / 0.026786 | 0.200000 / 0.044643 | 0.300000 / 0.133929 |
| 40 | S2 Random Forest | 0.787389 | 1.000000 / 0.089286 | 1.000000 / 0.223214 | 1.000000 / 0.446429 |
| 40 | S2 LightGBM (5 seeds) | 0.778240 | 1.000000 / 0.089286 | 1.000000 / 0.223214 | 1.000000 / 0.446429 |
| 41 | S1 legacy XGBoost | 0.239635 | 0.200000 / 0.017241 | 0.320000 / 0.068966 | 0.320000 / 0.137931 |
| 41 | S1 T0 XGBoost refit | 0.235758 | 0.100000 / 0.008621 | 0.320000 / 0.068966 | 0.300000 / 0.129310 |
| 41 | S2 Random Forest | 0.960300 | 1.000000 / 0.086207 | 1.000000 / 0.215517 | 1.000000 / 0.431034 |
| 41 | S2 LightGBM (5 seeds) | 0.964450 | 1.000000 / 0.086207 | 1.000000 / 0.215517 | 1.000000 / 0.431034 |
| 42 | S1 legacy XGBoost | 0.203043 | 0.300000 / 0.012552 | 0.320000 / 0.033473 | 0.260000 / 0.054393 |
| 42 | S1 T0 XGBoost refit | 0.196347 | 0.100000 / 0.004184 | 0.080000 / 0.008368 | 0.220000 / 0.046025 |
| 42 | S2 Random Forest | 0.901292 | 1.000000 / 0.041841 | 1.000000 / 0.104603 | 1.000000 / 0.209205 |
| 42 | S2 LightGBM (5 seeds) | 0.899011 | 1.000000 / 0.041841 | 1.000000 / 0.104603 | 1.000000 / 0.209205 |
| 43 | S1 legacy XGBoost | 0.018380 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.020000 / 0.041667 |
| 43 | S1 T0 XGBoost refit | 0.022253 | 0.100000 / 0.041667 | 0.040000 / 0.041667 | 0.040000 / 0.083333 |
| 43 | S2 Random Forest | 0.030866 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.040000 / 0.083333 |
| 43 | S2 LightGBM (5 seeds) | 0.070412 | 0.100000 / 0.041667 | 0.120000 / 0.125000 | 0.100000 / 0.208333 |
| 44 | S1 legacy XGBoost | 0.054823 | 0.100000 / 0.041667 | 0.040000 / 0.041667 | 0.020000 / 0.041667 |
| 44 | S1 T0 XGBoost refit | 0.040827 | 0.100000 / 0.041667 | 0.080000 / 0.083333 | 0.040000 / 0.083333 |
| 44 | S2 Random Forest | 0.065395 | 0.100000 / 0.041667 | 0.040000 / 0.041667 | 0.040000 / 0.083333 |
| 44 | S2 LightGBM (5 seeds) | 0.078821 | 0.200000 / 0.083333 | 0.160000 / 0.166667 | 0.100000 / 0.208333 |
| 45 | S1 legacy XGBoost | 0.003738 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| 45 | S1 T0 XGBoost refit | 0.003720 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| 45 | S2 Random Forest | 0.012634 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| 45 | S2 LightGBM (5 seeds) | 0.015369 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| 46 | S1 legacy XGBoost | 0.251603 | 0.100000 / 0.500000 | 0.040000 / 0.500000 | 0.020000 / 0.500000 |
| 46 | S1 T0 XGBoost refit | 0.251600 | 0.100000 / 0.500000 | 0.040000 / 0.500000 | 0.020000 / 0.500000 |
| 46 | S2 Random Forest | 0.509434 | 0.100000 / 0.500000 | 0.040000 / 0.500000 | 0.020000 / 0.500000 |
| 46 | S2 LightGBM (5 seeds) | 0.513699 | 0.100000 / 0.500000 | 0.040000 / 0.500000 | 0.020000 / 0.500000 |
| 47 | S1 legacy XGBoost | 0.026254 | 0.100000 / 0.045455 | 0.040000 / 0.045455 | 0.020000 / 0.045455 |
| 47 | S1 T0 XGBoost refit | 0.025131 | 0.000000 / 0.000000 | 0.040000 / 0.045455 | 0.020000 / 0.045455 |
| 47 | S2 Random Forest | 0.076978 | 0.000000 / 0.000000 | 0.040000 / 0.045455 | 0.060000 / 0.136364 |
| 47 | S2 LightGBM (5 seeds) | 0.122562 | 0.200000 / 0.090909 | 0.160000 / 0.181818 | 0.100000 / 0.227273 |
| 48 | S1 legacy XGBoost | 0.086124 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| 48 | S1 T0 XGBoost refit | 0.086331 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| 48 | S2 Random Forest | 0.269106 | 0.200000 / 0.055556 | 0.280000 / 0.194444 | 0.280000 / 0.388889 |
| 48 | S2 LightGBM (5 seeds) | 0.475737 | 0.500000 / 0.138889 | 0.480000 / 0.333333 | 0.480000 / 0.666667 |
| 49 | S1 legacy XGBoost | 0.121554 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.020000 / 0.017857 |
| 49 | S1 T0 XGBoost refit | 0.121544 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.020000 / 0.017857 |
| 49 | S2 Random Forest | 0.412428 | 0.200000 / 0.035714 | 0.440000 / 0.196429 | 0.380000 / 0.339286 |
| 49 | S2 LightGBM (5 seeds) | 0.539424 | 0.500000 / 0.089286 | 0.480000 / 0.214286 | 0.560000 / 0.500000 |

## Per-step Q-all outcomes and cutoff ties at K=50

| Step | Arm | Reviewed | Illicit | Licit | Unlabelled | Unlabelled share | Known-illicit recall | Random hits expected | Observed / expected | Strictly above | Tied at cut | Tied and reviewed | Spans cut |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | S1 legacy XGBoost | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 1.249721 | 0.000000 | 47 | 16 | 3 | Yes |
| 40 | S1 T0 XGBoost refit | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 1.249721 | 0.000000 | 1 | 64 | 49 | Yes |
| 40 | S2 Random Forest | 50 | 30 | 0 | 20 | 0.400000 | 0.267857 | 1.249721 | 24.005357 | 48 | 2 | 2 | No |
| 40 | S2 LightGBM (5 seeds) | 50 | 36 | 0 | 14 | 0.280000 | 0.321429 | 1.249721 | 28.806429 | 49 | 1 | 1 | No |
| 41 | S1 legacy XGBoost | 50 | 0 | 5 | 45 | 0.900000 | 0.000000 | 1.085736 | 0.000000 | 1 | 1149 | 49 | Yes |
| 41 | S1 T0 XGBoost refit | 50 | 1 | 5 | 44 | 0.880000 | 0.008621 | 1.085736 | 0.921034 | 32 | 1149 | 18 | Yes |
| 41 | S2 Random Forest | 50 | 22 | 0 | 28 | 0.560000 | 0.189655 | 1.085736 | 20.262759 | 49 | 1 | 1 | No |
| 41 | S2 LightGBM (5 seeds) | 50 | 25 | 0 | 25 | 0.500000 | 0.215517 | 1.085736 | 23.025862 | 49 | 1 | 1 | No |
| 42 | S1 legacy XGBoost | 50 | 3 | 5 | 42 | 0.840000 | 0.012552 | 1.673669 | 1.792469 | 4 | 1484 | 46 | Yes |
| 42 | S1 T0 XGBoost refit | 50 | 1 | 14 | 35 | 0.700000 | 0.004184 | 1.673669 | 0.597490 | 12 | 55 | 38 | Yes |
| 42 | S2 Random Forest | 50 | 28 | 0 | 22 | 0.440000 | 0.117155 | 1.673669 | 16.729707 | 29 | 22 | 21 | Yes |
| 42 | S2 LightGBM (5 seeds) | 50 | 35 | 0 | 15 | 0.300000 | 0.146444 | 1.673669 | 20.912134 | 49 | 1 | 1 | No |
| 43 | S1 legacy XGBoost | 50 | 0 | 1 | 49 | 0.980000 | 0.000000 | 0.237014 | 0.000000 | 40 | 693 | 10 | Yes |
| 43 | S1 T0 XGBoost refit | 50 | 0 | 1 | 49 | 0.980000 | 0.000000 | 0.237014 | 0.000000 | 29 | 23 | 21 | Yes |
| 43 | S2 Random Forest | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.237014 | 0.000000 | 49 | 1 | 1 | No |
| 43 | S2 LightGBM (5 seeds) | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.237014 | 0.000000 | 49 | 1 | 1 | No |
| 44 | S1 legacy XGBoost | 50 | 1 | 1 | 48 | 0.960000 | 0.041667 | 0.241206 | 4.145833 | 27 | 24 | 23 | Yes |
| 44 | S1 T0 XGBoost refit | 50 | 1 | 1 | 48 | 0.960000 | 0.041667 | 0.241206 | 4.145833 | 45 | 13 | 5 | Yes |
| 44 | S2 Random Forest | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.241206 | 0.000000 | 49 | 1 | 1 | No |
| 44 | S2 LightGBM (5 seeds) | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.241206 | 0.000000 | 49 | 1 | 1 | No |
| 45 | S1 legacy XGBoost | 50 | 0 | 7 | 43 | 0.860000 | 0.000000 | 0.044659 | 0.000000 | 7 | 961 | 43 | Yes |
| 45 | S1 T0 XGBoost refit | 50 | 0 | 14 | 36 | 0.720000 | 0.000000 | 0.044659 | 0.000000 | 38 | 961 | 12 | Yes |
| 45 | S2 Random Forest | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.044659 | 0.000000 | 49 | 1 | 1 | No |
| 45 | S2 LightGBM (5 seeds) | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.044659 | 0.000000 | 49 | 1 | 1 | No |
| 46 | S1 legacy XGBoost | 50 | 1 | 3 | 46 | 0.920000 | 0.500000 | 0.028417 | 35.190000 | 2 | 804 | 48 | Yes |
| 46 | S1 T0 XGBoost refit | 50 | 1 | 7 | 42 | 0.840000 | 0.500000 | 0.028417 | 35.190000 | 15 | 804 | 35 | Yes |
| 46 | S2 Random Forest | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.028417 | 0.000000 | 49 | 1 | 1 | No |
| 46 | S2 LightGBM (5 seeds) | 50 | 1 | 0 | 49 | 0.980000 | 0.500000 | 0.028417 | 35.190000 | 49 | 1 | 1 | No |
| 47 | S1 legacy XGBoost | 50 | 0 | 5 | 45 | 0.900000 | 0.000000 | 0.214802 | 0.000000 | 41 | 1173 | 9 | Yes |
| 47 | S1 T0 XGBoost refit | 50 | 0 | 4 | 46 | 0.920000 | 0.000000 | 0.214802 | 0.000000 | 45 | 42 | 5 | Yes |
| 47 | S2 Random Forest | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.214802 | 0.000000 | 48 | 2 | 2 | No |
| 47 | S2 LightGBM (5 seeds) | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.214802 | 0.000000 | 49 | 1 | 1 | No |
| 48 | S1 legacy XGBoost | 50 | 0 | 12 | 38 | 0.760000 | 0.000000 | 0.609343 | 0.000000 | 12 | 532 | 38 | Yes |
| 48 | S1 T0 XGBoost refit | 50 | 0 | 8 | 42 | 0.840000 | 0.000000 | 0.609343 | 0.000000 | 29 | 532 | 21 | Yes |
| 48 | S2 Random Forest | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.609343 | 0.000000 | 49 | 1 | 1 | No |
| 48 | S2 LightGBM (5 seeds) | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 0.609343 | 0.000000 | 49 | 1 | 1 | No |
| 49 | S1 legacy XGBoost | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 1.140994 | 0.000000 | 33 | 44 | 17 | Yes |
| 49 | S1 T0 XGBoost refit | 50 | 0 | 0 | 50 | 1.000000 | 0.000000 | 1.140994 | 0.000000 | 23 | 44 | 27 | Yes |
| 49 | S2 Random Forest | 50 | 1 | 0 | 49 | 0.980000 | 0.017857 | 1.140994 | 0.876429 | 49 | 1 | 1 | No |
| 49 | S2 LightGBM (5 seeds) | 50 | 1 | 0 | 49 | 0.980000 | 0.017857 | 1.140994 | 0.876429 | 49 | 1 | 1 | No |

## Interpretation and limitations

- Conclusions apply to the ten evaluated steps. The limited known-illicit support in steps 45 and 46 makes AP sensitive to individual ranks. Both steps enter the registered analysis. An interval that contains zero leaves the difference inconclusive for this period.

- The RF contrast compares modelling configurations at the same S2 feature contract. The legacy contrast compares the full scenarios. The descriptive S1 refit contrast matches the training horizon while feature availability and model families still differ.

- A1 remains unverified. In development the matched two-step embargo changed the S2 LightGBM AP from 0.935226 to 0.904803. This sensitivity and the prior exposure recorded in the [holdout ledger](test-ledger.md) limit generalisation.

- Unknown labels stay unlabelled. Q-all measures recovery of known illicit cases. F4 thresholds provide secondary operating points and the review queue uses top 50 per step.
