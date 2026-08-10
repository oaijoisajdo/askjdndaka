# Adversarial experiment analysis

Loaded **20 paired JSON files** covering **4 model families** and **5 seeds**.

## Interpretation rules

- PGD accuracy concerns the posterior-predictive classifier and is diagnostic, not a certificate.
- `p_safe_point` is the pessimistic ranking estimate (unknown counted unsafe).
- Lower/upper robustness bounds support threshold claims; inconclusive cases remain unknown.
- Alignment metrics were calculated within each seed. Seed-input rows were not pooled.
- All deltas are robust-training minus standard-training unless explicitly named an inverse-alignment gain.
- Predictive margin was not present in the JSON. The script used entropy, mutual information, expected entropy, and 1-confidence. Add the top-two probability margin to future experiment payloads; existing runs cannot recover it without checkpoint inference.

## Clean/PGD trade-off

| family | train_epsilon | rob_lam | n_seeds | clean_delta | pgd_auc_delta |
| --- | --- | --- | --- | --- | --- |
| bbb | 0.080 | 0.250 | 5.000 | -0.009 | 0.267 |
| deterministic | 0.080 | 0.250 | 5.000 | -0.000 | 0.540 |
| mc_dropout | 0.080 | 0.250 | 5.000 | -0.003 | 0.486 |
| vogn | 0.080 | 0.250 | 5.000 | -0.045 | -0.003 |

## Posterior robust-mass effect

| family | train_epsilon | rob_lam | eval_epsilon | n_seeds | mean | sd | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbb | 0.080 | 0.250 | 0.010 | 5.000 | 0.037 | 0.014 | 0.020 | 0.055 |
| bbb | 0.080 | 0.250 | 0.050 | 5.000 | 0.940 | 0.014 | 0.922 | 0.958 |
| bbb | 0.080 | 0.250 | 0.070 | 5.000 | 0.860 | 0.020 | 0.835 | 0.884 |
| bbb | 0.080 | 0.250 | 0.080 | 5.000 | 0.795 | 0.026 | 0.763 | 0.828 |
| bbb | 0.080 | 0.250 | 0.100 | 5.000 | 0.591 | 0.021 | 0.564 | 0.618 |
| deterministic | 0.080 | 0.250 | 0.010 | 5.000 | 0.012 | 0.018 | -0.010 | 0.034 |
| deterministic | 0.080 | 0.250 | 0.050 | 5.000 | 0.940 | 0.014 | 0.922 | 0.958 |
| deterministic | 0.080 | 0.250 | 0.070 | 5.000 | 0.928 | 0.027 | 0.895 | 0.961 |
| deterministic | 0.080 | 0.250 | 0.080 | 5.000 | 0.900 | 0.032 | 0.861 | 0.939 |
| deterministic | 0.080 | 0.250 | 0.100 | 5.000 | 0.760 | 0.032 | 0.721 | 0.799 |
| mc_dropout | 0.080 | 0.250 | 0.010 | 5.000 | 0.022 | 0.008 | 0.013 | 0.032 |
| mc_dropout | 0.080 | 0.250 | 0.050 | 5.000 | 0.929 | 0.010 | 0.916 | 0.941 |
| mc_dropout | 0.080 | 0.250 | 0.070 | 5.000 | 0.885 | 0.008 | 0.875 | 0.895 |
| mc_dropout | 0.080 | 0.250 | 0.080 | 5.000 | 0.841 | 0.008 | 0.831 | 0.851 |
| mc_dropout | 0.080 | 0.250 | 0.100 | 5.000 | 0.694 | 0.012 | 0.679 | 0.708 |
| vogn | 0.080 | 0.250 | 0.010 | 5.000 | 0.101 | 0.013 | 0.085 | 0.116 |
| vogn | 0.080 | 0.250 | 0.050 | 5.000 | 0.692 | 0.028 | 0.658 | 0.727 |
| vogn | 0.080 | 0.250 | 0.070 | 5.000 | 0.536 | 0.009 | 0.525 | 0.547 |
| vogn | 0.080 | 0.250 | 0.080 | 5.000 | 0.439 | 0.009 | 0.428 | 0.450 |
| vogn | 0.080 | 0.250 | 0.100 | 5.000 | 0.147 | 0.023 | 0.119 | 0.175 |

Here `mean` is the paired change in the mean pessimistic `p_safe_point` across seeds.

## Inverse uncertainty-safety alignment effect

| family | train_epsilon | rob_lam | eval_epsilon | score | subset | n_seeds | mean | sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbb | 0.080 | 0.250 | 0.010 | mutual_information | clean_correct | 1.000 | -0.405 |  |
| bbb | 0.080 | 0.250 | 0.010 | one_minus_confidence | clean_correct | 1.000 | -0.417 |  |
| bbb | 0.080 | 0.250 | 0.050 | mutual_information | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.050 | one_minus_confidence | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.070 | mutual_information | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.070 | one_minus_confidence | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.080 | mutual_information | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.080 | one_minus_confidence | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.100 | mutual_information | clean_correct | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.100 | one_minus_confidence | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.010 | mutual_information | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.010 | one_minus_confidence | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.050 | mutual_information | clean_correct | 1.000 | -0.084 |  |
| deterministic | 0.080 | 0.250 | 0.050 | one_minus_confidence | clean_correct | 1.000 | 0.109 |  |
| deterministic | 0.080 | 0.250 | 0.070 | mutual_information | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.070 | one_minus_confidence | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.080 | mutual_information | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.080 | one_minus_confidence | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.100 | mutual_information | clean_correct | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.100 | one_minus_confidence | clean_correct | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.010 | mutual_information | clean_correct | 5.000 | -0.445 | 0.082 |
| mc_dropout | 0.080 | 0.250 | 0.010 | one_minus_confidence | clean_correct | 5.000 | -0.452 | 0.070 |
| mc_dropout | 0.080 | 0.250 | 0.050 | mutual_information | clean_correct | 5.000 | 0.182 | 0.067 |
| mc_dropout | 0.080 | 0.250 | 0.050 | one_minus_confidence | clean_correct | 5.000 | 0.204 | 0.068 |
| mc_dropout | 0.080 | 0.250 | 0.070 | mutual_information | clean_correct | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.070 | one_minus_confidence | clean_correct | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.080 | mutual_information | clean_correct | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.080 | one_minus_confidence | clean_correct | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.100 | mutual_information | clean_correct | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.100 | one_minus_confidence | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.010 | mutual_information | clean_correct | 5.000 | -0.415 | 0.057 |
| vogn | 0.080 | 0.250 | 0.010 | one_minus_confidence | clean_correct | 5.000 | -0.535 | 0.026 |
| vogn | 0.080 | 0.250 | 0.050 | mutual_information | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.050 | one_minus_confidence | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.070 | mutual_information | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.070 | one_minus_confidence | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.080 | mutual_information | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.080 | one_minus_confidence | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.100 | mutual_information | clean_correct | 0.000 |  |  |
| vogn | 0.080 | 0.250 | 0.100 | one_minus_confidence | clean_correct | 0.000 |  |  |

A positive inverse-alignment gain means the risk score became more negatively associated with safe posterior mass after robust training.

## PGD-failure detection effect

| family | train_epsilon | rob_lam | eval_epsilon | score | n_seeds | mean | sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bbb | 0.080 | 0.250 | 0.010 | mutual_information | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.010 | one_minus_confidence | 0.000 |  |  |
| bbb | 0.080 | 0.250 | 0.050 | mutual_information | 3.000 | 0.036 | 0.015 |
| bbb | 0.080 | 0.250 | 0.050 | one_minus_confidence | 3.000 | -0.003 | 0.013 |
| bbb | 0.080 | 0.250 | 0.070 | mutual_information | 5.000 | -0.003 | 0.122 |
| bbb | 0.080 | 0.250 | 0.070 | one_minus_confidence | 5.000 | -0.045 | 0.111 |
| bbb | 0.080 | 0.250 | 0.080 | mutual_information | 5.000 | -0.017 | 0.060 |
| bbb | 0.080 | 0.250 | 0.080 | one_minus_confidence | 5.000 | -0.075 | 0.039 |
| bbb | 0.080 | 0.250 | 0.100 | mutual_information | 5.000 | -0.102 | 0.065 |
| bbb | 0.080 | 0.250 | 0.100 | one_minus_confidence | 5.000 | -0.134 | 0.056 |
| deterministic | 0.080 | 0.250 | 0.010 | mutual_information | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.010 | one_minus_confidence | 0.000 |  |  |
| deterministic | 0.080 | 0.250 | 0.050 | mutual_information | 1.000 | -0.320 |  |
| deterministic | 0.080 | 0.250 | 0.050 | one_minus_confidence | 1.000 | 0.098 |  |
| deterministic | 0.080 | 0.250 | 0.070 | mutual_information | 4.000 | 0.148 | 0.561 |
| deterministic | 0.080 | 0.250 | 0.070 | one_minus_confidence | 4.000 | 0.198 | 0.028 |
| deterministic | 0.080 | 0.250 | 0.080 | mutual_information | 4.000 | 0.165 | 0.547 |
| deterministic | 0.080 | 0.250 | 0.080 | one_minus_confidence | 4.000 | 0.185 | 0.092 |
| deterministic | 0.080 | 0.250 | 0.100 | mutual_information | 5.000 | 0.107 | 0.400 |
| deterministic | 0.080 | 0.250 | 0.100 | one_minus_confidence | 5.000 | 0.021 | 0.134 |
| mc_dropout | 0.080 | 0.250 | 0.010 | mutual_information | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.010 | one_minus_confidence | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.050 | mutual_information | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.050 | one_minus_confidence | 0.000 |  |  |
| mc_dropout | 0.080 | 0.250 | 0.070 | mutual_information | 3.000 | 0.048 | 0.088 |
| mc_dropout | 0.080 | 0.250 | 0.070 | one_minus_confidence | 3.000 | 0.045 | 0.066 |
| mc_dropout | 0.080 | 0.250 | 0.080 | mutual_information | 3.000 | -0.025 | 0.056 |
| mc_dropout | 0.080 | 0.250 | 0.080 | one_minus_confidence | 3.000 | -0.019 | 0.046 |
| mc_dropout | 0.080 | 0.250 | 0.100 | mutual_information | 5.000 | -0.170 | 0.127 |
| mc_dropout | 0.080 | 0.250 | 0.100 | one_minus_confidence | 5.000 | -0.166 | 0.134 |
| vogn | 0.080 | 0.250 | 0.010 | mutual_information | 1.000 | 0.000 |  |
| vogn | 0.080 | 0.250 | 0.010 | one_minus_confidence | 1.000 | 0.000 |  |
| vogn | 0.080 | 0.250 | 0.050 | mutual_information | 5.000 | -0.012 | 0.053 |
| vogn | 0.080 | 0.250 | 0.050 | one_minus_confidence | 5.000 | -0.138 | 0.084 |
| vogn | 0.080 | 0.250 | 0.070 | mutual_information | 5.000 | -0.008 | 0.062 |
| vogn | 0.080 | 0.250 | 0.070 | one_minus_confidence | 5.000 | -0.154 | 0.051 |
| vogn | 0.080 | 0.250 | 0.080 | mutual_information | 5.000 | 0.052 | 0.077 |
| vogn | 0.080 | 0.250 | 0.080 | one_minus_confidence | 5.000 | -0.109 | 0.065 |
| vogn | 0.080 | 0.250 | 0.100 | mutual_information | 5.000 | 0.093 | 0.058 |
| vogn | 0.080 | 0.250 | 0.100 | one_minus_confidence | 5.000 | -0.062 | 0.052 |

Here `mean` is the paired AUROC change on clean-correct inputs. Always interpret it beside failure prevalence and robust accuracy: discrimination may fall when nearly every input belongs to one class.

## Quality control

- Errors: **0**
- Warnings: **150**

See `tables/quality_checks.csv` before interpreting the results. A failed or inconclusive verification is not coded as non-robust.
