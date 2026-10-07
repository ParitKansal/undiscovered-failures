# Main study: results

Pre-registered in `PREREG.md` (commit `abafcac`, 2026-10-08). Run on 2026-10-08 on Google Colab (24 CPU cores) with
the code at that commit: 450 jobs, 283,500 rows (complete), 11.3 minutes. Executed notebook: `runs/main_executed.ipynb`.
As in the pilots, joblib warned once that a worker stopped; all 450 jobs finished and the row count is complete.

## Pre-registered verdicts

| Rule | Planted failures (45 cells) | Natural failures (135 cells) |
|---|---|---|
| **H1** the unseen-share estimate is unbiased (\|mean error\| <= 0.04 in >= 90% of cells) | **SUPPORTED** (95.6%) | **SUPPORTED** (100%) |
| **H2** its calibrated 95% interval is honest (coverage >= 0.90 in >= 80% of cells) | **NOT SUPPORTED** (71.1%) | **SUPPORTED** (95.6%) |
| **H3** ACE is closer to the true number of types than the raw count (>= 90% of cells) | **NOT SUPPORTED** (88.9%) | **NOT SUPPORTED** (85.2%) |
| H4 (reported) bias-calibrated ACE interval coverage at 300 / 1,000 / 3,000 rows | 0.65 / 0.77 / 0.84 | 0.73 / 0.74 / 0.81 |
| H5 (reported) Chao-Jost beats Turing's f1/n on mean absolute error | 71.1% of cells | 84.4% of cells |
| Typical error of the unseen share on one test (median cell MAE) at 300 / 1,000 / 3,000 rows | 0.146 / 0.050 / 0.011 | 0.086 / 0.023 / 0.005 |

## Reading, as fixed in PREREG.md

- **Natural failures:** H1 and H2 are both supported. The unseen share of errors can be estimated without bias and
  with an honest interval from test results alone.
- **Planted failures:** H1 is supported but H2 is not: the estimate is unbiased, but its uncertainty on one test is
  understated. The worst cells are Adult at 300 rows (bias up to +0.076, interval coverage 0.70-0.84), where the
  planted types are few and the test sees only a few dozen errors.
- **H3 is not supported in either group.** It narrowly misses (88.9% and 85.2% against the required 90%). ACE reduces
  the undercount on average in every pooled table, but not in every cell.
- **H4:** as expected, no interval for the number of types reaches 95% coverage.

These verdicts are reported as they came out. Nothing in the design or the rules was changed after the
pre-registration.

## Run log

```
ALL 16 ESTIMATOR CHECKS PASSED
450 jobs on 24 CPU cores
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 24 concurrent workers.
[Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:   53.0s
/usr/local/lib/python3.13/dist-packages/joblib/externals/loky/process_executor.py:787: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  warnings.warn(
[Parallel(n_jobs=-1)]: Done 114 tasks      | elapsed:  3.7min
[Parallel(n_jobs=-1)]: Done 240 tasks      | elapsed:  7.9min
[Parallel(n_jobs=-1)]: Done 402 tasks      | elapsed: 10.7min
[Parallel(n_jobs=-1)]: Done 450 out of 450 | elapsed: 11.3min finished
wrote /content/drive/MyDrive/undiscovered-failures/main.csv 283500 rows
```

## Analysis output (decisions and key numbers)

```
162000 rows, 180 cells

== PLANTED (45 cells)
H1 unseen share unbiased, |mean signed error| <= 0.04: 95.6% of cells -> SUPPORTED (needs >= 90%)
H2 unseen-share interval coverage >= 0.90: 71.1% of cells -> NOT SUPPORTED (needs >= 80%)
H3 ACE closer to the truth than the raw count: 88.9% of cells -> NOT SUPPORTED (needs >= 90%)
H4 (reported) bias-calibrated ACE interval coverage by budget: {300: 0.648, 1000: 0.769, 3000: 0.836}
H5 (reported) Chao-Jost beats Turing f1/n on mean |error|: 71.1% of cells
Reported: typical error per test (cell MAE) by budget: {300: 0.146, 1000: 0.05, 3000: 0.011}
Worst cells for H1:
dataset    model   types  budget  bias   mae  cov_unseen
  adult   logreg planted     300 0.076 0.265       0.703
  adult   forest planted     300 0.050 0.203       0.783
  adult boosting planted     300 0.035 0.147       0.840
  magic   logreg planted    1000 0.015 0.068       0.953
  magic boosting planted    1000 0.013 0.060       0.946

== NATURAL (135 cells)
H1 unseen share unbiased, |mean signed error| <= 0.04: 100.0% of cells -> SUPPORTED (needs >= 90%)
H2 unseen-share interval coverage >= 0.90: 95.6% of cells -> SUPPORTED (needs >= 80%)
H3 ACE closer to the truth than the raw count: 85.2% of cells -> NOT SUPPORTED (needs >= 90%)
H4 (reported) bias-calibrated ACE interval coverage by budget: {300: 0.73, 1000: 0.741, 3000: 0.814}
H5 (reported) Chao-Jost beats Turing f1/n on mean |error|: 84.4% of cells
Reported: typical error per test (cell MAE) by budget: {300: 0.086, 1000: 0.023, 3000: 0.005}
Worst cells for H1:
dataset    model            types  budget  bias   mae  cov_unseen
  magic boosting  natural_grid100     300 0.025 0.103       0.913
  magic boosting natural_kmeans25     300 0.022 0.085       0.942
  magic   forest  natural_grid100    1000 0.021 0.048       0.938
   bank   forest  natural_grid100     300 0.019 0.114       0.909
  magic boosting  natural_grid100    1000 0.018 0.046       0.938

== Key numbers ==
                         rel_s_obs  rel_ace  rel_jackknife1  rel_ace_bc  cov_ace_p  cov_jack_p  unseen_true  unseen_err  unseen_err_tur  cov_unseen  new_ratio
types            budget                                                                                                                                       
kmeans10         300        -0.172   -0.099          -0.081      -0.082      0.766       0.777        0.018       0.001           0.002       0.920      0.465
                 1000       -0.100   -0.056          -0.056      -0.043      0.808       0.824        0.003      -0.000          -0.000       0.877      0.284
                 3000       -0.053   -0.007          -0.014       0.002      0.877       0.873        0.001       0.000           0.000       0.909      0.451
kmeans25         300        -0.291   -0.155          -0.112      -0.135      0.714       0.705        0.088       0.001           0.004       0.940      0.773
                 1000       -0.140   -0.083          -0.060      -0.061      0.779       0.794        0.011       0.001           0.001       0.944      0.691
                 3000       -0.069   -0.001          -0.012       0.008      0.850       0.853        0.002       0.000           0.000       0.907      0.682
kmeans50         300        -0.421   -0.159          -0.176      -0.146      0.669       0.571        0.226       0.003           0.008       0.933      0.891
                 1000       -0.186   -0.113          -0.056      -0.087      0.697       0.733        0.038       0.002           0.003       0.961      0.827
                 3000       -0.086   -0.030          -0.019      -0.016      0.814       0.841        0.006       0.001           0.001       0.948      0.784
natural_grid100  300        -0.670   -0.143          -0.445      -0.103      0.685       0.318        0.512       0.008           0.015       0.924      0.927
                 1000       -0.361   -0.137          -0.110      -0.124      0.660       0.613        0.175       0.007           0.009       0.937      0.897
                 3000       -0.140   -0.068          -0.003      -0.045      0.745       0.822        0.029       0.005           0.005       0.953      0.847
natural_grid25   300        -0.330   -0.117          -0.100      -0.100      0.784       0.784        0.152       0.003           0.009       0.939      0.817
                 1000       -0.131   -0.079          -0.031      -0.054      0.786       0.834        0.020       0.001           0.001       0.952      0.681
                 3000       -0.056   -0.026          -0.008      -0.012      0.838       0.872        0.003       0.001           0.001       0.930      0.543
natural_kmeans25 300        -0.373   -0.152          -0.149      -0.136      0.721       0.678        0.154       0.004           0.010       0.935      0.826
                 1000       -0.169   -0.096          -0.063      -0.071      0.777       0.798        0.021       0.002           0.002       0.954      0.728
                 3000       -0.084   -0.010          -0.019       0.003      0.860       0.872        0.004       0.001           0.001       0.950      0.682
planted          300        -0.523   -0.161          -0.302       0.026      0.648       0.596        0.333       0.014           0.026       0.867      0.810
                 1000       -0.247   -0.086          -0.065      -0.063      0.769       0.812        0.085       0.005           0.009       0.937      0.780
                 3000       -0.095   -0.038           0.001      -0.014      0.836       0.880        0.013       0.003           0.003       0.954      0.684
```

The full summary tables are in the executed notebook.
