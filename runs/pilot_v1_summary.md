# Pilot v1 (2026-10-07)

Exploratory pilot, not pre-registered. Code at commit `5326254`, run on Google Colab (24 CPU cores). Design:
Adult, Bank Marketing, Covertype and MAGIC; logistic regression, random forest and gradient boosting; 3 seeds;
label-flip rates 0.6 and 0.9 inside planted groups; 15, 30 or 60 planted failure types; test budgets of 300, 1,000
and 3,000 rows with 30 draws each; bootstrap intervals with 200 draws. 216 jobs, 77,760 rows, 3.7 minutes.
The executed notebook is `pilot_v1_executed.ipynb`; its cells 3-4 also re-ran the v0 settings with this code
(estimates identical to v0, plus the new interval columns).

Note: joblib warned once that a worker stopped while jobs were queued (a timeout or memory warning); all 216 jobs
finished and the row count is complete (216 x 360).

## Run log

```
ALL 12 ESTIMATOR CHECKS PASSED
216 jobs on 24 CPU cores
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 24 concurrent workers.
[Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:   53.8s
[Parallel(n_jobs=-1)]: Done 114 tasks      | elapsed:  2.5min
/usr/local/lib/python3.13/dist-packages/joblib/externals/loky/process_executor.py:787: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  warnings.warn(
[Parallel(n_jobs=-1)]: Done 213 out of 216 | elapsed:  3.6min remaining:    3.1s
[Parallel(n_jobs=-1)]: Done 216 out of 216 | elapsed:  3.7min finished
wrote /content/drive/MyDrive/undiscovered-failures/pilot_v1.csv 77760 rows
```

## Summary output

```
== By type definition and test budget ==
                 rel_s_obs  rel_chao1  rel_ichao1  rel_ace  rel_jackknife1  cov_chao1_ln  cov_chao1_b  cov_ace_b  cov_jack_b  new_real  new_ratio  true_types
types    budget                                                                                                                                              
kmeans10 300        -0.197     -0.166      -0.155   -0.121          -0.101         0.380        0.538      0.658       0.691     0.550      0.431      10.000
         1000       -0.113     -0.102      -0.100   -0.068          -0.065         0.517        0.591      0.692       0.706     0.338      0.261      10.000
         3000       -0.062     -0.047      -0.046   -0.011          -0.018         0.658        0.746      0.817       0.832     0.295      0.378      10.000
kmeans25 300        -0.317     -0.197      -0.171   -0.173          -0.138         0.828        0.588      0.588       0.616     2.644      0.764      25.000
         1000       -0.160     -0.104      -0.092   -0.100          -0.070         0.667        0.659      0.650       0.732     1.381      0.651      25.000
         3000       -0.078     -0.038      -0.030   -0.009          -0.017         0.687        0.728      0.796       0.840     0.970      0.623      25.000
kmeans50 300        -0.442     -0.236      -0.198   -0.195          -0.213         0.827        0.539      0.567       0.428     7.251      0.883      50.000
         1000       -0.215     -0.126      -0.108   -0.138          -0.080         0.860        0.638      0.521       0.658     3.995      0.772      50.000
         3000       -0.101     -0.044      -0.032   -0.041          -0.023         0.784        0.787      0.731       0.852     2.428      0.744      50.000
planted  300        -0.486     -0.286      -0.260   -0.135          -0.263         0.807        0.508      0.652       0.462     4.284      0.805      27.037
         1000       -0.216     -0.119      -0.097   -0.070          -0.046         0.769        0.697      0.709       0.812     2.412      0.763      27.037
         3000       -0.075     -0.035      -0.024   -0.024           0.010         0.668        0.828      0.820       0.914     1.218      0.677      27.037

== Planted types, by model and budget ==
                 rel_s_obs  rel_chao1  rel_ichao1  rel_ace  rel_jackknife1  cov_chao1_ln  cov_chao1_b  cov_ace_b  cov_jack_b  new_real  new_ratio  true_types
model    budget                                                                                                                                              
boosting 300        -0.411     -0.239      -0.211   -0.120          -0.190         0.790        0.572      0.690       0.566     4.000      0.801      27.139
         1000       -0.184     -0.113      -0.096   -0.088          -0.050         0.754        0.676      0.675       0.792     2.056      0.747      27.139
         3000       -0.072     -0.035      -0.025   -0.029          -0.001         0.658        0.790      0.801       0.891     1.157      0.657      27.139
forest   300        -0.477     -0.275      -0.249   -0.118          -0.244         0.831        0.524      0.657       0.501     4.274      0.799      26.875
         1000       -0.208     -0.111      -0.087   -0.061          -0.036         0.759        0.714      0.710       0.833     2.256      0.754      26.875
         3000       -0.071     -0.033      -0.023   -0.021           0.012         0.666        0.835      0.831       0.924     1.107      0.681      26.875
logreg   300        -0.571     -0.344      -0.320   -0.167          -0.355         0.800        0.430      0.610       0.320     4.578      0.815      27.097
         1000       -0.256     -0.134      -0.106   -0.063          -0.051         0.793        0.701      0.741       0.812     2.925      0.782      27.097
         3000       -0.082     -0.036      -0.024   -0.022           0.019         0.680        0.859      0.828       0.927     1.390      0.691      27.097

== Planted types, by flip rate, number of planted types and budget ==
                       rel_s_obs  rel_chao1  rel_ichao1  rel_ace  rel_jackknife1  cov_chao1_ln  cov_chao1_b  cov_ace_b  cov_jack_b  new_real  new_ratio  true_types
flip k_planted budget                                                                                                                                              
0.6  15        300        -0.593     -0.413      -0.398   -0.199          -0.387         0.778        0.362      0.600       0.331     2.402      0.598      12.889
               1000       -0.281     -0.164      -0.136   -0.066          -0.074         0.751        0.669      0.767       0.778     1.666      0.663      12.889
               3000       -0.100     -0.050      -0.035   -0.025           0.013         0.600        0.812      0.855       0.912     0.861      0.580      12.889
     30        300        -0.555     -0.273      -0.243   -0.054          -0.312         0.851        0.549      0.737       0.383     4.183      0.862      23.500
               1000       -0.252     -0.120      -0.091   -0.067          -0.045         0.825        0.737      0.763       0.822     2.790      0.821      23.500
               3000       -0.087     -0.038      -0.026   -0.034           0.012         0.722        0.855      0.821       0.928     1.406      0.651      23.500
     60        300        -0.493     -0.264      -0.228   -0.165          -0.248         0.819        0.531      0.655       0.401     7.891      0.845      44.278
               1000       -0.204     -0.107      -0.086   -0.075          -0.031         0.804        0.691      0.656       0.832     4.042      0.827      44.278
               3000       -0.068     -0.025      -0.015   -0.026           0.013         0.794        0.856      0.777       0.924     2.026      0.769      44.278
0.9  15        300        -0.503     -0.343      -0.324   -0.159          -0.297         0.757        0.444      0.626       0.456     2.145      0.618      12.889
               1000       -0.235     -0.143      -0.120   -0.079          -0.063         0.680        0.649      0.749       0.787     1.383      0.633      12.889
               3000       -0.087     -0.048      -0.038   -0.022           0.005         0.583        0.785      0.830       0.891     0.725      0.537      12.889
     30        300        -0.444     -0.233      -0.201   -0.096          -0.214         0.829        0.590      0.691       0.553     3.620      0.850      23.694
               1000       -0.194     -0.104      -0.083   -0.073          -0.038         0.774        0.720      0.713       0.810     2.146      0.751      23.694
               3000       -0.068     -0.028      -0.018   -0.024           0.007         0.659        0.815      0.807       0.905     1.070      0.673      23.694
     60        300        -0.330     -0.189      -0.165   -0.138          -0.121         0.809        0.574      0.605       0.651     5.461      0.840      44.972
               1000       -0.130     -0.076      -0.063   -0.062          -0.021         0.779        0.717      0.604       0.844     2.448      0.746      44.972
               3000       -0.042     -0.017      -0.011   -0.013           0.009         0.648        0.845      0.831       0.923     1.220      0.709      44.972

== Planted types, by dataset and budget ==
                  rel_s_obs  rel_chao1  rel_ichao1  rel_ace  rel_jackknife1  cov_chao1_ln  cov_chao1_b  cov_ace_b  cov_jack_b  new_real  new_ratio  true_types
dataset   budget                                                                                                                                              
adult     300        -0.457     -0.330      -0.316   -0.133          -0.244         0.637        0.430      0.635       0.565     1.504      0.510       8.000
          1000       -0.166     -0.101      -0.082    0.001           0.001         0.524        0.730      0.817       0.846     0.743      0.573       8.000
          3000       -0.032     -0.018      -0.013    0.006           0.030         0.707        0.893      0.926       0.935     0.239      0.488       8.000
bank      300        -0.435     -0.205      -0.175   -0.079          -0.207         0.885        0.607      0.714       0.547     4.822      0.867      33.074
          1000       -0.186     -0.083      -0.060   -0.051          -0.024         0.823        0.757      0.740       0.870     2.578      0.809      33.074
          3000       -0.065     -0.038      -0.030   -0.032           0.002         0.554        0.802      0.778       0.902     1.123      0.570      33.074
covertype 300        -0.436     -0.210      -0.175   -0.089          -0.187         0.898        0.631      0.730       0.590     5.536      0.864      34.944
          1000       -0.173     -0.097      -0.078   -0.081          -0.024         0.848        0.740      0.681       0.864     2.419      0.750      34.944
          3000       -0.058     -0.028      -0.020   -0.024           0.009         0.583        0.839      0.811       0.927     0.904      0.674      34.944
magic     300        -0.618     -0.398      -0.373   -0.240          -0.415         0.809        0.367      0.530       0.148     5.273      0.771      32.130
          1000       -0.340     -0.196      -0.167   -0.150          -0.135         0.880        0.562      0.597       0.669     3.910      0.777      32.130
          3000       -0.145     -0.053      -0.033   -0.046          -0.001         0.828        0.778      0.765       0.891     2.607      0.742      32.130
```

## First reading

The pattern of v0 holds across all four datasets, both flip rates and all three numbers of planted types. Counting
the types found in 300 rows misses about half of the planted types (49%); ACE reduces that to 14% and is the least
biased estimator at 300 rows, and the first-order jackknife is within about 5% from 1,000 rows on. All estimators
still underestimate at small budgets. Bootstrap intervals reach their target only at large budgets (jackknife:
0.91 coverage at 3,000 rows) and cover poorly at 300 rows (0.46-0.65), mainly because the estimates are biased
low rather than because the intervals are too narrow. A second test set finds 15-46% more new types than
predicted. MAGIC is the hardest dataset. Coarse error clusters (k-means with 10 clusters) remain unreliable.
These are observations from an unregistered pilot, not findings.
