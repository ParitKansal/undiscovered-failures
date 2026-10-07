# Pilot v0 (2026-10-07)

Exploratory pilot, not pre-registered. Code at commit `493a8b2`, run on Google Colab (CPU runtime, 8 cores) with the
default settings: datasets Adult, Bank Marketing and Covertype; models logistic regression, random forest and gradient
boosting; 5 seeds; test budgets of 300, 1,000 and 3,000 rows; 30 repeated draws per budget. The executed notebook is
`pilot_v0_executed.ipynb`.

## Run log

```
45 jobs on 8 CPU cores
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:   43.2s
[Parallel(n_jobs=-1)]: Done  40 out of  45 | elapsed:  4.2min remaining:   31.7s
[Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed:  4.4min finished
wrote /content/drive/MyDrive/undiscovered-failures/pilot.csv 16200 rows
```

## Summary output

```
Mean relative error of each estimator (0 = perfect), Chao1 interval coverage, new-type prediction error
                 rel_s_obs  rel_chao1  rel_ichao1  rel_ace  rel_jackknife1  ci_cover  new_error  coverage  n_errors  true_types
types    budget                                                                                                                
kmeans10 300        -0.254     -0.219      -0.209   -0.164          -0.154     0.306     -0.326     0.982    66.032      10.000
         1000       -0.157     -0.137      -0.134   -0.090          -0.092     0.401     -0.295     0.997   220.072      10.000
         3000       -0.091     -0.065      -0.062   -0.011          -0.028     0.513     -0.230     0.999   663.290      10.000
kmeans25 300        -0.341     -0.227      -0.203   -0.208          -0.175     0.812     -0.531     0.928    66.524      25.000
         1000       -0.191     -0.116      -0.103   -0.107          -0.091     0.753     -0.505     0.987   221.027      25.000
         3000       -0.094     -0.042      -0.031   -0.005          -0.017     0.727     -0.395     0.997   661.887      25.000
kmeans50 300        -0.446     -0.259      -0.224   -0.230          -0.226     0.797     -0.757     0.816    66.307      50.000
         1000       -0.234     -0.132      -0.112   -0.155          -0.103     0.863     -0.839     0.965   220.107      50.000
         3000       -0.114     -0.046      -0.032   -0.038          -0.026     0.836     -0.573     0.992   662.407      50.000
planted  300        -0.404     -0.213      -0.182   -0.089          -0.173     0.816     -0.518     0.756    31.633      22.778
         1000       -0.165     -0.096      -0.079   -0.067          -0.027     0.736     -0.495     0.952   105.444      22.778
         3000       -0.061     -0.038      -0.031   -0.027          -0.002     0.581     -0.346     0.993   316.559      22.778

Same, split by model (planted types only):
                 rel_s_obs  rel_chao1  rel_ichao1  rel_ace  rel_jackknife1  ci_cover
model    budget                                                                     
boosting 300        -0.280     -0.154      -0.127   -0.099          -0.072     0.787
         1000       -0.109     -0.074      -0.064   -0.064          -0.023     0.727
         3000       -0.043     -0.030      -0.026   -0.020          -0.005     0.571
forest   300        -0.381     -0.208      -0.174   -0.110          -0.143     0.831
         1000       -0.149     -0.084      -0.068   -0.066          -0.020     0.693
         3000       -0.054     -0.036      -0.030   -0.021          -0.001     0.513
logreg   300        -0.550     -0.278      -0.246   -0.059          -0.304     0.829
         1000       -0.236     -0.130      -0.105   -0.070          -0.038     0.787
         3000       -0.085     -0.048      -0.038   -0.040           0.000     0.660
```

## First reading

Counting the failure types found in a test of 300 rows misses about 40% of the planted types; ACE reduces that to
about 9%, and the first-order jackknife is nearly unbiased from 1,000 rows on. Every estimator still underestimates,
as expected when types differ in how common they are. The Chao1 log-normal interval covers the truth in only 58-83% of
runs for planted types (31-51% for 10 k-means clusters), and its coverage falls as the budget grows. Predicted numbers
of new types in a second draw are too low throughout. These are first observations from one design, not findings.
