# Pre-registration: main study

**Written:** 2026-10-08, before any main-study run. The commit that adds this file is the pre-registration. Any later change is a dated note at the end.

## 1. Question

When a classical machine-learning model is tested, can species-richness statistics from ecology tell how many of its failure types are still undiscovered, or at least what share of its failures those types cause?

## 2. What the pilots showed

The three pilots (v0, v1, v2, recorded in `runs/`) were exploratory and are not evidence for this study.

- Counting the failure types found in 300 test rows missed about half of the planted types. ACE and the first-order jackknife reduced that error a lot.
- All count estimators were biased low at small budgets. Their 95% intervals covered the truth in only about 45-90% of runs.
- The estimated unseen share of errors (1 - sample coverage) was close to the truth on average, within about 1.4 points at 300 rows.
- Its error on a single test was large. The median absolute error was 0.16 for planted types at 300 rows, against a true share of about 0.31.

These facts set the rules below. The pilot data are not reused: the main study uses new seeds.

## 3. Design

Everything is fixed in `experiment.py` at the commit of this file.

- **Datasets:** Adult, Bank Marketing, Covertype (a seeded sample of 120,000 rows), MAGIC and Electricity.
  - Electricity was not used in any pilot.
  - Each dataset is split 50/50 into a training set and a test pool.
- **Models:** logistic regression, random forest (150 trees) and histogram gradient boosting, all from scikit-learn.
- **Seeds:** 100 to 104. None was used in a pilot.
- **Planted failures:**
  - 15, 30 or 60 disjoint subgroups, defined by rules on 1-3 features, each covering 0.01-2% of rows;
  - inside them, 60% or 90% of training labels are flipped.
- **Natural failures:** the same model trained on the clean labels. Its errors get types that are fixed before any error is seen:
  - `natural_grid25` and `natural_grid100`: cells of a 5 × 5 or 10 × 10 quantile grid over the first two principal components of the training data;
  - `natural_kmeans25`: k-means with 25 clusters on the natural errors.
- **Testing:** test sets of 300, 1,000 and 3,000 rows, each drawn 30 times from the test pool, with a second draw of the same size for the new-type check.
  - The true number of types is the number of types with at least one error in the whole pool.
  - The true unseen share is the share of all pool errors whose type the test set did not see.
- **Estimators:**
  - type counts: Chao1, iChao1, ACE and the first-order jackknife;
  - intervals for the counts: log-normal, bootstrap, and the bias-calibrated bootstrap (`pivot_ci`);
  - unseen share: 1 - sample coverage (Chao & Jost 2012) with a calibrated interval (`unseen_ci`), plus Turing's f1/n as a baseline;
  - new-type prediction: Shen, Chao & Lin (2003).
  - Bootstrap draws: 200.

## 4. Decision rules

A **cell** is one dataset × model × type definition × budget, pooled over seeds, flip rates, numbers of planted types and draws. The rules are applied separately to PLANTED cells (`planted`) and to NATURAL cells (`natural_grid25`, `natural_grid100`, `natural_kmeans25`) by `analyze_main.py` at the commit of this file.

| Rule | Claim | Supported if |
|---|---|---|
| **H1** (primary) | The unseen-share estimate is unbiased | the mean signed error is within ±0.04 in at least 90% of cells |
| **H2** (primary) | Its calibrated 95% interval is honest | coverage is at least 0.90 in at least 80% of cells |
| **H3** | ACE is closer to the true number of types than the raw count | true in at least 90% of cells (mean absolute relative error) |
| H4 (reported, no rule) | Coverage of the bias-calibrated interval for the number of types, by budget | expected to be below 0.95 at 300 rows |
| H5 (reported, no rule) | Chao-Jost unseen share versus Turing's f1/n | share of cells where Chao-Jost has the lower mean absolute error |

Also reported, with no rule attached:

- the typical error of the unseen share on one test (median cell absolute error), by budget;
- all estimator biases;
- the new-type prediction ratio;
- the k-means types of the planted model (`kmeans10`, `kmeans25`, `kmeans50`).

**Reading of outcomes:**

| H1 | H2 | Reading |
|---|---|---|
| supported | supported | the unseen share can be estimated without bias and with an honest interval, from test results alone, while the number of types cannot (H4) |
| supported | not supported | the estimate is unbiased, but its uncertainty on one test cannot be stated reliably |
| not supported | either | the unseen share is not reliably estimable in this setting |

Each outcome is reported as it comes out, for planted and natural failures separately.

## 5. Procedure and deviations

- **Run:** cells 9-10 of `pilot_colab.ipynb` on a Colab CPU runtime, all cores.
- **Re-runs:** a crashed or incomplete run may be re-run with the same seeds. The row count must be complete: 450 jobs × 7 type definitions × 3 budgets × 30 draws = 283,500 rows.
- **Natural failures repeat:** the natural model does not depend on the flip rate or the number of planted types. So for each dataset, model and seed it is trained 6 times with identical inputs, and its rows come from 6 independent sets of test draws. That is accepted and pooled.
- **Records:** the executed notebook and its output go into `runs/`.
- **Deviations:** any change after this commit is recorded below with its date and reason.

## Change log

(none yet)
