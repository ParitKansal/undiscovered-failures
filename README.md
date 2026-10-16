<div align="center">

# How Many Failure Types Are Still Undiscovered?
### Species-Richness Estimators for Testing Machine-Learning Models

**Parit Kansal**
Independent Researcher · [paritkansal.in](https://paritkansal.in) · [OpenReview](https://openreview.net/profile?id=~Parit_Kansal1) · [GitHub](https://github.com/ParitKansal)

October 2026

[![Status: pre-registered](https://img.shields.io/badge/status-pre--registered%20study-8a5cf6)](PREREG.md)
[![Unseen share: SUPPORTED](https://img.shields.io/badge/unseen%20share-SUPPORTED-1baf7a)](RESULTS.md)
[![Number of types: lower bound only](https://img.shields.io/badge/number%20of%20types-lower%20bound%20only-eda100)](RESULTS.md)
[![Code: MIT](https://img.shields.io/badge/code-MIT-2a78d6)](LICENSE)
[![Runs on Colab CPU](https://img.shields.io/badge/runs%20on-Google%20Colab%20CPU-52514e)](pilot_colab.ipynb)

</div>

---

## Abstract

When we test a machine-learning model, we find some of the ways it fails, but the test log alone does not say how many kinds of failure we have *not* found. Ecologists face the same problem when they count species, and they answer it with species-richness estimators such as Chao1 and ACE, which use how many species were seen exactly once or twice to estimate how many were never seen. We test whether these estimators work for model failures. Because nobody knows the true failure types of a real model, we use two settings where the truth is known: *planted* failures, where we flip most training labels inside 15, 30 or 60 rare subgroups of a tabular dataset so that the model is wrong exactly there, and *natural* failures of a model trained on clean labels, whose errors are grouped into types fixed before any error is seen. We then simulate tests of 300, 1,000 and 3,000 rows and compare each estimate with the truth in the full test pool. In a pre-registered main study (5 datasets, 3 model families, 5 fresh seeds, 450 jobs), the **share of errors that belong to failure types not yet seen** was estimated without bias in 95.6% of planted and 100% of natural cells, and its calibrated 95% interval was honest for natural failures (95.6% of cells) but too narrow for planted ones (71.1%, mainly on small tests). The **number of failure types** could not be estimated reliably: ACE beat the raw count of types found in 88.9% and 85.2% of cells, just short of the pre-registered 90%, and no interval for the number of types reached 95% coverage. The practical reading is that a test can tell what share of a model's errors comes from failure kinds nobody has seen yet, while the number of such kinds should be reported only as a lower bound.

## Design

The failure types of a real model are unknown, so an estimate cannot be checked against them. We avoid that in two ways. In the planted setting we choose disjoint subgroups, defined by simple rules on one to three features and each covering 0.01% to 2% of rows, and flip 60% or 90% of their training labels; the model learns to be wrong exactly there, so we know which failure types exist. In the natural setting we train the same model on clean labels and give its errors types that are fixed in advance: cells of a 5 × 5 or 10 × 10 quantile grid over the first two principal components of the training data, or 25 k-means clusters of the errors. In both settings the true number of types is the number of types with at least one error in the whole test pool, and the true unseen share is the share of all pool errors whose type a test did not see.

Each dataset (Adult, Bank Marketing, Covertype, MAGIC Gamma Telescope and Electricity) is split in half into a training set and a test pool. The models are logistic regression, a random forest and histogram gradient boosting from scikit-learn. Tests of 300, 1,000 and 3,000 rows are drawn 30 times each from the pool, with a second draw of the same size to check how many new types a further test would reveal. The estimators are Chao1, iChao1, ACE and the first-order jackknife for the number of types, with log-normal, bootstrap and bias-calibrated bootstrap intervals; the unseen share as one minus the sample coverage of Chao and Jost (2012), with a calibrated interval and Turing's f1/n as a baseline; and the new-type prediction of Shen, Chao and Lin (2003).

Three exploratory pilots, recorded in `runs/`, shaped the design. The main study was then pre-registered in [`PREREG.md`](PREREG.md) (commit `abafcac`, 8 October 2026) with fresh seeds 100 to 104 and fixed decision rules, and nothing was changed after that.

## Results

The full verdicts, the worst cells and the summary tables are in [`RESULTS.md`](RESULTS.md); the executed notebook is [`runs/main_executed.ipynb`](runs/main_executed.ipynb). The main study ran 450 jobs and produced 283,500 rows in 11.3 minutes on 24 CPU cores.

| Rule | Planted failures (45 cells) | Natural failures (135 cells) |
|---|---|---|
| **H1** the unseen-share estimate is unbiased (mean error within ±0.04 in at least 90% of cells) | **SUPPORTED** (95.6%) | **SUPPORTED** (100%) |
| **H2** its calibrated 95% interval is honest (coverage at least 0.90 in at least 80% of cells) | **NOT SUPPORTED** (71.1%) | **SUPPORTED** (95.6%) |
| **H3** ACE is closer to the true number of types than the raw count (at least 90% of cells) | **NOT SUPPORTED** (88.9%) | **NOT SUPPORTED** (85.2%) |
| H4 (reported) coverage of the bias-calibrated ACE interval at 300 / 1,000 / 3,000 rows | 0.65 / 0.77 / 0.84 | 0.73 / 0.74 / 0.81 |
| H5 (reported) Chao–Jost beats Turing's f1/n on mean absolute error | 71.1% of cells | 84.4% of cells |
| Typical error of the unseen share on one test (median absolute error) at 300 / 1,000 / 3,000 rows | 0.146 / 0.050 / 0.011 | 0.086 / 0.023 / 0.005 |

For natural failures, a single test can state without bias, and with an honest interval, what share of the model's errors comes from types it has not yet seen. For planted failures the estimate is still unbiased on average, but its interval understates the uncertainty of one small test; the worst cells are Adult at 300 rows, where the planted types are few and a test sees only a few dozen errors. Counting the types found misses about half of the planted types at 300 rows, and the estimators reduce that undercount on average but not in every cell, so the number of failure types is best reported as a lower bound.

This study has clear limits. It uses tabular data and classical models only, and the natural failure types are grid cells and clusters rather than types a person would name, so the results do not show how the estimators behave on, for example, the failure categories found by red-teaming a language model.

## What is in the repository

| File | What it does |
|---|---|
| `estimators.py` | Chao1 with its log-normal 95% interval, iChao1, ACE, first-order jackknife, sample coverage, the prediction of new types, bootstrap 95% intervals for any estimator, and a bias-calibrated bootstrap |
| `experiment.py` | One job per dataset, model and seed: plant failures, train, find errors, simulate testing, write one CSV row per draw. Jobs run in parallel on all CPU cores. |
| `summarize.py` | Tables of each estimator's bias, interval coverage and new-type prediction error |
| `analyze_main.py` | The pre-registered decisions H1–H5 for the main study |
| `calibrate_from_pilot.py` | The pilot numbers used to set the pre-registered thresholds |
| `PREREG.md`, `RESULTS.md` | The pre-registration of the main study and its results |
| `runs/` | The executed notebooks of the three pilots and the main study, with summaries |
| `tests/` | Checks of the estimators against hand-computed values and a simulation, and an end-to-end run on a small synthetic dataset |
| `pilot_colab.ipynb` | Runs the pilots and the main study on Google Colab, CPU only |

## Running it

On Colab, open `pilot_colab.ipynb`, choose a CPU runtime and run the cells in order; cells 9 and 10 run the main study and its pre-registered analysis.

Locally, with Python 3.10 or newer:

```
pip install -r requirements.txt
python -m tests.test_estimators
python -m tests.test_end_to_end
python experiment.py --datasets adult,bank,covertype,magic,electricity --models logreg,forest,boosting --seeds 5 --seed_offset 100 --repeats 30 --flips 0.6,0.9 --k_planted 15,30,60 --boot 200 --out results/main.csv
python analyze_main.py results/main.csv
python summarize.py results/main.csv
```

The datasets (Adult, Bank Marketing, Covertype, MAGIC Gamma Telescope and Electricity) are downloaded automatically from OpenML and scikit-learn.

## Related work

The idea of treating testing as species discovery comes from software testing, where Böhme's STADS framework (TOSEM 2018) applies these estimators to fuzzing. The estimators themselves come from Chao (1984, 1987), Chao and Lee (1992), Burnham and Overton (1978), Chiu, Wang, Walther and Chao (2014), Shen, Chao and Lin (2003), and Chao and Jost (2012).

## Citation

If you use this work, please cite:

```bibtex
@misc{kansal2026undiscoveredfailures,
  author       = {Kansal, Parit},
  title        = {How Many Failure Types Are Still Undiscovered? Species-Richness Estimators for Testing Machine-Learning Models},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/ParitKansal/undiscovered-failures}},
  note         = {Pre-registered study. GitHub repository}
}
```

The **Cite this repository** button on the right of this page provides the same reference in other formats.

## Acknowledgements

The datasets come from OpenML and scikit-learn, each under its own licence. Literature searches and code were developed with the assistance of an AI system (Claude); all experiments were run and all results verified by the author.

## Licence

MIT (see `LICENSE`).
