# How many failure types are still undiscovered?

When we test a machine-learning model, we find some of the ways it fails. The question nobody can answer from the test log alone is how many kinds of failure we have *not* found yet. Ecologists face the same problem when they count species: a survey finds some species, and the rare ones are easy to miss. They solved it with species-richness estimators such as Chao1, which use how many species were seen exactly once or twice to estimate how many were never seen.

This repository tests whether those estimators work for model failures. The difficulty with real models is that nobody knows the true number of failure types, so an estimate cannot be checked. We avoid that by planting failures. In a standard tabular dataset we choose 30 subgroups, from common ones to very rare ones, and flip most of their training labels. The model learns to be wrong exactly there, so we know which failure types exist. We then simulate testing with test sets of different sizes, apply the estimators to the errors found, and compare them with the truth. We also check a practical prediction: how many new failure types a second test set of the same size will reveal.

The same checks run on a second definition of a failure type, where the model's real errors are grouped by k-means at 10, 25 and 50 clusters. That version shows how much the answer depends on how finely failures are grouped.

**Status:** exploratory pilot, not pre-registered. The first run (pilot v0) is recorded in `runs/`. Version 1 adds bootstrap intervals, a fourth dataset, two label-flip rates and three numbers of planted types. Version 2 adds a bias-calibrated bootstrap and a second target: the share of all errors that belong to failure types not yet seen, which can be estimated even when the total number of types cannot.

## What is in the repository

| File | What it does |
|---|---|
| `estimators.py` | Chao1 with its log-normal 95% interval, iChao1, ACE, first-order jackknife, sample coverage, the prediction of new types, bootstrap 95% intervals for any estimator, and a bias-calibrated bootstrap |
| `experiment.py` | One job per dataset, model and seed: plant failures, train, find errors, simulate testing, write one CSV row per draw. Jobs run in parallel on all CPU cores. |
| `summarize.py` | Tables of each estimator's bias, interval coverage and new-type prediction error |
| `tests/` | Checks of the estimators against hand-computed values and a simulation, and an end-to-end run on a small synthetic dataset |
| `pilot_colab.ipynb` | Runs everything on Google Colab, CPU only |

## Running it

On Colab, open `pilot_colab.ipynb`, choose a CPU runtime and run the cells in order.

Locally, with Python 3.10 or newer:

```
pip install -r requirements.txt
python -m tests.test_estimators
python -m tests.test_end_to_end
python experiment.py --datasets adult,bank,covertype,magic --models logreg,forest,boosting --seeds 3 --flips 0.6,0.9 --k_planted 15,30,60
python summarize.py results/pilot.csv
```

The datasets (Adult, Bank Marketing, MAGIC Gamma Telescope and Covertype) are downloaded automatically from OpenML and scikit-learn.

## Related work

The idea of treating testing as species discovery comes from software testing, where Böhme's STADS framework (TOSEM 2018) applies these estimators to fuzzing. The estimators themselves come from Chao (1984, 1987), Chao and Lee (1992), Burnham and Overton (1978), Chiu, Wang, Walther and Chao (2014), Shen, Chao and Lin (2003), and Chao and Jost (2012).

## Citation

Use the **Cite this repository** button on this page, or the `CITATION.cff` file.

## Licence

MIT (see `LICENSE`).
