"""End-to-end check on a small synthetic dataset (no download): one job, the CSV, and the summary.
Run: python -m tests.test_end_to_end"""
import os
import tempfile

import pandas as pd

import experiment
import summarize

with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "e2e.csv")
    experiment.main(["--datasets", "synthetic", "--models", "logreg,boosting", "--seeds", "1",
                     "--repeats", "3", "--out", out, "--jobs", "2"])
    df = pd.read_csv(out)
    assert set(df.types) >= {"planted", "kmeans10"}, df.types.unique()
    assert (df.s_obs <= df.true_types).all(), "cannot observe more types than exist"
    assert (df.chao1 >= df.s_obs - 1e-9).all() and (df.ichao1 >= df.chao1 - 1e-9).all(), "estimator ordering"
    assert (df.chao1_lo <= df.chao1 + 1e-9).all() and (df.chao1 <= df.chao1_hi + 1e-9).all(), "interval"
    assert df.new_real.ge(0).all() and df.new_pred.ge(0).all(), "new-type counts"
    summarize.summarize(out)
print("END-TO-END TEST PASSED")
