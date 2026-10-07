"""Summary tables for the pilot CSV.

For each type definition and test budget:
  rel_error   (estimate - truth) / truth, averaged          -> bias (0 is perfect; S_obs shows the naive count)
  ci_cover    share of runs where the Chao1 95% interval contains the truth (should be about 0.95)
  new_error   predicted minus real number of new types in a second test set of the same size
"""
import sys

import pandas as pd

EST = ["s_obs", "chao1", "ichao1", "ace", "jackknife1"]


def summarize(path):
    df = pd.read_csv(path)
    df = df[df.true_types > 0]
    for e in EST:
        df[f"rel_{e}"] = (df[e] - df.true_types) / df.true_types
    df["ci_cover"] = (df.chao1_lo <= df.true_types) & (df.true_types <= df.chao1_hi)
    df["new_error"] = df.new_pred - df.new_real
    g = df.groupby(["types", "budget"])
    table = g[[f"rel_{e}" for e in EST] + ["ci_cover", "new_error", "coverage", "n_errors", "true_types"]].mean()
    pd.set_option("display.width", 200)
    print("Mean relative error of each estimator (0 = perfect), Chao1 interval coverage, new-type prediction error")
    print(table.round(3).to_string())
    print("\nSame, split by model (planted types only):")
    p = df[df.types == "planted"].groupby(["model", "budget"])
    print(p[[f"rel_{e}" for e in EST] + ["ci_cover"]].mean().round(3).to_string())
    return table


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "results/pilot.csv")
