"""Summary tables for a pilot CSV.

  rel_<estimator>  (estimate - truth) / truth, averaged: bias (0 is perfect; s_obs is the naive count of types found)
  cov_<interval>   share of runs whose 95% interval contains the truth (should be about 0.95)
                   chao1_ln = Chao1 log-normal interval; chao1_b, ace_b, jack_b = bootstrap intervals
  new_real         new types actually found by a second test set of the same size
  new_ratio        predicted / real new types, summed over runs (1 is perfect)
"""
import sys

import pandas as pd

EST = ["s_obs", "chao1", "ichao1", "ace", "jackknife1"]
CIS = {"chao1_ln": ("chao1_lo", "chao1_hi"), "chao1_b": ("chao1_blo", "chao1_bhi"),
       "ace_b": ("ace_blo", "ace_bhi"), "jack_b": ("jackknife1_blo", "jackknife1_bhi")}


def add_columns(df):
    df = df[df.true_types > 0].copy()
    for e in EST:
        df[f"rel_{e}"] = (df[e] - df.true_types) / df.true_types
    for name, (lo, hi) in CIS.items():
        if lo in df:
            df[f"cov_{name}"] = (df[lo] <= df.true_types) & (df.true_types <= df[hi])
    return df


def table(df, by):
    g = df.groupby(by)
    cols = [f"rel_{e}" for e in EST] + [c for c in df.columns if c.startswith("cov_")]
    t = g[cols].mean()
    t["new_real"] = g.new_real.mean()
    t["new_ratio"] = g.new_pred.sum() / g.new_real.sum().where(g.new_real.sum() > 0)
    t["true_types"] = g.true_types.mean()
    return t.round(3)


def summarize(path):
    df = add_columns(pd.read_csv(path))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("== By type definition and test budget ==")
    print(table(df, ["types", "budget"]).to_string())
    p = df[df.types == "planted"]
    print("\n== Planted types, by model and budget ==")
    print(table(p, ["model", "budget"]).to_string())
    if "flip" in p and (p.flip.nunique() > 1 or p.k_planted.nunique() > 1):
        print("\n== Planted types, by flip rate, number of planted types and budget ==")
        print(table(p, ["flip", "k_planted", "budget"]).to_string())
    if p.dataset.nunique() > 1:
        print("\n== Planted types, by dataset and budget ==")
        print(table(p, ["dataset", "budget"]).to_string())


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "results/pilot.csv")
