"""Applies the pre-registered decision rules in PREREG.md to the main-study CSV. Prints each rule's verdict.

A "cell" is one combination of dataset, model, type definition and test budget, pooled over seeds, flip rates,
numbers of planted types and repeated draws. Type definitions are grouped as PLANTED (planted) and NATURAL
(natural_grid25, natural_grid100, natural_kmeans25).
"""
import sys

import pandas as pd

GROUPS = {"planted": ["planted"], "natural": ["natural_grid25", "natural_grid100", "natural_kmeans25"]}


def main(path):
    df = pd.read_csv(path)
    df = df[df.types.isin(sum(GROUPS.values(), [])) & (df.true_types > 0)].copy()
    df["err"] = df.unseen_est - df.unseen_true
    df["abs_err"] = df.err.abs()
    df["abs_err_tur"] = (df.unseen_turing - df.unseen_true).abs()
    df["cov_unseen"] = (df.unseen_lo <= df.unseen_true + 1e-12) & (df.unseen_true <= df.unseen_hi + 1e-12)
    df["abs_rel_sobs"] = ((df.s_obs - df.true_types) / df.true_types).abs()
    df["abs_rel_ace"] = ((df.ace - df.true_types) / df.true_types).abs()
    df["cov_ace_p"] = (df.ace_plo <= df.true_types) & (df.true_types <= df.ace_phi)
    keys = ["dataset", "model", "types", "budget"]
    cells = df.groupby(keys).agg(bias=("err", "mean"), mae=("abs_err", "mean"), mae_tur=("abs_err_tur", "mean"),
                                 cov_unseen=("cov_unseen", "mean"), sobs=("abs_rel_sobs", "mean"),
                                 ace=("abs_rel_ace", "mean"), cov_ace_p=("cov_ace_p", "mean")).reset_index()
    cells["group"] = cells.types.map({t: g for g, ts in GROUPS.items() for t in ts})
    print(f"{len(df)} rows, {len(cells)} cells\n")
    for g in GROUPS:
        c = cells[cells.group == g]
        h1 = (c.bias.abs() <= 0.04).mean()
        h2 = (c.cov_unseen >= 0.90).mean()
        h3 = (c.ace < c.sobs).mean()
        h5 = (c.mae < c.mae_tur).mean()
        print(f"== {g.upper()} ({len(c)} cells)")
        print(f"H1 unseen share unbiased, |mean signed error| <= 0.04: {h1:.1%} of cells -> {'SUPPORTED' if h1 >= 0.90 else 'NOT SUPPORTED'} (needs >= 90%)")
        print(f"H2 unseen-share interval coverage >= 0.90: {h2:.1%} of cells -> {'SUPPORTED' if h2 >= 0.80 else 'NOT SUPPORTED'} (needs >= 80%)")
        print(f"H3 ACE closer to the truth than the raw count: {h3:.1%} of cells -> {'SUPPORTED' if h3 >= 0.90 else 'NOT SUPPORTED'} (needs >= 90%)")
        print("H4 (reported) bias-calibrated ACE interval coverage by budget:",
              c.groupby("budget").cov_ace_p.mean().round(3).to_dict())
        print(f"H5 (reported) Chao-Jost beats Turing f1/n on mean |error|: {h5:.1%} of cells")
        print("Reported: typical error per test (cell MAE) by budget:", c.groupby("budget").mae.median().round(3).to_dict())
        print("Worst cells for H1:")
        print(c.reindex(c.bias.abs().sort_values(ascending=False).index).head(5)[keys + ["bias", "mae", "cov_unseen"]].round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/main.csv")
