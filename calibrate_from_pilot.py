"""Reads the pilot v2 CSV and prints the per-cell error of the unseen-share estimate, to set the thresholds in
PREREG.md before the main study. A cell = dataset x model x type definition x budget, pooled over the rest."""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
df = df[df.true_types > 0].copy()
df["abs_err"] = (df.unseen_est - df.unseen_true).abs()
df["rel_abs_err"] = df.abs_err / df.unseen_true.where(df.unseen_true > 0)
cells = df.groupby(["dataset", "model", "types", "budget"]).agg(
    mae=("abs_err", "mean"), p90=("abs_err", lambda x: x.quantile(0.9)), truth=("unseen_true", "mean"),
    rel_mae=("rel_abs_err", "mean")).reset_index()
pd.set_option("display.width", 200)
print("Per-budget distribution of cell MAE (planted and k-means types):")
print(cells.groupby(["types", "budget"]).mae.describe(percentiles=[.5, .9]).round(3).to_string())
print("\nShare of cells with MAE <= 0.03 / 0.05 / 0.08, by budget:")
for b, c in cells.groupby("budget"):
    print(b, {t: round((c.mae <= t).mean(), 3) for t in (0.03, 0.05, 0.08)}, "| mean truth", round(c.truth.mean(), 3),
          "| median relative MAE", round(c.rel_mae.median(), 3))
