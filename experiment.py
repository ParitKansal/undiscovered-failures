"""Pilot: can ecology's richness estimators tell how many failure types of a model are still undiscovered?

One job = (dataset, model, seed):
  1. Load a standard tabular dataset and split it into train / test pool.
  2. PLANT failures: pick K subgroups (simple rules, from common to rare) and flip most of their TRAINING
     labels. The model learns to be wrong there, so we know exactly which failure types exist.
  3. Train the model. Find all its errors in the big test pool; each error gets a type:
       planted  -> the subgroup it falls in (true number of types = subgroups with >= 1 error in the pool)
       clusters -> k-means cluster of the error (k = 10, 25, 50), fitted on all pool errors
  4. Simulate testing: draw test sets of several sizes, count errors per type, apply the estimators,
     and compare with the truth. Also predict how many NEW types a second test set of the same size
     will reveal, and check it against a real second draw.
Everything is written to one CSV row per (job, type definition, budget, repeat).
"""
import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import estimators as E

# ---------------------------------------------------------------- data


def load(name, data_home=None):
    """Returns numeric features X (float32) and binary labels y (0/1)."""
    if name == "synthetic":  # small offline dataset for tests
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=20000, n_features=12, n_informative=6, random_state=0)
    elif name == "covertype":
        d = fetch_covtype(data_home=data_home)
        X, y = d.data, (d.target == 2).astype(int)  # most common class vs rest
    else:
        ids = {"adult": 1590, "bank": 1461, "magic": 1120, "electricity": 151}  # magic 19k rows, electricity 45k
        d = fetch_openml(data_id=ids[name], as_frame=True, data_home=data_home, parser="auto")
        X = pd.get_dummies(d.data, dummy_na=True).astype("float32").fillna(0).values
        y = (d.target == d.target.value_counts().index[1]).astype(int).values  # minority class = 1
    return np.asarray(X, dtype="float32"), np.asarray(y)


def plant(X, k, rng, min_share=0.0001, max_share=0.02):
    """K disjoint subgroups defined by rules on 1-3 features (quantile bins). More conditions = rarer group.
    Shares range from very rare (0.01% of rows) to common (2%), so some failure types are hard to find.
    Returns group id per row (-1 = no planted group)."""
    group = np.full(len(X), -1)
    useful = [j for j in range(X.shape[1]) if len(np.unique(X[:2000, j])) > 1]
    made, tries = 0, 0
    while made < k and tries < 5000:
        tries += 1
        n_cond = rng.choice([1, 2, 2, 3, 3])
        mask = np.ones(len(X), bool)
        for j in rng.choice(useful, size=n_cond, replace=False):
            lo, hi = np.quantile(X[:, j], sorted(rng.uniform(0, 1, 2)))
            mask &= (X[:, j] >= lo) & (X[:, j] <= hi)
        mask &= group == -1  # disjoint: earlier groups keep their rows
        share = mask.mean()
        if min_share <= share <= max_share:
            group[mask] = made
            made += 1
    return group


MODELS = {
    "logreg": lambda s: LogisticRegression(max_iter=300),
    "forest": lambda s: RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=s),
    "boosting": lambda s: HistGradientBoostingClassifier(random_state=s),
}

# ---------------------------------------------------------------- one job


def run_job(dataset, model, seed, k_planted=30, budgets=(300, 1000, 3000), repeats=30,
            flip=0.9, cluster_ks=(10, 25, 50), data_home=None, boot=200, natural=True):
    rng = np.random.default_rng(seed)
    X, y = load(dataset, data_home)
    if len(X) > 120000:  # keep jobs fast
        idx = rng.choice(len(X), 120000, replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    g_all = plant(np.vstack([Xtr, Xte]), k_planted, rng)
    g_tr, g_te = g_all[:len(Xtr)], g_all[len(Xtr):]
    ytr_noisy = ytr.copy()
    hit = (g_tr >= 0) & (rng.uniform(size=len(ytr)) < flip)
    ytr_noisy[hit] = 1 - ytr_noisy[hit]

    clf = MODELS[model](seed).fit(Xtr, ytr_noisy)
    wrong = clf.predict(Xte) != yte  # test labels are clean

    types = {"planted": np.where(wrong & (g_te >= 0), g_te, -1)}
    err_idx = np.where(wrong)[0]
    for kc in cluster_ks:
        lab = np.full(len(Xte), -1)
        if len(err_idx) >= kc:
            lab[err_idx] = KMeans(n_clusters=kc, n_init=3, random_state=seed).fit_predict(Xte[err_idx])
        types[f"kmeans{kc}"] = lab

    # NATURAL failures: the same model trained on the clean labels; its errors are real, nothing is planted.
    # Types are fixed before looking at any error: cells of a grid over the first two principal components
    # of the training data (quantile bins), plus k-means on the natural errors.
    if natural:
        nat = MODELS[model](seed).fit(Xtr, ytr)
        wrong_n = nat.predict(Xte) != yte
        pca = PCA(n_components=2, random_state=seed).fit(Xtr)
        ztr, zte = pca.transform(Xtr), pca.transform(Xte)
        for g in (5, 10):
            edges = [np.quantile(ztr[:, j], np.linspace(0, 1, g + 1)[1:-1]) for j in range(2)]
            cell = np.digitize(zte[:, 0], edges[0]) * g + np.digitize(zte[:, 1], edges[1])
            types[f"natural_grid{g * g}"] = np.where(wrong_n, cell, -1)
        lab = np.full(len(Xte), -1)
        e_n = np.where(wrong_n)[0]
        if len(e_n) >= 25:
            lab[e_n] = KMeans(n_clusters=25, n_init=3, random_state=seed).fit_predict(Xte[e_n])
        types["natural_kmeans25"] = lab

    rows = []
    for tname, t in types.items():
        true_types = len(set(t[t >= 0]))  # types with at least one error in the whole pool
        for b in budgets:
            if 2 * b > len(Xte):
                continue
            for r in range(repeats):
                draw = rng.choice(len(Xte), 2 * b, replace=False)
                a, nxt = draw[:b], draw[b:]  # first test set, and a second one to check predictions
                ta = t[a][t[a] >= 0]
                counts = np.bincount(ta) if len(ta) else np.array([], int)
                seen = set(ta.tolist())
                tb = t[nxt][t[nxt] >= 0]
                new_real = len(set(tb.tolist()) - seen)
                row = dict(dataset=dataset, model=model, seed=seed, flip=flip, k_planted=k_planted,
                           types=tname, budget=b, repeat=r,
                           true_types=true_types, n_errors=int(len(ta)), s_obs=len(seen),
                           f1=int((counts == 1).sum()), f2=int((counts == 2).sum()),
                           coverage=E.coverage(counts.tolist()),
                           new_pred=E.predict_new(counts.tolist(), len(tb)), new_real=new_real)
                cl = counts.tolist()
                for name, fn in E.ESTIMATORS.items():
                    row[name] = fn(cl)
                row["chao1_lo"], row["chao1_hi"] = E.chao1_ci(cl)
                for name in ("chao1", "ace", "jackknife1"):
                    fn = E.ESTIMATORS[name]
                    row[f"{name}_blo"], row[f"{name}_bhi"] = E.boot_ci(cl, fn, B=boot, seed=r)  # iNEXT method
                    row[f"{name}_bc"], row[f"{name}_plo"], row[f"{name}_phi"] = E.pivot_ci(cl, fn, B=boot, seed=r)
                # unseen mass: share of ALL pool errors (of this type definition) whose type was not seen in this test
                pool_types = t[t >= 0]
                row["unseen_true"] = float(np.isin(pool_types, list(seen), invert=True).mean()) if len(pool_types) else 0.0
                row["unseen_est"] = 1 - row["coverage"]
                row["unseen_turing"] = row["f1"] / len(ta) if len(ta) else 0.0  # Turing's original f1/n
                row["unseen_lo"], row["unseen_hi"] = E.unseen_ci(cl, B=boot, seed=r)
                rows.append(row)
    return rows


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="adult,bank,covertype")
    ap.add_argument("--models", default="logreg,forest,boosting")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0, help="first seed (the main study uses 100)")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--flips", default="0.9", help="share of training labels flipped inside planted groups")
    ap.add_argument("--k_planted", default="30", help="number of planted failure types")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap draws per interval")
    ap.add_argument("--out", default="results/pilot.csv")
    ap.add_argument("--data_home", default=None)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel jobs (-1 = all CPU cores)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for d in a.datasets.split(","):  # download once, before the parallel jobs
        load(d, a.data_home)
    jobs = [(d, m, s, float(f), int(k)) for d in a.datasets.split(",") for m in a.models.split(",")
            for s in range(a.seed_offset, a.seed_offset + a.seeds) for f in a.flips.split(",") for k in a.k_planted.split(",")]
    print(f"{len(jobs)} jobs on {os.cpu_count()} CPU cores")
    out = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(run_job)(d, m, s, k_planted=k, flip=f, repeats=a.repeats, data_home=a.data_home, boot=a.boot)
        for d, m, s, f, k in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), "rows")


if __name__ == "__main__":
    main()
