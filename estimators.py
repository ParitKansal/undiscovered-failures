"""Species-richness estimators from ecology, applied to counts of model errors per error type.

Input everywhere: `counts`, a list with how many errors of each OBSERVED type were found (zeros are ignored).
  n  = total errors found          S_obs = number of types seen
  f1 = types seen exactly once     f2 = types seen exactly twice   (and so on)

References: Chao (1984, 1987); Chao & Lee (1992, ACE); Burnham & Overton (1978, jackknife);
Chiu, Wang, Walther & Chao (2014, iChao1); Shen, Chao & Lin (2003, predicting new types);
Chao & Jost (2012, sample coverage).
"""
import math
from collections import Counter


def freq(counts):
    """f[k] = number of types seen exactly k times."""
    return Counter(c for c in counts if c > 0)


def basics(counts):
    c = [x for x in counts if x > 0]
    f = freq(c)
    return sum(c), len(c), f


def chao1(counts):
    """Bias-corrected Chao1 (Chao 2005): S_obs + (n-1)/n * f1(f1-1) / (2(f2+1))."""
    n, s, f = basics(counts)
    if n == 0:
        return 0.0
    f1, f2 = f[1], f[2]
    return s + (n - 1) / n * f1 * (f1 - 1) / (2 * (f2 + 1))


def chao1_ci(counts, z=1.96):
    """Log-normal 95% interval for Chao1 (Chao 1987). Lower end is never below S_obs.
    Variance: classic formula when f2 > 0, the f2 = 0 version otherwise."""
    n, s, f = basics(counts)
    est = chao1(counts)
    f1, f2 = f[1], f[2]
    if n == 0 or f1 == 0:
        return float(s), float(s)
    k = (n - 1) / n
    if f2 > 0:
        r = f1 / f2
        var = f2 * (k * r ** 2 / 2 + k ** 2 * r ** 3 + k ** 2 * r ** 4 / 4)
    else:
        var = (k * f1 * (f1 - 1) / 2 + k ** 2 * f1 * (2 * f1 - 1) ** 2 / 4
               - k ** 2 * f1 ** 4 / (4 * est) if est > 0 else 0.0)
        var = max(var, 0.0)
    t = est - s
    if t <= 0 or var <= 0:
        return float(s), float(est)
    c = math.exp(z * math.sqrt(math.log(1 + var / t ** 2)))
    return s + t / c, s + t * c


def ichao1(counts):
    """Improved Chao1 (Chiu et al. 2014): adds a term using f3 and f4 (f4 = 0 is replaced by 1)."""
    n, s, f = basics(counts)
    if n < 4:
        return chao1(counts)
    f1, f2, f3, f4 = f[1], f[2], f[3], max(f[4], 1)
    extra = (n - 3) / (4 * n) * f3 / f4 * max(f1 - (n - 3) / (2 * (n - 1)) * f2 * f3 / f4, 0)
    return chao1(counts) + extra


def ace(counts, rare=10):
    """Abundance-based Coverage Estimator (Chao & Lee 1992), rare types = seen <= `rare` times."""
    c = [x for x in counts if x > 0]
    rare_c = [x for x in c if x <= rare]
    s_abund = len(c) - len(rare_c)
    n_rare = sum(rare_c)
    if n_rare == 0:
        return float(len(c))
    f = freq(rare_c)
    f1 = f[1]
    if f1 == n_rare:  # every rare type is a singleton: ACE is undefined, fall back to Chao1
        return chao1(counts)
    cov = 1 - f1 / n_rare
    gamma2 = max(len(rare_c) / cov * sum(k * (k - 1) * fk for k, fk in f.items()) / (n_rare * (n_rare - 1)) - 1, 0)
    return s_abund + len(rare_c) / cov + f1 / cov * gamma2


def jackknife1(counts):
    """First-order jackknife: S_obs + f1 (n-1)/n."""
    n, s, f = basics(counts)
    return s + f[1] * (n - 1) / n if n else 0.0


def coverage(counts):
    """Estimated sample coverage (Chao & Jost 2012): share of all errors that belong to types already seen."""
    n, s, f = basics(counts)
    if n == 0:
        return 0.0
    f1, f2 = f[1], f[2]
    if f1 == 0:
        return 1.0
    a = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2) if (f2 > 0 or f1 > 1) else 1.0
    return 1 - f1 / n * a


def predict_new(counts, m):
    """Expected number of NEW types found if m more errors are collected (Shen, Chao & Lin 2003)."""
    n, s, f = basics(counts)
    f1 = f[1]
    f0 = chao1(counts) - s
    if n == 0 or f0 <= 0 or f1 == 0:
        return 0.0
    return f0 * (1 - (1 - f1 / (n * f0 + f1)) ** m)


ESTIMATORS = {"chao1": chao1, "ichao1": ichao1, "ace": ace, "jackknife1": jackknife1}


def boot_ci(counts, fn, B=200, seed=0, z=1.96):
    """Bootstrap 95% interval for any estimator `fn` (the method used by iNEXT; Chao et al. 2014).
    Builds a full "assemblage": each observed type keeps a share C * x_i / n (C = sample coverage), and
    f0 = round(Chao1 - S_obs) unseen types share the remaining 1 - C equally. Draws B new samples of the
    same size n from it, applies `fn` to each, and returns estimate +- z * bootstrap SD, never below S_obs."""
    import numpy as np
    n, s, _ = basics(counts)
    est = fn(counts)
    if n < 2:  # one error or none: no spread can be estimated
        return float(s), float(est)
    c = coverage(counts)
    f0 = int(round(chao1(counts) - s))
    x = np.array([v for v in counts if v > 0], dtype=float)
    p = c * x / n
    if f0 > 0 and c < 1:
        p = np.concatenate([p, np.full(f0, (1 - c) / f0)])
    if p.sum() <= 0:
        return float(s), float(est)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    sims = [fn(rng.multinomial(n, p).tolist()) for _ in range(B)]
    sd = float(np.std(sims, ddof=1))
    return max(float(s), est - z * sd), est + z * sd
