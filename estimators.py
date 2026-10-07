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


def pivot_ci(counts, fn, B=200, seed=0, alpha=0.05):
    """Bias-calibrated bootstrap for any estimator `fn`. Returns (corrected estimate, lower, upper).

    In a bootstrap world the true number of types is KNOWN, so we can measure how far `fn` falls short there.
    1. Build a world from the data: each observed type keeps a share C x_i / n (C = sample coverage), and m unseen
       types share the remaining 1 - C equally. m is chosen so that the world reproduces the observed number of
       singletons (exact expected value, no simulation). Its true number of types is S_obs + m.
    2. Draw B samples of the same size n, apply `fn`, and record the ratio r = (S_obs + m) / estimate.
    3. Corrected estimate = fn(counts) x median(r); interval = fn(counts) x the 2.5% and 97.5% quantiles of r.
    The lower end is never below S_obs.
    Caveat: with many very rare types the sample holds little information about them, and no interval can be
    trusted; the pilot measures how often this one covers the truth."""
    import numpy as np
    n, s, f = basics(counts)
    est = fn(counts)
    if n < 2:
        return float(est), float(s), float(est)
    c = coverage(counts)
    x = np.array([v for v in counts if v > 0], dtype=float)
    pdet = c * x / n
    f0 = int(round(max(est, chao1(counts)) - s))
    m = 0
    if c < 1 and f0 > 0:
        best = None
        for k in sorted({f0, *[int(f0 * t) for t in (1.25, 1.5, 2, 2.5, 3, 4, 6, 8, 10)]}):
            if k < 1:
                continue
            q = (1 - c) / k
            ef1 = float((n * pdet * (1 - pdet) ** (n - 1)).sum()) + k * n * q * (1 - q) ** (n - 1)
            if best is None or abs(ef1 - f[1]) < best[0]:
                best = (abs(ef1 - f[1]), k)
        m = best[1]
    p = np.concatenate([pdet, np.full(m, (1 - c) / m)]) if m else pdet
    if p.sum() <= 0:
        return float(est), float(s), float(est)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    r = [(s + m) / e for e in (fn(rng.multinomial(n, p).tolist()) for _ in range(B)) if e > 0]
    if not r:
        return float(est), float(s), float(est)
    lo, mid, hi = np.quantile(r, [alpha / 2, 0.5, 1 - alpha / 2])
    return float(max(s, est * mid)), float(max(s, est * lo)), float(max(s, est * hi))


def unseen_ci(counts, B=200, seed=0, alpha=0.05):
    """Calibrated 95% interval for the unseen share (the share of all errors whose type was not seen).
    In the bootstrap world (as in boot_ci) the true unseen share of each resample is KNOWN: the total
    probability of the types the resample missed. So we record the estimation error (estimate - truth)
    over B resamples and return estimate - [97.5%, 2.5%] quantiles of that error, clipped to [0, 1]."""
    import numpy as np
    n, s, _ = basics(counts)
    est = 1 - coverage(counts)
    if n < 2:
        return est, est
    c = 1 - est
    f0 = int(round(chao1(counts) - s))
    x = np.array([v for v in counts if v > 0], dtype=float)
    p = c * x / n
    if f0 > 0 and c < 1:
        p = np.concatenate([p, np.full(f0, (1 - c) / f0)])
    if p.sum() <= 0:
        return est, est
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    err = []
    for _ in range(B):
        draw = rng.multinomial(n, p)
        truth = float(p[draw == 0].sum())
        err.append((1 - coverage(draw.tolist())) - truth)
    lo_e, hi_e = np.quantile(err, [alpha / 2, 1 - alpha / 2])
    return float(min(1, max(0, est - hi_e))), float(min(1, max(0, est - lo_e)))
