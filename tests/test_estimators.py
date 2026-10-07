"""Checks for the estimators: hand-computed values and a simulation sanity check.
Run: python -m tests.test_estimators   (no numpy needed)."""
import random

from estimators import ace, chao1, chao1_ci, coverage, ichao1, jackknife1, predict_new

ok = 0


def check(cond, msg):
    global ok
    assert cond, msg
    ok += 1


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# no singletons: nothing is missing
check(chao1([3, 4, 5]) == 3 and jackknife1([3, 4, 5]) == 3 and coverage([3, 4, 5]) == 1.0, "no singletons")
# hand computation: counts 1,1,1,2,5 -> n=10, S=5, f1=3, f2=1
check(close(chao1([1, 1, 1, 2, 5]), 5 + 9 / 10 * 3 * 2 / (2 * 2)), "chao1 by hand")
check(close(jackknife1([1, 1, 1, 2, 5]), 5 + 3 * 9 / 10), "jackknife1 by hand")
check(close(coverage([1, 1, 1, 2, 5]), 1 - 3 / 10 * (9 * 3 / (9 * 3 + 2))), "coverage by hand")
# iChao1 adds a non-negative term
check(ichao1([1, 1, 1, 1, 2, 2, 3, 4, 9]) >= chao1([1, 1, 1, 1, 2, 2, 3, 4, 9]), "ichao1 >= chao1")
# interval contains the estimate and never goes below S_obs
lo, hi = chao1_ci([1, 1, 1, 1, 2, 2, 3, 7])
check(lo >= 8 - 1e-9 and lo <= chao1([1, 1, 1, 1, 2, 2, 3, 7]) <= hi, "chao1 interval")
# predict_new: zero extra samples -> zero new types; grows with m, bounded by chao1 - S_obs
c = [1, 1, 1, 2, 3, 6]
check(predict_new(c, 0) == 0 and predict_new(c, 10) < predict_new(c, 100) <= chao1(c) - 6 + 1e-9, "predict_new")
# ACE equals S_obs when no rare singletons
check(ace([12, 15, 3, 3]) == 4, "ace without singletons")

# simulation: 40 equally common types, 60 draws -> Chao1 should be close to 40 on average and beat S_obs
rng = random.Random(0)
s_obs, est = [], []
for _ in range(300):
    draws = [rng.randrange(40) for _ in range(60)]
    counts = [draws.count(k) for k in range(40)]
    s_obs.append(sum(1 for x in counts if x))
    est.append(chao1(counts))
m_obs, m_est = sum(s_obs) / 300, sum(est) / 300
check(m_obs < m_est and abs(m_est - 40) < abs(m_obs - 40) and abs(m_est - 40) < 4, f"simulation ({m_obs:.1f}, {m_est:.1f})")

# bootstrap interval: contains the estimate, never below S_obs, reproducible with the same seed
from estimators import boot_ci
c = [1, 1, 1, 1, 2, 2, 3, 5, 8]
lo, hi = boot_ci(c, chao1, B=100, seed=1)
check(lo >= 9 - 1e-9 and lo <= chao1(c) <= hi and boot_ci(c, chao1, B=100, seed=1) == (lo, hi), "boot_ci")
lo2, hi2 = boot_ci([4, 5, 6], ace, B=50)
check(lo2 == 3.0 and hi2 >= 3.0, "boot_ci with nothing missing: lower end is S_obs")

check(boot_ci([1], chao1) == (1.0, 1.0) and boot_ci([], chao1) == (0.0, 0.0), "boot_ci with 0 or 1 error")

print(f"ALL {ok} ESTIMATOR CHECKS PASSED")
