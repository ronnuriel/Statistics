# ------------------------------------------------------------
# Tests for HW1 — Question 1 (Binomial & Negative Binomial)
# Author IDs:
#   ID1: 207884883
#   ID2:
# ------------------------------------------------------------

import math
import numpy as np
import pytest

from hw1 import find_sample_size_binom, find_sample_size_nbinom
from scipy import stats  # SciPy is required here


# =========================
# Helpers
# =========================
def closed_form_n_x1(p, alpha):
    """ n* = ceil( ln(1-alpha) / ln(1-p) ) """
    return int(math.ceil(math.log(1 - alpha) / math.log(1 - p)))

def prob_at_least_one(n, p):
    """P(X>=1) = 1 - (1-p)^n for X~Binom(n,p)"""
    return 1.0 - (1.0 - p)**n


# =========================
# Q1(a): Binomial, x=1 — בדיקה מול נוסחה סגורה ומינימליות
# =========================
@pytest.mark.q1
@pytest.mark.parametrize(
    "p,alpha,expected",
    [
        (0.03, 0.85, 63),
        (0.03, 0.95, 99),
        (0.05, 0.85, 37),  # 37 (לא 45)
        (0.05, 0.95, 59),
    ],
)
def test_q1a_binom_closed_form_minimal(p, alpha, expected):
    n = find_sample_size_binom(p=p, alpha=alpha)
    assert isinstance(n, int) and n >= 1
    assert n == expected

    # עומד בסף
    assert prob_at_least_one(n, p) >= alpha
    # מינימליות
    if n > 1:
        assert prob_at_least_one(n - 1, p) < alpha


# =========================
# Q1(b): Negative Binomial, x=1 — עקביות עם Binomial
# SciPy: nbinom(r, p) מודל לכמות הכישלונות K לפני r הצלחות.
# אם N = מספר הניסיונות הכולל עד r הצלחות, אז: N = K + r  ⇒  K = N - r
# לכן: P(N ≤ n) = P(K ≤ n - r) = nbinom.cdf(n - r; r, p)
# =========================
@pytest.mark.q1
@pytest.mark.parametrize(
    "p,alpha",
    [
        (0.03, 0.85),
        (0.03, 0.95),
        (0.05, 0.85),
        (0.05, 0.95),
        (0.20, 0.90),
    ],
)
def test_q1b_nbinom_matches_binom_when_x1(p, alpha):
    x = 1

    # Binomial (x=1) — נוסחה סגורה
    n_binom = find_sample_size_binom(p=p, alpha=alpha)

    # Negative Binomial — P(N≤n) עם Y~NegBin(x,p)
    n_nbinom = find_sample_size_nbinom(p=p, alpha=alpha, x=x)
    assert isinstance(n_nbinom, int) and n_nbinom >= x

    # שתי הגישות חייבות להסכים
    assert n_nbinom == n_binom

    # אימות נכון עם SciPy: עובדים על K = N - x (כישלונות)
    k_n    = n_nbinom - x
    k_prev = (n_nbinom - 1) - x

    # SciPy מגדירה CDF רק ל-k≥0; אם k<0 אז P(K≤k)=0
    def cdf_failures_leq(k):
        if k < 0:
            return 0.0
        return stats.nbinom.cdf(k, x, p)

    cdf_n   = cdf_failures_leq(k_n)       # P(N ≤ n) = P(K ≤ n-x)
    cdf_prev = cdf_failures_leq(k_prev)   # P(N ≤ n-1) = P(K ≤ n-1-x)

    assert cdf_n >= alpha
    if n_nbinom > x:
        assert cdf_prev < alpha


# =========================
# ולידציות קלט (לשתי הפונקציות)
# =========================
@pytest.mark.q1
@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.2])
def test_q1_invalid_p_raises(bad_p):
    with pytest.raises(ValueError):
        find_sample_size_binom(p=bad_p, alpha=0.85)
    with pytest.raises(ValueError):
        find_sample_size_nbinom(p=bad_p, alpha=0.85, x=1)

@pytest.mark.q1
@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.2, 1.3])
def test_q1_invalid_alpha_raises(bad_alpha):
    with pytest.raises(ValueError):
        find_sample_size_binom(p=0.03, alpha=bad_alpha)
    with pytest.raises(ValueError):
        find_sample_size_nbinom(p=0.03, alpha=bad_alpha, x=1)

@pytest.mark.q1
@pytest.mark.parametrize("bad_x", [0, -1, 1.5, 2.0])
def test_q1_invalid_x_for_nbinom_raises(bad_x):
    with pytest.raises(ValueError):
        find_sample_size_nbinom(p=0.03, alpha=0.85, x=bad_x)


# =========================
# תכונות נדרשות: מונוטוניות
#   • ככל ש-alpha גדל, n לא קטן
#   • ככל ש-p גדל, n לא גדל (קל יותר להגיע ללפחות אחד)
# =========================
@pytest.mark.q1
def test_q1_monotonic_in_alpha():
    p = 0.04
    n1 = find_sample_size_binom(p, alpha=0.80)
    n2 = find_sample_size_binom(p, alpha=0.85)
    n3 = find_sample_size_binom(p, alpha=0.90)
    assert n1 <= n2 <= n3

@pytest.mark.q1
def test_q1_monotonic_in_p():
    alpha = 0.9
    n_small_p = find_sample_size_binom(p=0.02, alpha=alpha)
    n_big_p   = find_sample_size_binom(p=0.05, alpha=alpha)
    assert n_big_p <= n_small_p

@pytest.mark.q1
def test_q1_deterministic_same_inputs_same_output():
    p, alpha = 0.05, 0.9
    n1 = find_sample_size_binom(p, alpha)
    n2 = find_sample_size_binom(p, alpha)
    assert n1 == n2
