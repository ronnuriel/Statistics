# ------------------------------------------------------------
# Tests for HW1 — Question 1 (Binomial & Negative Binomial)
# Author IDs:
#   ID1: 207884883
#   ID2:
# ------------------------------------------------------------

import math
import numpy as np
import pytest

from hw1 import (
    find_sample_size_binom, 
    find_sample_size_nbinom, 
    same_prob,
    empirical_centralized_third_moment,
    class_moment,
    plot_moments,
    plot_moments_smaller_variance,
    NFoldConv,
    plot_dist,
    evenBinom
)
from scipy import stats  # SciPy is required here


# =========================
# Helpers
# =========================
def closed_form_n_x1(p, alpha):
    """ n* = ceil( ln(1-alpha) / ln(1-p) ) """
    return int(math.ceil(math.log(1 - alpha) / math.log(1 - p)))


def prob_at_least_one(n, p):
    """P(X>=1) = 1 - (1-p)^n for X~Binom(n,p)"""
    return 1.0 - (1.0 - p) ** n


def min_n_by_nbinom(p, alpha, x, n_max=100000):
    """
    Reference calculator for NegBin:
    N = total trials to get x successes, K = failures = N - x  ~ nbinom(x, p).
    Find minimal n such that P(N <= n) = P(K <= n - x) >= alpha.
    """
    if not (0 < p < 1) or not (0 < alpha < 1) or not (isinstance(x, int) and x >= 1):
        raise ValueError("bad inputs for reference calc")
    for n in range(x, n_max + 1):
        k = n - x
        cdf = stats.nbinom.cdf(k, x, p) if k >= 0 else 0.0
        if cdf >= alpha:
            return n
    return None


# =========================
# Q1(a): Binomial, x=1 — בדיקה מול נוסחה סגורה ומינימליות
# =========================
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
    n = find_sample_size_binom(defective_rate=p, target_prob=alpha)
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
    n_binom = find_sample_size_binom(defective_rate=p, target_prob=alpha)

    # Negative Binomial — P(N≤n) עם Y~NegBin(x,p)
    n_nbinom = find_sample_size_nbinom(defective_rate=p, target_prob=alpha, x=x)
    assert isinstance(n_nbinom, int) and n_nbinom >= x

    # שתי הגישות חייבות להסכים
    assert n_nbinom == n_binom

    # אימות נכון עם SciPy: עובדים על K = N - x (כישלונות)
    k_n = n_nbinom - x
    k_prev = (n_nbinom - 1) - x

    def cdf_failures_leq(k):
        if k < 0:
            return 0.0
        return stats.nbinom.cdf(k, x, p)

    cdf_n = cdf_failures_leq(k_n)  # P(N ≤ n) = P(K ≤ n-x)
    cdf_prev = cdf_failures_leq(k_prev)  # P(N ≤ n-1) = P(K ≤ n-1-x)

    assert cdf_n >= alpha
    if n_nbinom > x:
        assert cdf_prev < alpha


# =========================
# Q1(b) — כללי: x>1 (בדיקת התאמה לעוגן רפרנס)
# =========================
@pytest.mark.parametrize(
    "p,alpha,x",
    [
        (0.10, 0.90, 5),  # חלק א' של compare_q1
        (0.30, 0.90, 15),  # חלק ב' של compare_q1
        (0.05, 0.95, 3),
        (0.20, 0.80, 7),
    ],
)
def test_q1b_nbinom_general_x_gt_1_against_reference(p, alpha, x):
    n_ref = min_n_by_nbinom(p, alpha, x)
    n_func = find_sample_size_nbinom(defective_rate=p, target_prob=alpha, x=x)
    assert isinstance(n_func, int) and n_func >= x
    assert n_func == n_ref

    # מינימליות
    k_n = n_func - x
    k_prev = (n_func - 1) - x
    cdf_n = stats.nbinom.cdf(k_n, x, p) if k_n >= 0 else 0.0
    cdf_prev = stats.nbinom.cdf(k_prev, x, p) if k_prev >= 0 else 0.0
    assert cdf_n >= alpha
    if n_func > x:
        assert cdf_prev < alpha


# =========================
# ולידציות קלט (לשתי הפונקציות)
# =========================
@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.2])
def test_q1_invalid_p_raises(bad_p):
    with pytest.raises(ValueError):
        find_sample_size_binom(defective_rate=bad_p, target_prob=0.85)
    with pytest.raises(ValueError):
        find_sample_size_nbinom(defective_rate=bad_p, target_prob=0.85, x=1)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.2, 1.3])
def test_q1_invalid_alpha_raises(bad_alpha):
    with pytest.raises(ValueError):
        find_sample_size_binom(defective_rate=0.03, target_prob=bad_alpha)
    with pytest.raises(ValueError):
        find_sample_size_nbinom(defective_rate=0.03, target_prob=bad_alpha, x=1)


@pytest.mark.parametrize("bad_x", [0, -1, 1.5, 2.0])
def test_q1_invalid_x_for_nbinom_raises(bad_x):
    with pytest.raises(ValueError):
        find_sample_size_nbinom(defective_rate=0.03, target_prob=0.85, x=bad_x)


# =========================
# תכונות נדרשות: מונוטוניות (על binom x=1)
# =========================
def test_q1_monotonic_in_alpha():
    p = 0.04
    n1 = find_sample_size_binom(p, target_prob=0.80)
    n2 = find_sample_size_binom(p, target_prob=0.85)
    n3 = find_sample_size_binom(p, target_prob=0.90)
    assert n1 <= n2 <= n3


def test_q1_monotonic_in_p():
    alpha = 0.9
    n_small_p = find_sample_size_binom(defective_rate=0.02, target_prob=alpha)
    n_big_p = find_sample_size_binom(defective_rate=0.05, target_prob=alpha)
    assert n_big_p <= n_small_p


def test_q1_deterministic_same_inputs_same_output():
    p, alpha = 0.05, 0.9
    n1 = find_sample_size_binom(p, alpha)
    n2 = find_sample_size_binom(p, alpha)
    assert n1 == n2


# =========================
# אופציונלי: compare_q1 אם קיים
# =========================
def test_q1_compare_q1_if_exists():
    try:
        from hw1 import compare_q1
    except Exception:
        pytest.skip("compare_q1 not implemented; skipping.")
        return

    # ברירות המחדל: (0.10, 0.90, 5) ו-(0.30, 0.90, 15)
    n1, n2 = compare_q1()
    assert isinstance(n1, int) and isinstance(n2, int)

    # אימות מול מחשבון רפרנס
    assert n1 == min_n_by_nbinom(0.10, 0.90, 5)
    assert n2 == min_n_by_nbinom(0.30, 0.90, 15)


# =========================
# השוואת שיטות: binom vs nbinom
# =========================

def test_q1_compare_q1_methods_agree():
    """
    verify that compare_q1 returns the same (n1,n2) with both methods
    """
    try:
        from hw1 import compare_q1
    except Exception:
        pytest.skip("compare_q1 not implemented; skipping.")
        return

    n_binom = compare_q1(method="binom")
    n_nbinom = compare_q1(method="nbinom")
    assert isinstance(n_binom, tuple) and isinstance(n_nbinom, tuple)
    assert n_binom == n_nbinom


@pytest.mark.parametrize(
    "p,alpha,x",
    [
        (0.03, 0.85, 1),  # x=1 (נוסחה סגורה מול NegBin)
        (0.03, 0.95, 1),
        (0.10, 0.90, 5),  # x>1 כמו ב-compare_q1 (חלק א)
        (0.30, 0.90, 15),  # x>1 כמו ב-compare_q1 (חלק ב)
        (0.05, 0.95, 3),  # עוד מקרה כללי לבדיקה
    ],
)
def test_q1_binom_equals_nbinom_general(p, alpha, x):
    """
    verify that find_sample_size_binom(p,alpha,x) == find_sample_size_nbinom(p,alpha,x)
    for several (p,alpha,x) including x>1
    """
    n_b = find_sample_size_binom(defective_rate=p, target_prob=alpha, x=x)
    n_nb = find_sample_size_nbinom(defective_rate=p, target_prob=alpha, x=x)
    assert isinstance(n_b, int) and isinstance(n_nb, int)
    assert n_b == n_nb


def test_same_prob_basic_behavior():
    """
    Tests that same_prob returns an integer n such that:
    P(X>=k1; n, p1) ≈ P(X>=k2; n, p2) within atol=1e-2,
    when both probabilities are > 0.
    """
    n = same_prob()
    assert isinstance(n, int)

    p1, k1 = 0.10, 5
    p2, k2 = 0.30, 15
    atol = 1e-2

    p_case1 = stats.binom.sf(k1 - 1, n, p1)
    p_case2 = stats.binom.sf(k2 - 1, n, p2)

    assert p_case1 > 0 and p_case2 > 0
    assert np.isclose(p_case1, p_case2, atol=atol), (
        f"Probabilities not close enough: {p_case1} vs {p_case2}"
    )

#
# ----------------------- Question 2 -----------------------
#

def test_q2_empirical_centralized_third_moment_runs_and_is_float():
    """
    2.A: נדרש לבצע סימולציה אמפירית (k=100 נסיונות) ולחשב מומנט מרכזי שלישי של Y.
    הבדיקה בודקת רק שהפונקציה רצה ומחזירה מספר סופי (float),
    בלי לאכוף קרבה מספרית לערך תיאורטי (כדי לשקף את אופי הסימולציה בשאלה).
    """
    try:
        from hw1 import empirical_centralized_third_moment
    except Exception:
        pytest.skip("empirical_centralized_third_moment not implemented; skipping.")
        return

    emp = empirical_centralized_third_moment()
    assert isinstance(emp, (float, np.floating))
    assert np.isfinite(emp)


def test_q2_class_moment_runs_and_is_float():
    """
    2.B: נדרש לחשב את המומנט השלישי המרכזי באופן תיאורטי/כיתתי.
    בודקים שהפונקציה רצה ומחזירה מספר סופי (float).
    """
    try:
        from hw1 import class_moment
    except Exception:
        pytest.skip("class_moment not implemented; skipping.")
        return

    theo = class_moment()
    assert isinstance(theo, (float, np.floating))
    assert np.isfinite(theo)


def test_q2_plot_moments_runs_without_error():
    """
    2.C: פונקציה שמציירת/מחזירה אובייקט גרפי – השאלה דורשת הפקה/הדגמה,
    לא אימות תקני מסוים. הבדיקה רק מאשרת שהיא רצה ללא חריגות.
    """
    try:
        from hw1 import plot_moments
    except Exception:
        pytest.skip("plot_moments not implemented; skipping.")
        return

    _ = plot_moments()  # אין דרישה להגדרת ערך החזרה


def test_q2_plot_moments_smaller_variance_runs_without_error():
    """
    2.D: כמו 2.C – רק לבדוק שהפונקציה רצה, בהתאם לדרישה בשאלה.
    """
    try:
        from hw1 import plot_moments_smaller_variance
    except Exception:
        pytest.skip("plot_moments_smaller_variance not implemented; skipping.")
        return

    _ = plot_moments_smaller_variance()


#
# ----------------------- Question 3 -----------------------
#

def _expected_binomial_dict(n, p):
    from math import comb
    return {k: comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(n + 1)}


def _to_dict_distribution(Q):
    """
    מאחד פלט ל-dict עבור השוואה נוחה.
    תומך ב-{val:prob} או np.array([[values],[probs]]).
    """
    if isinstance(Q, dict):
        return {int(k): float(v) for k, v in Q.items()}
    Q = np.asarray(Q)
    if Q.ndim == 2 and Q.shape[0] == 2:
        vals, probs = Q
        return {int(v): float(p) for v, p in zip(vals, probs)}
    raise TypeError("Unexpected NFoldConv output type (expected dict or 2xM array)")


def test_q3_nfoldconv_coin_basic_properties_and_binomial_case():
    """
    3.A: השאלה מבקשת לייצר את התפלגות סכום n חזרות i.i.d. עם התפלגות P.
    כדי לוודא נכונות לוגית בצורה התואמת את רוח השאלה, נבדוק:
      1) הפונקציה רצה ומחזירה התפלגות חוקית (סכום הסתברויות ~ 1, הסתברויות >= 0)
      2) במקרה מבחן קלאסי: P של מטבע הוגן, n=3 ⇒ תואם לבינומי
    לא נכפה פורמט פלט ספציפי: מתקבלים גם dict וגם np.array([[values],[probs]]).
    """
    try:
        from hw1 import NFoldConv
    except Exception:
        pytest.skip("NFoldConv not implemented; skipping.")
        return

    # קלט של מטבע הוגן
    P_dict = {0: 0.5, 1: 0.5}
    n = 3

    Q = NFoldConv(P_dict, n)
    Qd = _to_dict_distribution(Q)

    # 1) בדיקות בסיסיות של התפלגות
    total_prob = sum(Qd.values())
    assert np.isclose(total_prob, 1.0, atol=1e-12)
    assert all(p >= -1e-15 for p in Qd.values())  # מאפשר רפרוף נומרי קטן מאוד
    assert all(isinstance(k, (int, np.integer)) for k in Qd.keys())

    # 2) התאמה לבינומי במקרה מבחן סטנדרטי
    expected = _expected_binomial_dict(n, 0.5)
    for k, p_exp in expected.items():
        p_got = Qd.get(k, 0.0)
        assert np.isclose(p_got, p_exp, atol=1e-12), f"k={k}: got {p_got}, expected {p_exp}"


def test_q3_plot_dist_runs_without_error():
    """
    3.B: הפונקציה אמורה להציג/להחזיר ציור של התפלגות בדידה.
    נבדוק רק שהיא רצה ללא חריגות, בהתאם לרוח השאלה.
    """
    try:
        from hw1 import plot_dist
    except Exception:
        pytest.skip("plot_dist not implemented; skipping.")
        return

    P = {0: 0.2, 1: 0.3, 3: 0.5}
    _ = plot_dist(P)

def closed_form(n: int, p: float) -> float:
    # P(X is even) = (1 + (1 - 2p)^n) / 2  for X ~ Binom(n, p)
    return 0.5 * (1.0 + (1.0 - 2.0 * p) ** n)


@pytest.mark.parametrize(
    "n,p",
    [
        (0, 0.00),
        (0, 0.25),
        (0, 0.50),
        (0, 0.90),
        (1, 0.50),
        (2, 0.10),
        (3, 0.50),
        (4, 0.50),
        (5, 0.30),
        (10, 0.10),
        (10, 0.50),
        (12, 0.75),
        (25, 0.33),
    ],
)
def test_evenBinom_matches_closed_form(n, p):
    val = evenBinom(n, p)
    exp = closed_form(n, p)
    assert isinstance(val, (float, np.floating))
    assert np.isfinite(val)
    assert np.isclose(val, exp, atol=1e-12), f"n={n}, p={p}: got {val}, expected {exp}"


@pytest.mark.parametrize(
    "n,p",
    [(k, q) for k in [0, 1, 2, 3, 5, 10, 25] for q in [0.0, 0.1, 0.5, 0.9, 1.0]]
)
def test_evenBinom_in_probability_range(n, p):
    val = evenBinom(n, p)
    assert 0.0 - 1e-15 <= val <= 1.0 + 1e-15, f"Out of [0,1] range: {val}"
    assert np.isfinite(val)


def test_evenBinom_edge_cases_n0_any_p():
    # אם n=0 אז X=0 בהסתברות 1 -> זוגי תמיד
    for p in [0.0, 0.2, 0.5, 1.0]:
        assert np.isclose(evenBinom(0, p), 1.0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 25])
def test_evenBinom_edge_cases_p0(n):
    # אם p=0 אז X=0 תמיד -> זוגי תמיד
    assert np.isclose(evenBinom(n, 0.0), 1.0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 25])
def test_evenBinom_edge_cases_p1(n):
    # אם p=1 אז X=n תמיד -> זוגי אם ורק אם n זוגי
    expected = 1.0 if n % 2 == 0 else 0.0
    assert np.isclose(evenBinom(n, 1.0), expected, atol=1e-12)

# test_evenBinomFormula.py
import numpy as np
import pytest

# מניח שהפונקציה מוגדרת ב-hw1.py
from hw1 import evenBinomFormula


def closed_form(n: int, p: float) -> float:
    # P(X is even) = (1 + (1 - 2p)^n) / 2
    return 0.5 * (1.0 + (1.0 - 2.0 * p) ** n)


@pytest.mark.parametrize(
    "n,p",
    [
        (0, 0.00),
        (0, 0.25),
        (0, 0.50),
        (0, 0.90),
        (1, 0.50),
        (2, 0.10),
        (3, 0.50),
        (4, 0.50),
        (5, 0.30),
        (10, 0.10),
        (10, 0.50),
        (12, 0.75),
        (25, 0.33),
    ],
)
def test_evenBinomFormula_matches_closed_form_and_in_range(n, p):
    val = evenBinomFormula(n, p)
    exp = closed_form(n, p)
    assert isinstance(val, (float, np.floating))
    assert np.isfinite(val)
    # ערך נכון לפי הנוסחה
    assert np.isclose(val, exp, atol=1e-12), f"n={n}, p={p}: got {val}, expected {exp}"
    # טווח הסתברויות
    assert -1e-15 <= val <= 1.0 + 1e-15


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 25])
def test_evenBinomFormula_p_equals_half_is_half(n):
    # כש-p=0.5, (1-2p)^n = 0^n ⇒ עבור n>=1 זה 0 ⇒ תוצאה 0.5; וגם ל-n=0 מתקבל 1.
    val = evenBinomFormula(n, 0.5)
    expected = 1.0 if n == 0 else 0.5
    assert np.isclose(val, expected, atol=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 10, 24])
def test_evenBinomFormula_edge_cases_p0_p1(n):
    # p=0: תמיד 0 הצלחות ⇒ זוגי ⇒ הסתברות 1
    assert np.isclose(evenBinomFormula(n, 0.0), 1.0, atol=1e-12)
    # p=1: תמיד n הצלחות ⇒ זוגי אם ורק אם n זוגי
    expected = 1.0 if (n % 2 == 0) else 0.0
    assert np.isclose(evenBinomFormula(n, 1.0), expected, atol=1e-12)
