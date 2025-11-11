###### Your ID ######
# ID1: 207884883
# ID2: 
#####################

# imports 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 1 ###

def find_sample_size_binom(p=0.03, alpha=0.85, x=1):
    """
    Finds minimal n such that P(X >= x) >= alpha, where X~Binomial(n,p).

    • For x = 1 → uses closed-form: n = ceil(ln(1-alpha)/ln(1-p))
    • For x > 1 → computes 1 - CDF(x-1; n, p) iteratively.

    Validates inputs: 0 < p < 1, 0 < alpha < 1, x ≥ 1 (integer)
    """
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if x < 1 or not isinstance(x, int):
        raise ValueError("x must be a positive integer")

    if x == 1:
        n_real = np.log(1 - alpha) / np.log(1 - p)
        n = int(np.ceil(n_real))
        print(f"[Binom] p={p:.2%}, α={alpha:.2%}, x={x} → n={n}")
        return n

    # General case: numerical search
    n_values = np.arange(x, 10000, dtype=int)
    probs = 1 - stats.binom.cdf(x - 1, n_values, p)
    mask = probs >= alpha
    if not np.any(mask):
        print(f"[Binom] No n found for p={p:.2%}, α={alpha:.2%}, x={x}")
        return None

    n = int(n_values[np.argmax(mask)])
    print(f"[Binom] p={p:.2%}, α={alpha:.2%}, x={x} → n={n}")
    return n


def find_sample_size_nbinom(p=0.03, alpha=0.85, x=1):
    """
    Using NBinom to returns the minimal number of samples required to have requested probability
    of receiving at least x defective products from a production line with a defective rate.

    Validates inputs:
        0 < p < 1, 0 < alpha < 1, x ≥ 1 (integer)
    """
    # --- input validation ---
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if x < 1 or not isinstance(x, int):
        raise ValueError("x must be a positive integer")

    # search over total trials n (not failures!)
    n_values = np.arange(x, 10000, dtype=int)  # minimal trials is x
    k_values = n_values - x  # failures = trials - successes
    probs = stats.nbinom.cdf(k_values, x, p)  # P(K <= n - x) = P(N <= n)

    mask = probs >= alpha
    if not np.any(mask):
        print(f"[NegBinom] No n found for p={p:.2%}, alpha={alpha:.2%}, x={x}")
        return None

    n = int(n_values[np.argmax(mask)])
    print(f"[NegBinom] p={p:.2%}, alpha={alpha:.2%}, x={x} → n={n}")
    return n


def compare_q1(p1=0.10, alpha1=0.90, x1=5,
               p2=0.30, alpha2=0.90, x2=15,
               method="binom"):
    """
    Returns (n1, n2): minimal #samples for:
      1) p1 defective, P(X >= x1) >= alpha1
      2) p2 defective, P(X >= x2) >= alpha2

    method:
      - "binom" (default): uses find_sample_size_binom (supports x>=1)
      - "nbinom": uses find_sample_size_nbinom (supports x>=1)
    """
    # Validates inputs:
    for p in (p1, p2):
        if not (0 < p < 1):
            raise ValueError("p must be between 0 and 1")
    for a in (alpha1, alpha2):
        if not (0 < a < 1):
            raise ValueError("alpha must be between 0 and 1")
    for x in (x1, x2):
        if not (isinstance(x, int) and x >= 1):
            raise ValueError("x must be a positive integer")

    if method == "binom":
        n1 = find_sample_size_binom(p=p1, alpha=alpha1, x=x1)
        n2 = find_sample_size_binom(p=p2, alpha=alpha2, x=x2)
        print(f"[CompareQ1/binom] Case1: p={p1:.0%}, α={alpha1:.0%}, x={x1} → n={n1}")
        print(f"[CompareQ1/binom] Case2: p={p2:.0%}, α={alpha2:.0%}, x={x2} → n={n2}")
        return (n1, n2)

    elif method == "nbinom":
        n1 = find_sample_size_nbinom(p=p1, alpha=alpha1, x=x1)
        n2 = find_sample_size_nbinom(p=p2, alpha=alpha2, x=x2)
        print(f"[CompareQ1/nbinom] Case1: p={p1:.0%}, α={alpha1:.0%}, x={x1} → n={n1}")
        print(f"[CompareQ1/nbinom] Case2: p={p2:.0%}, α={alpha2:.0%}, x={x2} → n={n2}")
        return (n1, n2)

    else:
        raise ValueError('method must be one of: "binom", "nbinom"')


def same_prob(p=0.10, x=5, n_max=100000, atol=1e-2):
    """
    Finds the minimal n such that:
        P_binom(X >= x; n, p) ≈ P_nbinom(N <= n; x, p)
    within absolute tolerance `atol` using np.isclose.

    Comparison is performed only when both probabilities > 0.

    Returns:
        n (int) if found, else None.
    """
    # input validation
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1")
    if not (isinstance(x, int) and x >= 1):
        raise ValueError("x must be a positive integer")
    if not (isinstance(n_max, int) and n_max >= x):
        raise ValueError("n_max must be an integer ≥ x")
    if atol <= 0:
        raise ValueError("atol must be positive")

    for n in range(x, n_max + 1):
        # Binomial: P(X >= x) = 1 - CDF(x-1; n, p)
        p_binom = 1.0 - stats.binom.cdf(x - 1, n, p)

        # NegBin: N = total trials to get x successes; K = failures = N - x
        k = n - x
        p_nbinom = stats.nbinom.cdf(k, x, p) if k >= 0 else 0.0

        # compare only if both probabilities are strictly positive
        if p_binom > 0.0 and p_nbinom > 0.0:
            if np.isclose(p_binom, p_nbinom, atol=atol):
                print(f"[same_prob] p={p:.2%}, x={x}, n={n} "
                      f"→ Binom={p_binom:.4f}, NegBin={p_nbinom:.4f}")
                return n

    print(f"[same_prob] No n found within atol={atol} up to n_max={n_max} (p={p}, x={x})")
    return None


### Question 2 ###

def empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=100, seed=None):
    """
    Create k experiments where X is sampled. Calculate the empirical centralized third moment of Y based 
    on your k experiments.
    """
    if seed is not None:
        np.random.seed(seed)

    return empirical_moment


def class_moment():
    return moment


def plot_moments():
    return dist_var


def plot_moments_smaller_variance():
    return dist_var


### Question 3 ###

def NFoldConv(P, n):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables, 
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """

    return Q


def plot_dist(P):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """

    pass


### Qeustion 4 ###

def evenBinom(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).
    
    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """

    return prob


def evenBinomFormula(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.
    
    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """

    return prob


### Question 5 ###

def three_RV(values, joint_probs):
    """
 
    Input:          
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution
    
    Returns:
    - v: The variance of X + Y + Z. (you cannot create the RV U = X + Y + Z) 
    """

    return v


def three_RV_pairwise_independent(values, joint_probs):
    """
 
    Input:          
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution
    
    Returns:
    - v: The variance of X + Y + Z. (you cannot create the RV U = X + Y + Z)
    """

    return v


def is_pairwise_collectively(X, Y, Z, joint_probs):
    """

    Input:
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution
    
    Returns:
    TRUE or FALSE
    """

    pass


### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    """

    pass
