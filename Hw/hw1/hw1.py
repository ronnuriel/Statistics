###### Your ID ######
# ID1:
# ID2: 207884883
#####################

# imports 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from math import comb


### Question 1 ###

def find_sample_size_binom(defective_rate=0.03, target_prob=0.85, x=1):
    """
        Using Binom to returns the minimal number of samples required to have requested probability
        (target_prob) of receiving at least x defective products from a production line
        with a defective_rate.

        • For x = 1 → uses a closed-form formula:
            # P(at least 1 defective) = 1 - P(0 defective) >= target_prob
            # 1 - (1 - defective_rate)^n >= target_prob
            # (1 - defective_rate)^n <= 1 - target_prob
            # n * log(1 - defective_rate) <= log(1 - target_prob)
            # n >= log(1 - target_prob) / log(1 - defective_rate)  (Dividing by negative flips inequality)

        • For x > 1 → computes 1 - CDF(x-1; n, p) iteratively.

        Validates inputs:
          0 < defective_rate < 1
          0 < target_prob < 1
          x ≥ 1 (integer)
    """

    # --- Input Validation ---
    if not (0 < defective_rate < 1):
        raise ValueError("defective_rate must be between 0 and 1")
    if not (0 < target_prob < 1):
        raise ValueError("target_prob must be between 0 and 1")
    if x < 1 or not isinstance(x, int):
        raise ValueError("x must be a positive integer")

    if x == 1:
        n_real = np.log(1 - target_prob) / np.log(1 - defective_rate)
        n = int(np.ceil(n_real))
        print(f"To have a {target_prob:.0%} probability of at least one defective product, "
              f"we need to ask for {n} independent samples.")

        print(f"[Binom] p={defective_rate:.2%}, α={target_prob:.2%}, x={x} → n={n}")
        return n

    # General case: numerical search
    n_values = np.arange(x, 10000, dtype=int)
    probs = 1 - stats.binom.cdf(x - 1, n_values, defective_rate)
    mask = probs >= target_prob
    if not np.any(mask):
        print(f"[Binom] No n found for p={defective_rate:.2%}, α={target_prob:.2%}, x={x}")
        return None

    n = int(n_values[np.argmax(mask)])
    print(f"[Binom] p={defective_rate:.2%}, α={target_prob:.2%}, x={x} → n={n}")
    return n


def find_sample_size_nbinom(defective_rate=0.03, target_prob=0.85, x=1):
    """
    Using NBinom to return the minimal number of samples (n) required to have
    a requested probability (target_prob) of receiving at least x defective products.

    This function finds the minimal n such that P(N <= n) >= target_prob,
    where N is the total number of trials to get x successes (defective products).

    --- Technical Details ---
    It uses the Percent Point Function (ppf), which is the inverse of the CDF.
    1.  We ask for the number of *failures* (k) required to achieve 'target_prob'.
        `k = stats.nbinom.ppf(target_prob, x, defective_rate)`
    2.  The total number of samples (n) is the sum of failures and successes.
        `n = k + x`

    Validates inputs:
      0 < defective_rate < 1
      0 < target_prob < 1
      x ≥ 1 (integer)
    """

    # --- Input Validation ---
    if not (0 < defective_rate < 1):
        raise ValueError("defective_rate must be between 0 and 1")
    if not (0 < target_prob < 1):
        raise ValueError("target_prob must be between 0 and 1")
    if x < 1 or not isinstance(x, int):
        raise ValueError("x must be a positive integer")

    # nbinom.ppf(q, n, p) returns the number of failures (k) that gives cumulative probability q.
    # Here 'n' in standard scipy notation is our number of successes 'x'.

    # ppf(q, n, p) returns the number of *failures* (k) for a given probability q.
    # In scipy's nbinom, 'n' is the number of successes (our 'x').
    k_failures = stats.nbinom.ppf(target_prob, x, defective_rate)

    # Total samples = failures + successes
    n_total = int(k_failures + x)

    # Use the consistent print format
    print(f"[NegBinom] To have a {target_prob:.0%} probability of at least {x} defective product(s), "
          f"we need to ask for {n_total} independent samples.")

    return n_total


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

    if method == "binom":
        n1 = find_sample_size_binom(defective_rate=p1, target_prob=alpha1, x=x1)
        n2 = find_sample_size_binom(defective_rate=p2, target_prob=alpha2, x=x2)
        print(f"[CompareQ1/binom] Case1: p={p1:.0%}, α={alpha1:.0%}, x={x1} → n={n1}")
        print(f"[CompareQ1/binom] Case2: p={p2:.0%}, α={alpha2:.0%}, x={x2} → n={n2}")

    elif method == "nbinom":
        n1 = find_sample_size_nbinom(defective_rate=p1, target_prob=alpha1, x=x1)
        n2 = find_sample_size_nbinom(defective_rate=p2, target_prob=alpha2, x=x2)
        print(f"[CompareQ1/nbinom] Case1: p={p1:.0%}, α={alpha1:.0%}, x={x1} → n={n1}")
        print(f"[CompareQ1/nbinom] Case2: p={p2:.0%}, α={alpha2:.0%}, x={x2} → n={n2}")

    else:
        raise ValueError('method must be one of: "binom", "nbinom"')

    return n1, n2


def same_prob(p1=0.1, k1=5, p2=0.3, k2=15):
    # Define a reasonable search range for n (e.g., from k2 up to 2000).
    # Start from k2 because we need at least k2 samples to have k2 successes.
    n_values = np.arange(k2, 2001)

    # Vectorized calculation of P(X >= k) for all n values at once.
    # binom.sf(k-1, n, p) is equivalent to 1 - binom.cdf(k-1, n, p), which is P(X >= k).
    prob1 = stats.binom.sf(k1 - 1, n_values, p1)
    prob2 = stats.binom.sf(k2 - 1, n_values, p2)

    # Find indices where probabilities are close enough (within tolerance) AND are positive.
    matches = np.isclose(prob1, prob2, atol=1e-2) & (prob1 > 0) & (prob2 > 0)

    # Check if any match was found.
    if np.any(matches):
        # np.argmax on a boolean array returns the index of the first True value.
        first_match_idx = np.argmax(matches)
        n_found = n_values[first_match_idx]

        print(f"Found n={n_found}: P(case1)={prob1[first_match_idx]:.4f}, "
              f"P(case2)={prob2[first_match_idx]:.4f}")

        return int(n_found)

    return None  # In case no suitable n is found within the defined range.


### Question 2 ###

def empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=100, seed=None):
    """
    Create k experiments where X is sampled. Calculate the empirical centralized third moment of Y based 
    on your k experiments.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Perform k experiments. The result is a (k, 6) array.
    X_samples = np.random.multinomial(n, p, size=k)

    # 2. Calculate Y for each experiment. Y is the sum of elements at indices 1, 2, 3 (which correspond to X2, X3, X4).
    # The result is a 1D array of length k.
    Y_samples = np.sum(X_samples[:, 1:4], axis=1)

    # 3. Calculate the empirical mean of Y.
    y_mean = np.mean(Y_samples)

    # 4. Calculate the empirical centralized third moment: mean of (value minus mean) cubed.
    empirical_moment = np.mean((Y_samples - y_mean) ** 3)
    return empirical_moment


def class_moment(n=20, p=0.3):
    moment = n * p * (1 - p) * (1 - 2 * p)
    return moment


def plot_moments(num_experiments=1000, k=100):
    moments = np.array([
        empirical_centralized_third_moment()
        for _ in range(num_experiments)
    ])

    true_mu3 = class_moment()
    plt.figure(figsize=(10, 6))
    plt.hist(moments, bins=30, alpha=0.7, edgecolor='black')

    # Step 4: Add vertical red line for theoretical value
    plt.axvline(true_mu3, color='red', linestyle='dashed', linewidth=2,
                label=f'Theoretical: {true_mu3:.4f}')

    plt.title("Distribution of Empirical Centralized Third Moments")
    plt.xlabel("Third Central Moment Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Step 5: Calculate and return variance of this distribution
    dist_var = np.var(moments)
    print(f"Variance of the distribution with k={k}: {dist_var:.6f}")

    return dist_var


def plot_moments_smaller_variance(num_experiments=1000, k=1000):
    """
    By increasing 'k' (e.g., from 100 to 1000), each individual empirical
        moment we calculate becomes more accurate and less noisy.
        This means the distribution of these 1000 moments will be much
        narrower (i.e., it will have a smaller variance) and more tightly
        clustered around the true theoretical value.
    """

    moments = np.array([
        empirical_centralized_third_moment(k=k)
        for _ in range(num_experiments)
    ])

    # Step 2: Get the theoretical third moment from class
    true_mu3 = class_moment()

    # Step 3: Create histogram with 30 bins
    plt.figure(figsize=(10, 6))
    plt.hist(moments, bins=30, alpha=0.7, edgecolor='black')

    # Step 4: Add vertical red line for theoretical value
    plt.axvline(true_mu3, color='red', linestyle='dashed', linewidth=2,
                label=f'Theoretical value: {true_mu3:.4f}')

    plt.title(f"Distribution of Empirical Third Moments (k={k})")
    plt.xlabel("Third Central Moment Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Step 5: Calculate and return the variance of the distribution
    variance_of_moments = np.var(moments)
    print(f"Variance of the distribution with k={k}: {variance_of_moments:.6f}")

    return variance_of_moments


### Question 3 ###

def NFoldConv(P=np.array([[0, 1], [0.5, 0.5]]), n=2):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables,
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """

    if isinstance(P, dict):
        values = np.array(list(P.keys()))
        probs = np.array(list(P.values()))
    elif isinstance(P, (list, tuple)) and len(P) == 2:
        values, probs = map(np.asarray, P)
    else:
        P = np.asarray(P)
        values, probs = P[0], P[1]


    if n == 1:
        return P.copy()

    result_values = values.copy()
    result_probs = probs.copy()

    for _ in range(n - 1):
        # Create meshgrid of all combinations
        v1_grid, v2_grid = np.meshgrid(result_values, values)
        p1_grid, p2_grid = np.meshgrid(result_probs, probs)

        # Calculate all possible sums and probabilities
        all_sums = (v1_grid + v2_grid).flatten()
        all_probs = (p1_grid * p2_grid).flatten()

        # Group by sum value and add probabilities
        unique_vals = np.unique(all_sums)
        new_probs = np.array([
            all_probs[all_sums == val].sum()
            for val in unique_vals
        ])

        result_values = unique_vals
        result_probs = new_probs

    Q = np.array([result_values, result_probs])

    return Q


def plot_dist(P=np.array([[0, 1], [0.5, 0.5]])):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """
    Q = NFoldConv(P)
    values = Q[0]
    probs = Q[1]

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(values, probs, width=0.6, color='skyblue', edgecolor='black')
    plt.xlabel("Values")
    plt.ylabel("Probability")
    plt.title("Distribution P")
    plt.xticks(values)  # Show integer x-ticks for clarity
    plt.show()


### Qeustion 4 ###

def evenBinom(n, p):
    r"""
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    # We will calculate P(X is even) = P(X=0) + P(X=2) + ... + P(X=k) where k is the largest even number <= n
    even_arr = np.arange(0, n + 1, 2) # even numbers from 0 to n
    prob = np.sum(stats.binom.pmf(even_arr, n, p)) # sum of probabilities for even outcomes

    return prob


def evenBinomFormula(n, p):
    r"""
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    # Lets look at the definition of binomial distribution:
    # from the binomial theorem we know that:
    # (q + p)^n = sum_{k=0}^{n} C(n, k) * p^k * q^(n-k)
    # where q = 1 - p (the probability of failure)
    # (p + (1 - p))^n = 1^n = 1
    # So we have:
    # 1 = S_even + S_odd

    # We want to find an expression for S_even to Solve the equation above.:
    # S_even - S_odd = ?
    # We can make S_odd negative if we will choose p with -p:
    # It will give us (q - p)^n = sum_{k=0}^{n} C(n, k) * (-p)^k * q^(n-k) -> when k is odd we get negative sign.

    #Now we have:
    # 1 = S_even + S_odd
    # (1 - 2p)^n = S_even - S_odd

    # We can solve the system of equations:
    # 2*S_even = 1 + (1 - 2p)^n => S_even = (1 + (1 - 2p)^n) / 2

    # Therefore, the probability that X is even is:
    # P(X is even) = S_even = (1 + (1 - 2p)^n) / 2

    print("=== Proof for P(X is even) ===")
    print("From the Binomial theorem:")
    print("  (q + p)^n = sum_{k=0}^{n} C(n, k) * p^k * q^(n-k)")
    print("Let q = 1 - p -> (p + (1 - p))^n = 1 = S_even + S_odd")
    print("Now consider (q - p)^n = sum_{k=0}^{n} C(n, k) * (-p)^k * q^(n-k)")
    print("-> When k is odd, the term becomes negative, so:")
    print("  (1 - 2p)^n = S_even - S_odd")
    print("We now have two equations:")
    print("  1 = S_even + S_odd")
    print("  (1 - 2p)^n = S_even - S_odd")
    print("Solving these gives:")
    print("  2*S_even = 1 + (1 - 2p)^n")
    print("Therefore:")
    print("  P(X is even) = S_even = (1 + (1 - 2p)^n) / 2\n")

    prob = (1 + (1 - 2 * p) ** n) / 2
    print(f"For n={n}, p={p}: P(X is even) = {prob:.6f}")
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

    r"""
    We are looking for  Var(X + Y + Z):
    Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z) + 2Cov(X,Y) + 2Cov(X,Z) + 2Cov(Y,Z)
    We will calculate Var(X), Var(Y), Var(Z), Cov(X,Y), Cov(X,Z), Cov(Y,Z) from the joint distribution
    Var(x) = E[X^2] - (E[X])^2
    E[X] = sum_x x * P(X=x)
    E[XY] = sum_{x,y} x*y * P(X=x, Y=y)
    Formula for Cov(X,Y) = E[XY] - E[X]E[Y]


    Input Data will look like this table:
        | X | Y | Z | P(X,Y,Z) |
        |---|---|---|----------|
        | x1| y1| z1|   p1     |
        | x1| y1| z2|   p2     |
        | x1| y2| z1|   p3     |
        |...|...|...|   ...    |
    """
    # חילוץ X
    X = np.vectorize(lambda t: t[0])(values)

    # חילוץ Y
    Y = np.vectorize(lambda t: t[1])(values)

    # חילוץ Z
    Z = np.vectorize(lambda t: t[2])(values)
    # calculate expectations and variances
    E_X = np.sum(X * joint_probs)
    E_Y = np.sum(Y * joint_probs)
    E_Z = np.sum(Z * joint_probs)
    E_X2 = np.sum(X**2 * joint_probs)
    E_Y2 = np.sum(Y**2 * joint_probs)
    E_Z2 = np.sum(Z**2 * joint_probs)
    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Var_Z = E_Z2 - E_Z**2
    E_XY = np.sum(X * Y * joint_probs)
    E_XZ = np.sum(X * Z * joint_probs)
    E_YZ = np.sum(Y * Z * joint_probs)
    Cov_XY = E_XY - E_X * E_Y
    Cov_XZ = E_XZ - E_X * E_Z
    Cov_YZ = E_YZ - E_Y * E_Z

    v = Var_X + Var_Y + Var_Z + 2 * Cov_XY + 2 * Cov_XZ + 2 * Cov_YZ

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
    # We are looking for  Var(X + Y + Z):
    # Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z) + 2Cov(X,Y) + 2Cov(X,Z) + 2Cov(Y,Z)
    # We will calculate Var(X), Var(Y), Var(Z), Cov(X,Y), Cov(X,Z), Cov(Y,Z) from the joint distribution
    # Var(x) = E[X^2] - (E[X])^2
    # E[X] = sum_x x * P(X=x)
    # E[XY] = sum_{x,y} x*y * P(X=x, Y=y)
    # Formula for Cov(X,Y) = E[XY] - E[X]E[Y]
    # Because X, Y, Z are pairwise independent:
    # Cov(X,Y) = 0, Cov(X,Z) = 0, Cov(Y,Z) = 0
    r"""
    Input Data will look like this table:
        | X | Y | Z | P(X,Y,Z) |
        |---|---|---|----------|
        | x1| y1| z1|   p1     |
        | x1| y1| z2|   p2     |
        | x1| y2| z1|   p3     |
        |...|...|...|   ...    |
    """
    # חילוץ X
    X = np.vectorize(lambda t: t[0])(values)

    # חילוץ Y
    Y = np.vectorize(lambda t: t[1])(values)

    # חילוץ Z
    Z = np.vectorize(lambda t: t[2])(values)

    E_X = np.sum(X * joint_probs)
    E_Y = np.sum(Y * joint_probs)
    E_Z = np.sum(Z * joint_probs)
    E_X2 = np.sum(X**2 * joint_probs)
    E_Y2 = np.sum(Y**2 * joint_probs)
    E_Z2 = np.sum(Z**2 * joint_probs)
    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Var_Z = E_Z2 - E_Z**2
    E_XY = np.sum(X * Y * joint_probs)
    E_XZ = np.sum(X * Z * joint_probs)
    E_YZ = np.sum(Y * Z * joint_probs)
    ### Because pairwise independent: -> Cov(X,Y) = 0, Cov(X,Z) = 0, Cov(Y,Z) = 0
    # Cov_XY = E_XY - E_X * E_Y
    # Cov_XZ = E_XZ - E_X * E_Z
    # Cov_YZ = E_YZ - E_Y * E_Z

    v = Var_X + Var_Y + Var_Z

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
    r"""
    Although X, Y, Z are pairwise independent, they are not collectively independent if:
    P(X=x, Y=y, Z=z) != P(X=x) * P(Y=y) * P(Z=z)
    As we have seen in the lecture every two RVs are independent but the three together are not,
    and the famous example is the following:
    X, Y are independent coin flips, and Z = X XOR Y.
    Explain:
    From having X and Y we can determine Z with probability 1, and thus they are not collectively independent,
    but knowing only X or only Y gives no information about Z.
    
    Probability Table:
        | X | Y | Z | P(X,Y,Z) |
        |---|---|---|----------|
        | 0 | 0 | 0 |   1/4    |
        | 0 | 1 | 1 |   1/4    |
        | 1 | 0 | 1 |   1/4    |
        | 1 | 1 | 0 |   1/4    |
        
    
    """

    return False


### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    n = number of trials
    p = probability of success in each trial
    For example,
    if n = 3:
    We will have number of value 1 can be placed in 3 positions:
    C(1): 001, 010, 100 = 3 There are 3 combinations to get value 1.
    number of value 2 can be placed in 3 positions:
    value 2:
    C(2): 011, 101, 110 = 3
    value 3:
    C(3): 111 = 1
    The distribution of C will be:
    k | C(k)
    ---------
    0 | 1
    1 | 3
    2 | 3
    3 | 1
    """
    E = 0

    for k in range(0, n + 1):
        C_k = comb(n, k)
        P_k = stats.binom.pmf(k, n, p)
        E += C_k * P_k

    return float(E)