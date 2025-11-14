###### Your ID ######
# ID1: 
# ID2: 
#####################

# imports 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 1 ###

def find_sample_size_binom(defective_rate=0.03, target_prob=0.85):
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    # P(at least 1 defective) = 1 - P(0 defective) >= target_prob
    # 1 - (1 - defective_rate)^n >= target_prob
    # (1 - defective_rate)^n <= 1 - target_prob
    # n * log(1 - defective_rate) <= log(1 - target_prob)
    # n >= log(1 - target_prob) / log(1 - defective_rate)  (Dividing by negative flips inequality)
    
    n = np.ceil(np.log(1 - target_prob) / np.log(1 - defective_rate))
    print(f"To have a {target_prob:.0%} probability of at least one defective product, "
          f"we need to ask for {n} independent samples.")
    return int(n)

def find_sample_size_nbinom(defective_rate=0.03, target_prob=0.85, x=1):
    """
    Using NBinom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    from scipy.stats import nbinom
    # nbinom.ppf(q, n, p) returns the number of failures (k) that gives cumulative probability q.

    # Here 'n' in standard scipy notation is our number of successes 'x'.

    # Total samples = (number of failures) + (number of successes)


    k_failures = nbinom.ppf(target_prob, x, defective_rate)

    n_total = k_failures + x

    print(f"To have a {target_prob:.0%} probability of at least {x} defective product(s), "
            f"we need to ask for {n_total} independent samples.")
    
    return int(n_total)



def compare_q1():
    # Case 1: 10% defective, need at least 5, 90% confidence
    case1 = find_sample_size_nbinom(defective_rate=0.10, target_prob=0.90, x=5)
    
    # Case 2: 30% defective, need at least 15, 90% confidence
    case2 = find_sample_size_nbinom(defective_rate=0.30, target_prob=0.90, x=15)

    result = (case1, case2)

    print(f"Required samples for Case 1 and Case 2: {result}")

    return (case1, case2)

def same_prob(p1=0.1, k1=5, p2=0.3, k2=15):
    from scipy.stats import binom
    # Define a reasonable search range for n (e.g., from k2 up to 2000).
    # Start from k2 because we need at least k2 samples to have k2 successes.
    n_values = np.arange(k2, 2001)
    
    # Vectorized calculation of P(X >= k) for all n values at once.
    # binom.sf(k-1, n, p) is equivalent to 1 - binom.cdf(k-1, n, p), which is P(X >= k).
    prob1 = binom.sf(k1 - 1, n_values, p1)
    prob2 = binom.sf(k2 - 1, n_values, p2)
    
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
    empirical_moment = np.mean((Y_samples - y_mean)**3)
    return empirical_moment

def class_moment(n=20, p=0.3):    
    moment = n * p * (1 - p) * (1 - 2 * p)
    return moment

def plot_moments(num_experiments=1000,k=100):

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

def NFoldConv(P = np.array([[0, 1], [0.5, 0.5]]) , n = 2):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables, 
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """
    values = P[0].astype(int)
    probs = P[1]
    
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
    
def plot_dist(P = np.array([[0, 1], [0.5, 0.5]])):
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



    
    
    
    
