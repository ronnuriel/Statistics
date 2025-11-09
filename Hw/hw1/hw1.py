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

def find_sample_size_binom():
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    pass

def find_sample_size_nbinom():
    """
    Using NBinom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    pass

def compare_q1():
    pass

def same_prob():
    pass


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











    
    
    
    
    
    