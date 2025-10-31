from scipy.stats import binom

"""This is a binomial experiment situation…
o There are n = 40 patients and we are counting the number of patients
that survive 5 or more years.
o The individual patient outcomes are independent and under the NULL
MODEL the probability of success is p = 0.2 for all patients.
(that is: we assume that Tx is NOT better than the standard of care)
• So the random variable X = # of “successes” in the clinical trial
is, under the NULL model, Binomial with n = 40 and p = 0.2,
i.e., under the null: 𝑋 ~ 𝐵𝑖𝑛𝑜𝑚𝑖𝑎𝑙(40,0.2)"""

print("Binomial Distribution: n=40, p=0.2")
rv = binom(n=40, p=0.2)

############# Question 1 #############
print("Probability that exactly 16 patients survive at least 5 years:")
print(rv.pmf(16))
############# Question 2 #############
print("Probability that 16 or more patients survive at least 5 years:")
x_16_or_more = sum(rv.pmf(k) for k in range(16, 41))
print(x_16_or_more)
# Another way to calculate it:
print("Alternative calculation using CDF:")
print(1 - rv.cdf(15))
# cdf gives P(X <= k), so we use 1 - P(X <= 15) to get P(X >= 16)

