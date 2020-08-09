"""Simulation of non-normal distribution using Fleishman's Power Method.
   We derive coefficients using modified Powell method.
   
   Sources:
   Implementation in SAS. https://support.sas.com/content/dam/SAS/support/en/books/simulating-data-with-sas/65378_Appendix_D_Functions_for_Simulating_Data_by_Using_Fleishmans_Transformation.pdf
   Hao Luo , Generation of Non-normal Data – A Study of Fleishman’s Power Method. https://www.diva-portal.org/smash/get/diva2:407995/FULLTEXT01.pdf
   Tables from Fleishman program and SAS guide. https://chandrakant721.wordpress.com/fleishmans-power-method-coefficient-table/
"""

import numpy as np
from scipy.stats import moment
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def fleishman(x, *data):
    """Fleishman's function to solve."""
    b = x[0]
    c = x[1]
    d = x[2]
    
    target_skew, target_kurt = data

    f1 = b**2 + 6*b*d + 2*c**2 + 15*d**2 - 1
    f2 = 2*c*(b**2 + 24*b*d + 105*d**2 + 2) - target_skew
    f3 = 24 *(b*d + c**2*( 1 + b**2 + 28*b*d) + 
              d**2*( 12 + 48*b*d + 141*c**2 + 225*d**2)) - target_kurt

    return (f1, f2, f3)

def fleishman_jac(x, *data):
    """Jacobian of Fleishman func from SAS implementation."""
    b,c,d = x
    
    df1db = 2*b + 6*d
    df1dc = 4*c
    df1dd = 6*b + 30*d
    df2db = 4*c * (b + 12*d)
    df2dc = 2 * (b**2 + 24*b*d + 105*d**2 + 2)
    df2dd = 4 * c * (12*b + 105*d)
    df3db = 24 * (d + c**2 * (2*b + 28*d) + 48 * d**3)
    df3dc = 48 * c * (1 + b**2 + 28*b*d + 141*d**2)
    df3dd = 24 * (b + 28*b * c**2 + 2 * d * (12 + 48*b*d + 
                  141*c**2 + 225*d**2) + d**2 * (48*b + 450*d))
    return np.matrix([[df1db, df1dc, df1dd],
                      [df2db, df2dc, df2dd],
                      [df3db, df3dc, df3dd]])

def initial_guess(target_skew, target_kurt):
    """Initial guess for Fleishman func from SAS implementation."""
    c1 = 0.95357 - 0.05679 * target_kurt + 0.03520 * target_skew**2 + 0.00133 * target_kurt**2
    c2 = 0.10007 * target_skew + 0.00844 * target_skew**3
    c3 = 0.30978 - 0.31655 * c1
    return (c1, c2, c3)

def f_moments(data):
    """Find first 4 moments of data."""
    mean = np.mean(data)
    var = moment(data,2)
    
    # From Cramér (1957)
    skew = moment(data,3)/var**1.5
    # Also called "relative" kurtosis
    kurt = moment(data,4)/var**2 - 3
    
    return (mean,var,skew,kurt)

def check_target_vals(skew,kurt):
    """Check if kurtosis, skew is valid."""
    if (kurt < -1.2264489 + 1.6410373 * skew ** 2):
        raise Exception("Combination of kurtosis, skew not allowed.")
        
def plot_compare(dist_x, label_x, dist_y, label_y):
    """Generate plot of target, Fleishman data."""
    bins = np.linspace(-2, 5, 100)
    plt.hist(dist_x, bins, alpha=0.5, label=label_x)
    plt.hist(dist_y, bins, alpha=0.5, label=label_y)
    plt.legend(loc='upper right')
    plt.show()
    
    
def generate_Fleishman(target, n):
    """Given target moments, return transformed normally distributed data."""
    target_mean, target_var, target_skew, target_kurt = target
    
    # Check if Fleishman's method is appropriate
    check_target_vals(target_skew, target_kurt)

    # Solve for coefficients of transformation
    guess = initial_guess(target_skew, target_kurt)
    root = fsolve(fleishman, guess, args=(target_skew, target_kurt), fprime = fleishman_jac)
    
    # Generate normal distribution
    mu, sigma = 0, 1 # mean and standard deviation
    s_x = np.random.normal(mu, sigma, n)

    # Derive Y using coefficients
    a = -1 * root[1]
    b,c,d = root
    s_y = a + b*s_x + c*s_x**2 + d*s_x**3

    # Get required distribution Z using target mean, variance
    s_z = target_mean + (target_var**.5 * s_y)
    return s_z


# Ex. 1: Exponential
n = 10000
s_target = np.random.exponential(.5, n)
target = f_moments(s_target)
print(target)
s_z = generate_Fleishman(target, n)
print(f_moments(s_z))
plot_compare(s_target, "Exponential X", s_z, "Fleishman Z")


# Ex. 2: Uniform
n = 1000000
s_target = np.random.uniform(-1,3,n)
target = f_moments(s_target)
print("Target", target)
s_z = generate_Fleishman(target, n)
print("Fleishman", f_moments(s_z))
plot_compare(s_target, "Uniform X", s_z, "Fleishman Z")
