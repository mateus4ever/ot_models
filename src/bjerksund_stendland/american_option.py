# File Contains: Python code containing closed-form solutions for the valuation of European Options,
# American Options, Asian Options, Spread Options, Heat Rate Options, and Implied Volatility
#
# This document demonstrates a Python implementation of some option models described in books written by Davis
# Edwards: "Energy Trading and Investing", "Risk Management in Trading", "Energy Investing Demystified".
#
# for backward compatability with Python 2.7
from __future__ import division

# import necessary libaries
import unittest
import math
import numpy as np
from scipy.stats import norm
from scipy.stats import mvn

# -----------
# Generalized American Option Pricer
# This is a wrapper to check inputs and route to the current "best" American option model
def _american_option(option_type, fs, x, t, r, b, v):

    # -----------
    if option_type == "c":
        # Call Option
        return _bjerksund_stensland_2002(fs, x, t, r, b, v)
    else:
        # Put Option

        # Using the put-call transformation: P(X, FS, T, r, b, V) = C(FS, X, T, -b, r-b, V)
        # WARNING - When reconciling this code back to the B&S paper, the order of variables is different

        put__x = fs
        put_fs = x
        put_b = -b
        put_r = r - b

        # pass updated values into the Call Valuation formula
        return _bjerksund_stensland_2002(put_fs, put__x, t, put_r, put_b, v)


# The primary class for calculating Generalized Black Scholes option prices and deltas
# It is not intended to be part of this module's public interface

# Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
#         b = cost of carry, v = implied volatility
# Outputs: value, delta, gamma, theta, vega, rho
def _gbs(option_type, fs, x, t, r, b, v):

    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        # it's a call
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)

    return value, delta, gamma, theta, vega, rho


# -----------
# American Call Option (Bjerksund Stensland 2002 approximation)
def _bjerksund_stensland_2002(fs, x, t, r, b, v):
    # -----------
    # initialize output
    # using GBS greeks (TO DO: update greek calculations)
    my_output = _gbs("c", fs, x, t, r, b, v)

    e_value = my_output[0]
    delta = my_output[1]
    gamma = my_output[2]
    theta = my_output[3]
    vega = my_output[4]
    rho = my_output[5]

    # If b >= r, it is never optimal to exercise before maturity
    # so we can return the GBS value
    if b >= r:
        return e_value, delta, gamma, theta, vega, rho

    # -----------
    # Create preliminary calculations
    v2 = v ** 2
    t1 = 0.5 * (math.sqrt(5) - 1) * t
    t2 = t

    beta_inside = ((b / v2 - 0.5) ** 2) + 2 * r / v2
    # forcing the inside of the sqrt to be a positive number
    beta_inside = abs(beta_inside)
    beta = (0.5 - b / v2) + math.sqrt(beta_inside)
    b_infinity = (beta / (beta - 1)) * x
    b_zero = max(x, (r / (r - b)) * x)

    h1 = -(b * t1 + 2 * v * math.sqrt(t1)) * ((x ** 2) / ((b_infinity - b_zero) * b_zero))
    h2 = -(b * t2 + 2 * v * math.sqrt(t2)) * ((x ** 2) / ((b_infinity - b_zero) * b_zero))

    i1 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h1))
    i2 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h2))

    alpha1 = (i1 - x) * (i1 ** (-beta))
    alpha2 = (i2 - x) * (i2 ** (-beta))

    # check for immediate exercise
    if fs >= i2:
        value = fs - x
    else:
        # Perform the main calculation
        value = (alpha2 * (fs ** beta)
                 - alpha2 * _phi(fs, t1, beta, i2, i2, r, b, v)
                 + _phi(fs, t1, 1, i2, i2, r, b, v)
                 - _phi(fs, t1, 1, i1, i2, r, b, v)
                 - x * _phi(fs, t1, 0, i2, i2, r, b, v)
                 + x * _phi(fs, t1, 0, i1, i2, r, b, v)
                 + alpha1 * _phi(fs, t1, beta, i1, i2, r, b, v)
                 - alpha1 * _psi(fs, t2, beta, i1, i2, i1, t1, r, b, v)
                 + _psi(fs, t2, 1, i1, i2, i1, t1, r, b, v)
                 - _psi(fs, t2, 1, x, i2, i1, t1, r, b, v)
                 - x * _psi(fs, t2, 0, i1, i2, i1, t1, r, b, v)
                 + x * _psi(fs, t2, 0, x, i2, i1, t1, r, b, v))

    # in boundary conditions, this approximation can break down
    # Make sure option value is greater than or equal to European value
    value = max(value, e_value)

    # -----------
    # Return Data
    return value, delta, gamma, theta, vega, rho

# ---------------------------
# American Option Intermediate Calculations

# -----------
# The Psi() function used by _Bjerksund_Stensland_2002 model
def _psi(fs, t2, gamma, h, i2, i1, t1, r, b, v):
    vsqrt_t1 = v * math.sqrt(t1)
    vsqrt_t2 = v * math.sqrt(t2)

    bgamma_t1 = (b + (gamma - 0.5) * (v ** 2)) * t1
    bgamma_t2 = (b + (gamma - 0.5) * (v ** 2)) * t2

    d1 = (math.log(fs / i1) + bgamma_t1) / vsqrt_t1
    d3 = (math.log(fs / i1) - bgamma_t1) / vsqrt_t1

    d2 = (math.log((i2 ** 2) / (fs * i1)) + bgamma_t1) / vsqrt_t1
    d4 = (math.log((i2 ** 2) / (fs * i1)) - bgamma_t1) / vsqrt_t1

    e1 = (math.log(fs / h) + bgamma_t2) / vsqrt_t2
    e2 = (math.log((i2 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e3 = (math.log((i1 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e4 = (math.log((fs * (i1 ** 2)) / (h * (i2 ** 2))) + bgamma_t2) / vsqrt_t2

    tau = math.sqrt(t1 / t2)
    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    psi = math.exp(lambda1 * t2) * (fs ** gamma) * (_cbnd(-d1, -e1, tau)
                                                    - ((i2 / fs) ** kappa) * _cbnd(-d2, -e2, tau)
                                                    - ((i1 / fs) ** kappa) * _cbnd(-d3, -e3, -tau)
                                                    + ((i1 / i2) ** kappa) * _cbnd(-d4, -e4, -tau))
    return psi

# -----------
# The Phi() function used by _Bjerksund_Stensland_2002 model and the _Bjerksund_Stensland_1993 model
def _phi(fs, t, gamma, h, i, r, b, v):
    d1 = -(math.log(fs / h) + (b + (gamma - 0.5) * (v ** 2)) * t) / (v * math.sqrt(t))
    d2 = d1 - 2 * math.log(i / fs) / (v * math.sqrt(t))

    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    phi = math.exp(lambda1 * t) * (fs ** gamma) * (norm.cdf(d1) - ((i / fs) ** kappa) * norm.cdf(d2))

    return phi

# -----------
# Cumulative Bivariate Normal Distribution
# Primarily called by Psi() function, part of the _Bjerksund_Stensland_2002 model
def _cbnd(a, b, rho):
    # This distribution uses the Genz multi-variate normal distribution
    # code found as part of the standard SciPy distribution
    lower = np.array([0, 0])
    upper = np.array([a, b])
    infin = np.array([0, 0])
    correl = rho
    error, value, inform = mvn.mvndst(lower, upper, infin, correl)
    return value

# -----------
# Generalized American Option Pricer
# This is a wrapper to check inputs and route to the current "best" American option model
def _american_option(option_type, fs, x, t, r, b, v):

    # -----------
    if option_type == "c":
        # Call Option
        return _bjerksund_stensland_2002(fs, x, t, r, b, v)
    else:

        # Using the put-call transformation: P(X, FS, T, r, b, V) = C(FS, X, T, -b, r-b, V)
        # WARNING - When reconciling this code back to the B&S paper, the order of variables is different

        put__x = fs
        put_fs = x
        put_b = -b
        put_r = r - b

        # pass updated values into the Call Valuation formula
        return _bjerksund_stensland_2002(put_fs, put__x, t, put_r, put_b, v)

# This function tests that two floating point numbers are the same
# Numbers less than 1 million are considered the same if they are within .000001 of each other
# Numbers larger than 1 million are considered the same if they are within .0001% of each other
# User can override the default precision if necessary
def assert_close(value_a, value_b, precision=.000001):
    my_precision = precision

    if (value_a < 1000000.0) and (value_b < 1000000.0):
        my_diff = abs(value_a - value_b)
        my_diff_type = "Difference"
    else:
        my_diff = abs((value_a - value_b) / value_a)
        my_diff_type = "Percent Difference"

    if my_diff < my_precision:
        my_result = True
    else:
        my_result = False

    if (__name__ is "__main__") and (my_result is False):
        print("  FAILED TEST. Comparing {0} and {1}. Difference is {2}, Difference Type is {3}".format(value_a, value_b,
                                                                                                       my_diff,
                                                                                                       my_diff_type))
    else:
        print(".")

    return my_result


print("=====================================")
print("American Options Testing")
print("=====================================")

print("testing _Bjerksund_Stensland_2002()")
# _american_option(option_type, X, FS, T, b, r, V)
assert_close(_bjerksund_stensland_2002(fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 0.8099, precision=.001)
assert_close(_bjerksund_stensland_2002(fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.7661, precision=.001)
assert_close(_bjerksund_stensland_2002(fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 15.5137, precision=.001)

assert_close(_bjerksund_stensland_2002(fs=100, x=90, t=0.5, r=.1, b=0, v=0.15)[0], 10.5400, precision=.001)
assert_close(_bjerksund_stensland_2002(fs=100, x=100, t=0.5, r=.1, b=0, v=0.25)[0], 6.7661, precision=.001)
assert_close(_bjerksund_stensland_2002(fs=100, x=110, t=0.5, r=.1, b=0, v=0.35)[0], 5.8374, precision=.001)

# print("testing _Bjerksund_Stensland_1993()")
# # Prices for 1993 model slightly different than those presented in Haug's Complete Guide to Option Pricing Formulas
# # Possibly due to those results being based on older CBND calculation?
# assert_close(_bjerksund_stensland_1993(fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 0.8089, precision=.001)
# assert_close(_bjerksund_stensland_1993(fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.757, precision=.001)
# assert_close(_bjerksund_stensland_1993(fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 15.4998, precision=.001)

print("testing _american_option()")
assert_close(_american_option("p", fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 10.5400, precision=.001)
assert_close(_american_option("p", fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.7661, precision=.001)
assert_close(_american_option("p", fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 5.8374, precision=.001)

assert_close(_american_option('c', fs=100, x=95, t=0.00273972602739726, r=0.000751040922831883, b=0, v=0.2)[0], 5.0,
             precision=.01)
assert_close(_american_option('c', fs=42, x=40, t=0.75, r=0.04, b=-0.04, v=0.35)[0], 5.28, precision=.01)
assert_close(_american_option('c', fs=90, x=100, t=0.1, r=0.10, b=0, v=0.15)[0], 0.02, precision=.01)

print("Testing that American valuation works for integer inputs")
assert_close(_american_option('c', fs=100, x=100, t=1, r=0, b=0, v=0.35)[0], 13.892, precision=.001)
assert_close(_american_option('p', fs=100, x=100, t=1, r=0, b=0, v=0.35)[0], 13.892, precision=.001)

print("Testing valuation works at minimum/maximum values for T")
assert_close(_american_option('c', 100, 100, 0.00396825396825397, 0.000771332656950173, 0, 0.15)[0], 0.3769,
             precision=.001)
assert_close(_american_option('p', 100, 100, 0.00396825396825397, 0.000771332656950173, 0, 0.15)[0], 0.3769,
             precision=.001)
assert_close(_american_option('c', 100, 100, 100, 0.042033868311581, 0, 0.15)[0], 18.61206, precision=.001)
assert_close(_american_option('p', 100, 100, 100, 0.042033868311581, 0, 0.15)[0], 18.61206, precision=.001)

print("Testing valuation works at minimum/maximum values for X")
assert_close(_american_option('c', 100, 0.01, 1, 0.00330252458693489, 0, 0.15)[0], 99.99, precision=.001)
assert_close(_american_option('p', 100, 0.01, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)
assert_close(_american_option('c', 100, 2147483248, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)
assert_close(_american_option('p', 100, 2147483248, 1, 0.00330252458693489, 0, 0.15)[0], 2147483148, precision=.001)

print("Testing valuation works at minimum/maximum values for F/S")
assert_close(_american_option('c', 0.01, 100, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)
assert_close(_american_option('p', 0.01, 100, 1, 0.00330252458693489, 0, 0.15)[0], 99.99, precision=.001)
assert_close(_american_option('c', 2147483248, 100, 1, 0.00330252458693489, 0, 0.15)[0], 2147483148, precision=.001)
assert_close(_american_option('p', 2147483248, 100, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)

print("Testing valuation works at minimum/maximum values for b")
assert_close(_american_option('c', 100, 100, 1, 0, -1, 0.15)[0], 0.0, precision=.001)
assert_close(_american_option('p', 100, 100, 1, 0, -1, 0.15)[0], 63.2121, precision=.001)
assert_close(_american_option('c', 100, 100, 1, 0, 1, 0.15)[0], 171.8282, precision=.001)
assert_close(_american_option('p', 100, 100, 1, 0, 1, 0.15)[0], 0.0, precision=.001)

print("Testing valuation works at minimum/maximum values for r")
assert_close(_american_option('c', 100, 100, 1, -1, 0, 0.15)[0], 16.25133, precision=.001)
assert_close(_american_option('p', 100, 100, 1, -1, 0, 0.15)[0], 16.25133, precision=.001)
assert_close(_american_option('c', 100, 100, 1, 1, 0, 0.15)[0], 3.6014, precision=.001)
assert_close(_american_option('p', 100, 100, 1, 1, 0, 0.15)[0], 3.6014, precision=.001)

print("Testing valuation works at minimum/maximum values for V")
assert_close(_american_option('c', 100, 100, 1, 0.05, 0, 0.005)[0], 0.1916, precision=.001)
assert_close(_american_option('p', 100, 100, 1, 0.05, 0, 0.005)[0], 0.1916, precision=.001)
assert_close(_american_option('c', 100, 100, 1, 0.05, 0, 1)[0], 36.4860, precision=.001)
assert_close(_american_option('p', 100, 100, 1, 0.05, 0, 1)[0], 36.4860, precision=.001)

