import math

from src.experiments.archive.bjerksund_stendland.cb.bivariate_normal import bivariatenormal


class blackscholes:
    def __init__(self):
        pass

    # Waissi and Rossin normal CDF approximation
    def normcdf(self, z):
        if z <= -7.0:
            return 0.0
        elif z >= 7.0:
            return 1.0
        else:
            pi = math.pi
            b1 = -0.0004406
            b2 = 0.0418198
            b3 = 0.9
            return 1.0 / (1.0 + math.exp(-math.sqrt(pi) * (b1 * z**5 + b2 * z**3 + b3 * z)))

    # Black-Scholes call price using drift (b = r - q)
    def BSCall(self, S, K, r, b, v, T):
        d1 = (math.log(S / K) + (b + v * v / 2.0) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        return S * math.exp((b - r) * T) * self.normcdf(d1) - K * math.exp(-r * T) * self.normcdf(d2)

    # Black-Scholes call or put price
    def BSPrice(self, S, K, r, q, v, T, PutCall):
        d1 = (math.log(S / K) + (r - q + v * v / 2.0) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        BSCall = S * math.exp(-q * T) * self.normcdf(d1) - K * math.exp(-r * T) * self.normcdf(d2)
        if PutCall == 'C':
            return BSCall
        else:
            return BSCall - S * math.exp(-q * T) + K * math.exp(-r * T)

# Example Usage
bvn = bivariatenormal()
a = -0.5
b = -0.5
rho = -0.2
result = bvn.binormcdf(a, b, rho)
print(f"Bivariate Normal CDF: {result}")

bs = blackscholes()
S = 100  # Stock price
K = 95   # Strike price
r = 0.05 # Risk-free rate
q = 0.02 # Dividend yield
v = 0.2  # Volatility
T = 1    # Time to maturity
PutCall = 'C'  # Call option
price = bs.BSPrice(S, K, r, q, v, T, PutCall)
print(f"Black-Scholes Price: {price}")
