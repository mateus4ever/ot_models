import math
from scipy.stats import norm

class bivariatenormal:
    def __init__(self):
        pass

    # Bivariate normal CDF with correlation rho
    def binormcdf(self, a, b, rho):
        pi = math.pi

        if a <= 0.0 and b <= 0.0 and rho <= 0.0:
            aprime = a / math.sqrt(2.0 * (1.0 - rho * rho))
            bprime = b / math.sqrt(2.0 * (1.0 - rho * rho))
            A = [0.3253030, 0.4211071, 0.1334425, 0.006374323]
            B = [0.1337764, 0.6243247, 1.3425378, 2.2626645]
            sum_result = 0.0
            for i in range(4):
                for j in range(4):
                    sum_result += A[i] * A[j] * self.f(B[i], B[j], aprime, bprime, rho)
            sum_result *= math.sqrt(1.0 - rho * rho) / pi
            return sum_result

        elif a * b * rho <= 0.0:
            if a <= 0.0 and b >= 0.0 and rho >= 0.0:
                return norm.cdf(a) - self.binormcdf(a, -b, -rho)
            elif a >= 0.0 and b <= 0.0 and rho >= 0.0:
                return norm.cdf(b) - self.binormcdf(-a, b, -rho)
            elif a >= 0.0 and b >= 0.0 and rho <= 0.0:
                return norm.cdf(a) + norm.cdf(b) - 1.0 + self.binormcdf(-a, -b, rho)

        elif a * b * rho >= 0.0:
            denum = math.sqrt(a * a - 2.0 * rho * a * b + b * b)
            rho1 = ((rho * a - b) * self.sign(a)) / denum
            rho2 = ((rho * b - a) * self.sign(b)) / denum
            delta = (1.0 - self.sign(a) * self.sign(b)) / 4.0
            return (
                self.binormcdf(a, 0.0, rho1) +
                self.binormcdf(b, 0.0, rho2) -
                delta
            )

    # Sign function
    def sign(self, a):
        if a < 0:
            return -1.0
        elif a > 0:
            return 1.0
        else:
            return 0.0

    # Support function
    def f(self, x, y, aprime, bprime, rho):
        r = (
            aprime * (2.0 * x - aprime) +
            bprime * (2.0 * y - bprime) +
            2.0 * rho * (x - aprime) * (y - bprime)
        )
        return math.exp(r)

