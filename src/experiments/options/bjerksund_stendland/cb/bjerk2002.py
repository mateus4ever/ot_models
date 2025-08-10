import math

from src.experiments.options.bjerksund_stendland.cb.bivariate_normal import bivariatenormal
from src.experiments.options.bjerksund_stendland.cb.blackscholes import blackscholes

class bjerk2002:
    def __init__(self):
        self.bs = blackscholes()
        self.bn = bivariatenormal()

    def bjerk_price_2002(self, S, K, T, r, b, sigma, put_call):
        if put_call == 'C':
            return self.call_price_2002(S, K, T, r, b, sigma)
        elif put_call == 'P':
            return self.call_price_2002(K, S, T, r-b, -b, sigma)
        else:
            return 0.0

    def call_price_2002(self, S, K, T, r, b, sigma):
        if b >= r:
            return self.bs.bscall(S, K, r, b, sigma, T)
        else:
            beta = (0.5 - b/sigma**2) + math.sqrt((b/sigma**2 - 0.5)**2 + 2*r/sigma**2)
            B_inf = K * beta / (beta - 1.0)
            B0 = max(K, K*r/(r-b))
            hT = -(b*T + 2*sigma*math.sqrt(T)) * K**2 / B0 / (B_inf - B0)
            X = B0 + (B_inf - B0) * (1.0 - math.exp(hT))

            if S >= X:
                return S - K
            else:
                alpha_X = (X - K) * X**(-beta)
                t = 0.5 * (math.sqrt(5.0) - 1.0) * T
                hTt = -(b*(T-t) + 2*sigma*math.sqrt(T-t)) * K**2 / B0 / (B_inf - B0)
                x = B0 + (B_inf - B0) * (1.0 - math.exp(hTt))
                alpha_x = (x - K) * x**(-beta)

                return (alpha_X * S**beta
                        - alpha_X * self.phi(S, t, beta, X, X, r, b, sigma)
                        + self.phi(S, t, 1.0, X, X, r, b, sigma)
                        - self.phi(S, t, 1.0, x, X, r, b, sigma)
                        - K * self.phi(S, t, 0.0, X, X, r, b, sigma)
                        + K * self.phi(S, t, 0.0, x, X, r, b, sigma)
                        + alpha_x * self.phi(S, t, beta, x, X, r, b, sigma)
                        - alpha_x * self.psi(S, T, beta, x, X, x, t, r, b, sigma)
                        + self.psi(S, T, 1.0, x, X, x, t, r, b, sigma)
                        - self.psi(S, T, 1.0, K, X, x, t, r, b, sigma)
                        - K * self.psi(S, T, 0.0, x, X, x, t, r, b, sigma)
                        + K * self.psi(S, T, 0.0, K, X, x, t, r, b, sigma))

    def phi(self, S, T, gamma, H, X, r, b, sigma):
        lambda_ = -r + gamma*b + 0.5*gamma*(gamma-1.0)*sigma**2
        kappa = 2*b/sigma**2 + (2*gamma-1.0)
        d1 = -(math.log(S/H) + (b+(gamma-0.5)*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = -(math.log(X**2/S/H) + (b+(gamma-0.5)*sigma**2)*T) / (sigma*math.sqrt(T))
        return math.exp(lambda_*T) * S**gamma * (self.bs.normcdf(d1) - (X/S)**kappa * self.bs.normcdf(d2))

    def psi(self, S, T, gamma, H, X, x, t, r, b, sigma):
        d1 = -(math.log(S/x) + (b+(gamma-0.5)*sigma**2)*t) / (sigma*math.sqrt(t))
        d2 = -(math.log(X**2/S/x) + (b+(gamma-0.5)*sigma**2)*t) / (sigma*math.sqrt(t))
        d3 = -(math.log(S/x) - (b+(gamma-0.5)*sigma**2)*t) / (sigma*math.sqrt(t))
        d4 = -(math.log(X**2/S/x) - (b+(gamma-0.5)*sigma**2)*t) / (sigma*math.sqrt(t))

        D1 = -(math.log(S/H) + (b+(gamma-0.5)*sigma**2)*T) / (sigma*math.sqrt(T))
        D2 = -(math.log(X**2/S/H) + (b+(gamma-0.5)*sigma**2)*T) / (sigma*math.sqrt(T))
        D3 = -(math.log(x**2/S/H) + (b+(gamma-0.5)*sigma**2)*T) / (sigma*math.sqrt(T))
        D4 = -(math.log(S*x**2/H/X**2) + (b+(gamma-0.5)*sigma**2)*T) / (sigma*math.sqrt(T))

        rho = math.sqrt(t/T)
        lambda_ = -r + gamma*b + 0.5*gamma*(gamma-1.0)*sigma**2
        kappa = 2*b/sigma**2 + (2*gamma-1.0)

        psi = math.exp(lambda_*T) * S**gamma * (
            self.bn.binormcdf(d1, D1, rho) -
            self.bn.binormcdf(d2, D2, rho) * (X/S)**kappa -
            self.bn.binormcdf(d3, D3, -rho) * (x/S)**kappa +
            self.bn.binormcdf(d4, D4, -rho) * (x/X)**kappa
        )
        return psi