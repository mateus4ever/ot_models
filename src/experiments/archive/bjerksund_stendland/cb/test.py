from src.experiments.archive.bjerksund_stendland.cb.bjerk2002 import bjerk2002

BS02 = bjerk2002()

# Time steps for trinomial tree
N = 25

# Example usage
# S = 254.77  # Current stock price
# K = 250  # Strike price
# t = 0.0199  # Time to maturity (1 year)
# # https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_long_term_rate&field_tdr_date_value_month=202412
# r = 0.0487  # Risk-free interest rate (5%)
# sigma = 0.1924  # Volatility (20%)

# Table 1 of Bjerksund and Stensland (2002)
# Default
# K = [230.0, 240.0, 250.0, 260.0, 270.0]  # Strike price
# S = 255.65
# # Spot price
# r = [0.0487, 0.0487, 0.0487, 0.0487]  # Risk-free rates
# sigma = [0.190966, 0.2, 0.22, 0.25]  # Volatilities
# T = [0.0199, 0.0199, 0.0199, 0.0199]  # Time to maturity
# b = -0.04  # Cost of carry
# q = [ri - b for ri in r]  # Dividend yields

# Table 1 increasing volatility
K = [240.0, 245.0, 250.0, 255.0, 260.0]  # Strike price
S = [250.0, 243.0, 248.0, 242.0, 240.0] # Spot price
r = [0.0462, 0.0462, 0.0462, 0.0462]  # Risk-free rates
sigma = [0.20, 0.24, 0.26, 0.28]  # Volatilities
T = [0.5, 0.462, 0.424, 0.386]  # Time to maturity
b = -0.04  # Cost of carry
q = [ri - b for ri in r]  # Dividend yields

# Initialize results storage
TreeCall = [[0.0 for _ in range(4)] for _ in range(5)]
Call93 = [[0.0 for _ in range(4)] for _ in range(5)]
Call02 = [[0.0 for _ in range(4)] for _ in range(5)]
TreePut = [[0.0 for _ in range(4)] for _ in range(5)]
Put93 = [[0.0 for _ in range(4)] for _ in range(5)]
Put02 = [[0.0 for _ in range(4)] for _ in range(5)]

print("Table 1 of Bjerksund and Stensland (1993,2002)")
print("Trinomial tree price, flat boundary price (1993)")
print("and two-step boundary price (2002)")
print("-----------------------------------+-----------------------------")
print(" Strike      American Call         |         American Put")
print(" Price  Tree   Price93   Price02   |    Tree   Price93   Price02")
print("-----------------------------------+-----------------------------")

for j in range(4):
    for k in range(5):
        # TreeCall[k][j] = TT.trinomial(S[k], K, r[j], q[j], sigma[j], T[j], 'C', 'A', N)
        # Call93[k][j] = BS93.bjerk_price_1993(S[k], K, T[j], r[j], b, sigma[j], 'C')
        Call02[k][j] = BS02.bjerk_price_2002(S[k], K[k], T[j], r[j], b, sigma[j], 'C')
        # TreePut[k][j] = TT.trinomial(S[k], K, r[j], q[j], sigma[j], T[j], 'P', 'A', N)
        # Put93[k][j] = BS93.bjerk_price_1993(S[k], K, T[j], r[j], b, sigma[j], 'P')
        Put02[k][j] = BS02.bjerk_price_2002(S[k], K[k], T[j], r[j], b, sigma[j], 'P')

        print(f"{K[k]:5.0f} {TreeCall[k][j]:8.4f} {Call93[k][j]:8.4f} {Call02[k][j]:8.4f}   |  "
              f"{TreePut[k][j]:8.4f} {Put93[k][j]:8.4f} {Put02[k][j]:8.4f}")
    print("-----------------------------------+-----------------------------")
