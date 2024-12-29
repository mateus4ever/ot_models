from src.bjerksund_stendland.cb.bjerk2002 import bjerk2002

BS02 = bjerk2002()

# Time steps for trinomial tree
N = 25

# Table 1 of Bjerksund and Stensland (2002)
K = [230.0, 240.0, 250.0, 260.0, 270.0]  # Strike price
S = 254.78
# Spot price
r = [0.0462, 0.0462, 0.0462, 0.0462]  # Risk-free rates
sigma = [0.2236, 0.2236, 0.2236, 0.2236]  # Volatilities
T = [0.25, 0.5, 1.0, 1.5]  # Time to maturity
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
        Call02[k][j] = BS02.bjerk_price_2002(S, K[k], T[j], r[j], b, sigma[j], 'C')
        # TreePut[k][j] = TT.trinomial(S[k], K, r[j], q[j], sigma[j], T[j], 'P', 'A', N)
        # Put93[k][j] = BS93.bjerk_price_1993(S[k], K, T[j], r[j], b, sigma[j], 'P')
        Put02[k][j] = BS02.bjerk_price_2002(S, K[k], T[j], r[j], b, sigma[j], 'P')

        print(f"{K[k]:5.0f} {TreeCall[k][j]:8.4f} {Call93[k][j]:8.4f} {Call02[k][j]:8.4f}   |  "
              f"{TreePut[k][j]:8.4f} {Put93[k][j]:8.4f} {Put02[k][j]:8.4f}")
    print("-----------------------------------+-----------------------------")
