"""
Item 43 - Phase 1: Triangular Arbitrage Test
Tests if EUR/USD vs synthetic spread is stationary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats
from datetime import datetime
from pathlib import Path


# ============================================================================
# MULTI-PAIR DATA LOADER
# ============================================================================

def load_pair_data(pair_name, data_dir='../data'):
    """
    Load all CSV files for a specific pair

    Args:
        pair_name: 'EURUSD', 'EURGBP', or 'GBPUSD'
        data_dir: Base data directory

    Returns:
        pandas.Series: Daily close prices
    """
    # Try subdirectory structure first
    pair_path = Path(__file__).parent / data_dir / pair_name

    if not pair_path.exists():
        # Fallback: all files in one directory
        pair_path = Path(__file__).parent / data_dir

    print(f"\n  Loading {pair_name}...")
    print(f"    Path: {pair_path}")

    # Find files for this pair
    pattern = f"*{pair_name}*.csv"
    csv_files = sorted(pair_path.glob(pattern))

    if not csv_files:
        print(f"    ❌ No files found matching: {pattern}")
        return None

    print(f"    Found {len(csv_files)} file(s)")

    # Load all files
    all_prices = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(
                csv_file,
                names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
                parse_dates=False
            )
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'],
                                            format='%Y.%m.%d %H:%M')
            df.set_index('datetime', inplace=True)
            prices = df['close'].sort_index()
            all_prices.append(prices)
        except Exception as e:
            print(f"    ⚠️ Error loading {csv_file.name}: {e}")
            continue

    if not all_prices:
        return None

    # Combine all data
    combined = pd.concat(all_prices).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    # Resample to daily
    daily = combined.resample('D').last().dropna()

    print(f"    ✓ Loaded {len(daily)} daily bars")
    print(f"      Range: {daily.index[0].date()} to {daily.index[-1].date()}")
    print(f"      Price: {daily.min():.5f} to {daily.max():.5f}")

    return daily


def load_triangular_data(data_dir='../data'):
    """Load all three pairs for triangular arbitrage"""
    print("=" * 70)
    print("LOADING TRIANGULAR ARBITRAGE DATA")
    print("=" * 70)

    eur_usd = load_pair_data('EURUSD', data_dir)
    eur_gbp = load_pair_data('EURGBP', data_dir)
    gbp_usd = load_pair_data('GBPUSD', data_dir)

    # Check all loaded
    if eur_usd is None:
        print("\n❌ EUR/USD data not found!")
        return None, None, None
    if eur_gbp is None:
        print("\n❌ EUR/GBP data not found!")
        print("   Download EUR/GBP M1 data (2017-2024)")
        return None, None, None
    if gbp_usd is None:
        print("\n❌ GBP/USD data not found!")
        print("   Download GBP/USD M1 data (2017-2024)")
        return None, None, None

    # Align dates (use intersection)
    common_dates = eur_usd.index.intersection(eur_gbp.index).intersection(gbp_usd.index)

    eur_usd = eur_usd.loc[common_dates]
    eur_gbp = eur_gbp.loc[common_dates]
    gbp_usd = gbp_usd.loc[common_dates]

    print(f"\n✓ All pairs loaded successfully!")
    print(f"  Common trading days: {len(common_dates)}")
    print(f"  Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

    return eur_usd, eur_gbp, gbp_usd


# ============================================================================
# CALCULATE SPREAD
# ============================================================================

def calculate_triangular_spread(eur_usd, eur_gbp, gbp_usd):
    """Calculate spread between actual and synthetic EUR/USD"""
    print("\n" + "=" * 70)
    print("CALCULATING TRIANGULAR ARBITRAGE SPREAD")
    print("=" * 70)

    # Synthetic EUR/USD = EUR/GBP × GBP/USD
    synthetic = eur_gbp * gbp_usd

    # Spread = Actual - Synthetic (in pips)
    spread_price = eur_usd - synthetic
    spread_pips = spread_price * 10000  # Convert to pips

    print(f"\nSynthetic EUR/USD calculation:")
    print(f"  EUR/GBP × GBP/USD = EUR/USD (synthetic)")
    print(f"  Example (last day):")
    print(f"    {eur_gbp.iloc[-1]:.5f} × {gbp_usd.iloc[-1]:.5f} = {synthetic.iloc[-1]:.5f}")
    print(f"    Actual EUR/USD: {eur_usd.iloc[-1]:.5f}")
    print(f"    Spread: {spread_pips.iloc[-1]:.2f} pips")

    print(f"\nSpread statistics:")
    print(f"  Mean: {spread_pips.mean():.2f} pips")
    print(f"  Std Dev: {spread_pips.std():.2f} pips")
    print(f"  Min: {spread_pips.min():.2f} pips")
    print(f"  Max: {spread_pips.max():.2f} pips")

    return spread_price, spread_pips, synthetic


# ============================================================================
# TEST SPREAD STATIONARITY
# ============================================================================

def test_spread_stationarity(spread):
    """Test if spread is stationary (mean-reverting)"""
    print("\n" + "=" * 70)
    print("TESTING SPREAD STATIONARITY")
    print("=" * 70)

    # ADF test
    result = adfuller(spread, autolag='AIC')
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f"\n1. ADF Stationarity Test:")
    print(f"   ADF Statistic: {adf_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Critical Values:")
    for key, value in critical_values.items():
        marker = " ←" if adf_stat < value else ""
        print(f"     {key:>3}: {value:.4f}{marker}")

    is_stationary = p_value < 0.05

    if p_value < 0.01:
        print(f"\n   ✅✅ HIGHLY STATIONARY (p={p_value:.4f} < 0.01)")
        print(f"      → Spread is STRONGLY mean-reverting")
        print(f"      → Excellent for trading!")
    elif is_stationary:
        print(f"\n   ✅ STATIONARY (p={p_value:.4f} < 0.05)")
        print(f"      → Spread is mean-reverting")
        print(f"      → Good for trading")
    else:
        print(f"\n   ❌ NON-STATIONARY (p={p_value:.4f} >= 0.05)")
        print(f"      → Spread is NOT mean-reverting")
        print(f"      → Don't trade this")

    # Hurst exponent
    lags = range(2, min(100, len(spread) // 2))
    tau = [np.sqrt(np.std(np.subtract(spread[lag:].values, spread[:-lag].values)))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0

    print(f"\n2. Hurst Exponent: {hurst:.4f}")
    if hurst < 0.5:
        print(f"   ✅ MEAN-REVERTING (H < 0.5)")
    elif hurst > 0.5:
        print(f"   ❌ TRENDING (H > 0.5)")
    else:
        print(f"   ⚠️ RANDOM WALK (H ≈ 0.5)")

    return is_stationary, p_value, hurst


# ============================================================================
# CALCULATE HALF-LIFE
# ============================================================================

def calculate_half_life(spread):
    """Calculate how fast spread reverts to mean"""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    slope, intercept, r, p, stderr = stats.linregress(
        spread_lag[1:], spread_diff[1:]
    )

    lambda_param = -slope

    if lambda_param <= 0:
        return np.inf

    half_life = np.log(2) / lambda_param

    print(f"\n3. Mean Reversion Speed:")
    print(f"   Half-life: {half_life:.1f} days")

    if half_life < 5:
        print(f"   ✅✅ VERY FAST reversion (< 5 days)")
        print(f"      → Capital not tied up long")
        print(f"      → High trading frequency possible")
    elif half_life < 20:
        print(f"   ✅ FAST reversion (< 20 days)")
    elif half_life < 60:
        print(f"   ⚠️ MODERATE reversion (< 60 days)")
    else:
        print(f"   ❌ SLOW reversion (> 60 days)")
        print(f"      → Capital tied up too long")

    return half_life


# ============================================================================
# VISUALIZE
# ============================================================================

def visualize_triangular_arbitrage(eur_usd, synthetic, spread_pips):
    """Create visualization"""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Actual vs Synthetic
    ax1.plot(eur_usd, label='Actual EUR/USD', linewidth=1, alpha=0.8)
    ax1.plot(synthetic, label='Synthetic (EUR/GBP × GBP/USD)',
             linewidth=1, alpha=0.8, linestyle='--')
    ax1.set_title('Triangular Arbitrage: Actual vs Synthetic EUR/USD',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spread in pips
    ax2.plot(spread_pips, label='Spread (pips)', linewidth=1, color='purple')
    ax2.axhline(spread_pips.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {spread_pips.mean():.2f} pips')
    ax2.fill_between(spread_pips.index,
                     spread_pips.mean() - 2 * spread_pips.std(),
                     spread_pips.mean() + 2 * spread_pips.std(),
                     alpha=0.2, color='gray', label='±2σ')
    ax2.set_title('Spread (Should Oscillate Around Mean if Stationary)',
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Pips')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Z-score
    z_score = (spread_pips - spread_pips.mean()) / spread_pips.std()
    ax3.plot(z_score, label='Z-Score', linewidth=1, color='green')
    ax3.axhline(0, color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axhline(2, color='orange', linestyle=':', label='Entry threshold (±2σ)')
    ax3.axhline(-2, color='orange', linestyle=':')
    ax3.fill_between(z_score.index, -2, 2, alpha=0.2, color='green')
    ax3.set_title('Z-Score: Trade When |Z| > 2',
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel('Standard Deviations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / 'triangular_arbitrage_spread.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Chart saved: {output_path}")
    plt.show()


# ============================================================================
# FINAL DECISION
# ============================================================================

def final_decision(is_stationary, p_value, hurst, half_life):
    """Make trading decision"""
    print("\n" + "=" * 70)
    print("FINAL DECISION")
    print("=" * 70)

    tests_passed = sum([
        p_value < 0.05,
        hurst < 0.5,
        half_life < 60
    ])

    print(f"\nTest Results:")
    print(f"  1. Spread Stationarity: {'✅ PASS' if p_value < 0.05 else '❌ FAIL'} (p={p_value:.4f})")
    print(f"  2. Mean Reversion: {'✅ PASS' if hurst < 0.5 else '❌ FAIL'} (H={hurst:.4f})")
    print(f"  3. Reversion Speed: {'✅ PASS' if half_life < 60 else '❌ FAIL'} ({half_life:.1f} days)")

    print(f"\n" + "=" * 70)

    if tests_passed >= 3:
        print(f"✅✅✅ EXCELLENT - ALL TESTS PASSED!")
        print(f"\nTriangular arbitrage spread is:")
        print(f"  ✓ Stationary (mean-reverting)")
        print(f"  ✓ Fast reverting (half-life {half_life:.1f} days)")
        print(f"  ✓ TRADEABLE!")
        print(f"\nNext Steps:")
        print(f"  1. Item 21: Implement pairs trading strategy")
        print(f"  2. Trade the SPREAD, not individual pairs")
        print(f"  3. Entry: |Z-score| > 2")
        print(f"  4. Exit: Z-score returns to 0")
        decision = "TRADE"

    elif tests_passed >= 2:
        print(f"⚠️ ACCEPTABLE - Most tests passed")
        print(f"\nSpread shows some mean reversion")
        print(f"Consider trading with caution")
        decision = "CAUTION"

    else:
        print(f"❌ FAILED - Spread not stationary")
        print(f"\nTriangular arbitrage doesn't work for this combination")
        print(f"\nNext Steps:")
        print(f"  1. Try other pair combinations (AUD/NZD, etc.)")
        print(f"  2. Or return to ML (Item 42)")
        decision = "REJECT"

    print(f"=" * 70)

    return decision


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ITEM 43 - PHASE 1: TRIANGULAR ARBITRAGE TEST")
    print("Testing: EUR/USD vs Synthetic (EUR/GBP × GBP/USD)")
    print("=" * 70)

    try:
        # Load data
        eur_usd, eur_gbp, gbp_usd = load_triangular_data(data_dir='../data')

        if eur_usd is None:
            print("\n❌ Cannot proceed - missing data files")
            print("\nDownload needed:")
            print("  - EUR/GBP M1 (2017-2024)")
            print("  - GBP/USD M1 (2017-2024)")
            exit(1)

        # Calculate spread
        spread_price, spread_pips, synthetic = calculate_triangular_spread(
            eur_usd, eur_gbp, gbp_usd
        )

        # Test stationarity
        is_stationary, p_value, hurst = test_spread_stationarity(spread_price)

        # Calculate half-life
        half_life = calculate_half_life(spread_price)

        # Visualize
        visualize_triangular_arbitrage(eur_usd, synthetic, spread_pips)

        # Decision
        decision = final_decision(is_stationary, p_value, hurst, half_life)

        # Save results
        results = {
            'test': 'triangular_arbitrage',
            'pairs': ['EURUSD', 'EURGBP', 'GBPUSD'],
            'n_observations': len(eur_usd),
            'date_range': f"{eur_usd.index[0].date()} to {eur_usd.index[-1].date()}",
            'spread_mean_pips': float(spread_pips.mean()),
            'spread_std_pips': float(spread_pips.std()),
            'adf_pvalue': float(p_value),
            'hurst_exponent': float(hurst),
            'half_life_days': float(half_life),
            'is_stationary': bool(is_stationary),
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        }

        import json

        output_path = Path(__file__).parent / 'triangular_arbitrage_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()