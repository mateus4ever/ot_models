"""
ITEM 43 - PHASE 2G: Full Triangle Parameter Sweep
12 triangles, CHF 5000 capital at Oanda
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_pair(pair_name, data_dir):
    """Load and combine all yearly files for a pair"""
    pair_dir = Path(data_dir) / pair_name.lower()
    if not pair_dir.exists():
        print(f"  ⚠️ {pair_name} data not found at {pair_dir}")
        return None
    files = sorted(pair_dir.glob("*.csv"))
    if not files:
        print(f"  ⚠️ No CSV files found for {pair_name}")
        return None
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None,
                         names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
        dfs.append(df)
    print(f"  ✓ {pair_name}: {len(dfs)} files loaded")
    return pd.concat(dfs).sort_values('datetime').reset_index(drop=True)


def calculate_spread_series(pair_dfs, triangle_type, config):
    """
    Calculate spread for different triangle types
    """
    # Merge on datetime
    merged = pair_dfs[0][['datetime', 'close']].rename(columns={'close': 'pair1'})
    merged = merged.merge(pair_dfs[1][['datetime', 'close']].rename(columns={'close': 'pair2'}),
                          on='datetime', how='inner')
    merged = merged.merge(pair_dfs[2][['datetime', 'close']].rename(columns={'close': 'pair3'}),
                          on='datetime', how='inner')

    # Calculate based on formula type
    if config['formula'] == 'p1 = p2 * p3':
        merged['actual'] = merged['pair1']
        merged['synthetic'] = merged['pair2'] * merged['pair3']
    elif config['formula'] == 'p1 = p2 / p3':
        merged['actual'] = merged['pair1']
        merged['synthetic'] = merged['pair2'] / merged['pair3']
    elif config['formula'] == 'p1 = p3 / p2':
        merged['actual'] = merged['pair1']
        merged['synthetic'] = merged['pair3'] / merged['pair2']

    merged['spread'] = merged['actual'] - merged['synthetic']
    merged['spread_pips'] = merged['spread'] * config['pip_multiplier']

    return merged


def simulate_trading(test_df, train_mean, train_std, entry_z, exit_z, stop_z, cost_per_trade, lot_size):
    """Run trading simulation with given parameters"""
    test = test_df.copy()
    test['z_score'] = (test['spread_pips'] - train_mean) / train_std

    trades = []
    position = None
    entry_price = None
    entry_time = None

    for idx, row in test.iterrows():
        z = row['z_score']
        spread = row['spread_pips']

        if position is None:
            if z <= -entry_z:
                position = 'LONG'
                entry_price = spread
                entry_time = row['datetime']
            elif z >= entry_z:
                position = 'SHORT'
                entry_price = spread
                entry_time = row['datetime']
        else:
            exit_trade = False
            exit_reason = None

            if position == 'LONG':
                if z >= -exit_z:
                    exit_trade = True
                    exit_reason = 'TARGET'
                elif z <= -stop_z:
                    exit_trade = True
                    exit_reason = 'STOP'
            else:
                if z <= exit_z:
                    exit_trade = True
                    exit_reason = 'TARGET'
                elif z >= stop_z:
                    exit_trade = True
                    exit_reason = 'STOP'

            if exit_trade:
                if position == 'LONG':
                    pnl_pips = spread - entry_price
                else:
                    pnl_pips = entry_price - spread

                pnl_gross = pnl_pips * lot_size * 10
                pnl_net = pnl_gross - (cost_per_trade * lot_size)

                trades.append({
                    'pnl_pips': pnl_pips,
                    'pnl_gross': pnl_gross,
                    'pnl_net': pnl_net,
                    'exit_reason': exit_reason,
                    'duration_hours': (row['datetime'] - entry_time).total_seconds() / 3600
                })

                position = None
                entry_price = None
                entry_time = None

    return pd.DataFrame(trades)


# ============================================================
# CONFIGURATION
# ============================================================
print("=" * 80)
print("🚀 FULL TRIANGLE PARAMETER SWEEP (12 TRIANGLES)")
print("   Capital: CHF 5,000 at Oanda")
print("=" * 80)

# Capital constraints
OANDA_CAPITAL_CHF = 5000
TOTAL_CAPITAL_CHF = 50000
CHF_USD_RATE = 1.10
OANDA_CAPITAL_USD = OANDA_CAPITAL_CHF * CHF_USD_RATE

MARGIN_PER_LOT = 6600
MAX_LOT_SIZE = (OANDA_CAPITAL_USD * 0.8) / MARGIN_PER_LOT
LOT_SIZE = min(0.5, MAX_LOT_SIZE)

print(f"\nCapital: CHF {OANDA_CAPITAL_CHF:,} = ~${OANDA_CAPITAL_USD:,.0f}")
print(f"Max lot size (80% margin): {MAX_LOT_SIZE:.2f}")
print(f"Using lot size: {LOT_SIZE:.2f}")

# All 12 triangle definitions
TRIANGLES = {
    # Original 3 (tested)
    'EUR_GBP': {
        'name': 'EUR/USD-EUR/GBP-GBP/USD',
        'pairs': ['EURUSD', 'EURGBP', 'GBPUSD'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 10000,
        'spreads_only': [1.0, 1.1, 1.2],
        'spreads_core': [0.1, 0.2, 0.1],
    },
    'EUR_JPY': {
        'name': 'EUR/JPY-EUR/USD-USD/JPY',
        'pairs': ['EURJPY', 'EURUSD', 'USDJPY'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 100,
        'spreads_only': [1.6, 1.0, 1.1],
        'spreads_core': [0.6, 0.1, 0.1],
    },
    'EUR_CHF': {
        'name': 'EUR/CHF-EUR/USD-USD/CHF',
        'pairs': ['EURCHF', 'EURUSD', 'USDCHF'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 10000,
        'spreads_only': [1.5, 1.0, 1.3],
        'spreads_core': [0.4, 0.1, 0.1],
    },
    # JPY triangles
    'GBP_JPY': {
        'name': 'GBP/JPY-GBP/USD-USD/JPY',
        'pairs': ['GBPJPY', 'GBPUSD', 'USDJPY'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 100,
        'spreads_only': [1.0, 1.2, 1.1],
        'spreads_core': [0.3, 0.1, 0.1],
    },
    'AUD_JPY': {
        'name': 'AUD/JPY-AUD/USD-USD/JPY',
        'pairs': ['AUDJPY', 'AUDUSD', 'USDJPY'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 100,
        'spreads_only': [1.4, 1.1, 1.1],
        'spreads_core': [0.5, 0.3, 0.1],
    },
    'CAD_JPY': {
        'name': 'CAD/JPY-USD/JPY-USD/CAD',
        'pairs': ['CADJPY', 'USDJPY', 'USDCAD'],
        'formula': 'p1 = p2 / p3',
        'pip_multiplier': 100,
        'spreads_only': [1.5, 1.1, 1.5],
        'spreads_core': [0.5, 0.1, 0.3],
    },
    'CHF_JPY': {
        'name': 'CHF/JPY-USD/JPY-USD/CHF',
        'pairs': ['CHFJPY', 'USDJPY', 'USDCHF'],
        'formula': 'p1 = p2 / p3',
        'pip_multiplier': 100,
        'spreads_only': [1.5, 1.1, 1.3],
        'spreads_core': [0.5, 0.1, 0.1],
    },
    'NZD_JPY': {
        'name': 'NZD/JPY-NZD/USD-USD/JPY',
        'pairs': ['NZDJPY', 'NZDUSD', 'USDJPY'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 100,
        'spreads_only': [1.8, 1.4, 1.1],
        'spreads_core': [0.7, 0.4, 0.1],
    },
    # CHF triangles
    'GBP_CHF': {
        'name': 'GBP/CHF-GBP/USD-USD/CHF',
        'pairs': ['GBPCHF', 'GBPUSD', 'USDCHF'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 10000,
        'spreads_only': [1.6, 1.2, 1.3],
        'spreads_core': [0.5, 0.1, 0.1],
    },
    # AUD triangles
    'EUR_AUD': {
        'name': 'EUR/AUD-EUR/USD-AUD/USD',
        'pairs': ['EURAUD', 'EURUSD', 'AUDUSD'],
        'formula': 'p1 = p2 / p3',
        'pip_multiplier': 10000,
        'spreads_only': [1.8, 1.0, 1.1],
        'spreads_core': [0.6, 0.1, 0.3],
    },
    'GBP_AUD': {
        'name': 'GBP/AUD-GBP/USD-AUD/USD',
        'pairs': ['GBPAUD', 'GBPUSD', 'AUDUSD'],
        'formula': 'p1 = p2 / p3',
        'pip_multiplier': 10000,
        'spreads_only': [1.9, 1.2, 1.1],
        'spreads_core': [0.7, 0.1, 0.3],
    },
    # CAD triangle
    'EUR_CAD': {
        'name': 'EUR/CAD-EUR/USD-USD/CAD',
        'pairs': ['EURCAD', 'EURUSD', 'USDCAD'],
        'formula': 'p1 = p2 * p3',
        'pip_multiplier': 10000,
        'spreads_only': [1.8, 1.0, 1.5],
        'spreads_core': [0.6, 0.1, 0.3],
    },
}


def calculate_costs(triangle_config, lot_size):
    """Calculate costs for a triangle"""
    spread_only_pips = sum(triangle_config['spreads_only'])
    core_pips = sum(triangle_config['spreads_core'])
    commission_per_100k = 2.45

    return {
        'Spread-Only': {
            'spread_cost': spread_only_pips * 2 * 10 * lot_size,
            'commission': 0,
            'total': spread_only_pips * 2 * 10 * lot_size,
        },
        'Core': {
            'spread_cost': core_pips * 2 * 10 * lot_size,
            'commission': commission_per_100k * 6 * lot_size,
            'total': (core_pips * 2 * 10 * lot_size) + (commission_per_100k * 6 * lot_size),
        }
    }


# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

data_dir = r"C:\temp\git\ot_models\src\experiments\data"

# Get all unique pairs needed
all_pairs_needed = set()
for config in TRIANGLES.values():
    all_pairs_needed.update(config['pairs'])

print(f"\nPairs needed: {len(all_pairs_needed)}")

# Load all pairs
all_pairs = {}
for pair in sorted(all_pairs_needed):
    all_pairs[pair] = load_pair(pair, data_dir)

# Summary
print("\n" + "-" * 40)
print("Data availability:")
available = sum(1 for p in all_pairs.values() if p is not None)
print(f"  Available: {available}/{len(all_pairs_needed)}")

for pair in sorted(all_pairs_needed):
    status = "✅" if all_pairs[pair] is not None else "❌"
    print(f"  {pair}: {status}")

# ============================================================
# PROCESS EACH TRIANGLE
# ============================================================
all_results = []
triangle_stats = {}

for triangle_type, config in TRIANGLES.items():
    print("\n" + "=" * 80)
    print(f"TRIANGLE: {config['name']}")
    print("=" * 80)

    # Check if all pairs are available
    pairs = config['pairs']
    pair_data = [all_pairs.get(p) for p in pairs]

    if any(p is None for p in pair_data):
        missing = [pairs[i] for i, p in enumerate(pair_data) if p is None]
        print(f"  ❌ Missing data: {missing}, skipping...")
        continue

    # Calculate spread
    print(f"  Calculating spread...")
    try:
        merged = calculate_spread_series(pair_data, triangle_type, config)
    except Exception as e:
        print(f"  ❌ Error calculating spread: {e}")
        continue

    # Aggregate to hourly
    merged = merged.set_index('datetime')
    hourly = merged.resample('1h').agg({
        'actual': 'last',
        'synthetic': 'last',
        'spread_pips': 'last'
    }).dropna().reset_index()

    # Split train/test
    train = hourly[hourly['datetime'] < '2025-01-01'].copy()
    test = hourly[hourly['datetime'] >= '2025-01-01'].copy()

    if len(test) == 0:
        print(f"  ❌ No 2025 data for {triangle_type}, skipping...")
        continue

    train_mean = train['spread_pips'].mean()
    train_std = train['spread_pips'].std()

    # Store stats for later use
    triangle_stats[triangle_type] = {
        'train_mean': train_mean,
        'train_std': train_std,
    }

    print(f"  Train: {len(train):,} hours, mean={train_mean:.2f}, std={train_std:.2f}")
    print(f"  Test:  {len(test):,} hours (2025)")

    # Calculate costs
    costs = calculate_costs(config, LOT_SIZE)

    print(f"\n  Costs ({LOT_SIZE} lot):")
    for cost_type, cost_data in costs.items():
        print(f"    {cost_type}: ${cost_data['total']:.2f}/trade")

    # Parameter sweep
    ENTRY_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    EXIT_Z = 0.5
    STOP_Z_MULTIPLIER = 2.0

    for entry_z in ENTRY_THRESHOLDS:
        stop_z = entry_z * STOP_Z_MULTIPLIER
        entry_pips = entry_z * train_std

        for cost_name, cost_struct in costs.items():
            trades_df = simulate_trading(
                test, train_mean, train_std,
                entry_z, EXIT_Z, stop_z,
                cost_struct['total'] / LOT_SIZE,
                lot_size=LOT_SIZE
            )

            if len(trades_df) > 0:
                n_trades = len(trades_df)
                n_winners = (trades_df['pnl_net'] > 0).sum()
                win_rate = n_winners / n_trades * 100
                gross_pnl = trades_df['pnl_gross'].sum()
                net_pnl = trades_df['pnl_net'].sum()
                targets = (trades_df['exit_reason'] == 'TARGET').sum()
                stops = (trades_df['exit_reason'] == 'STOP').sum()
                dur_mean = trades_df['duration_hours'].mean()
                dur_median = trades_df['duration_hours'].median()
                dur_min = trades_df['duration_hours'].min()
                dur_max = trades_df['duration_hours'].max()
            else:
                n_trades = n_winners = 0
                win_rate = gross_pnl = net_pnl = 0
                targets = stops = 0
                dur_mean = dur_median = dur_min = dur_max = 0

            all_results.append({
                'Triangle': config['name'],
                'Triangle_Type': triangle_type,
                'Entry (σ)': entry_z,
                'Entry (pips)': entry_pips,
                'Cost Type': cost_name,
                'Trades': n_trades,
                'Win %': win_rate,
                'Targets': targets,
                'Stops': stops,
                'Gross $': gross_pnl,
                'Net $': net_pnl,
                'Dur Mean (h)': dur_mean,
                'Dur Median (h)': dur_median,
                'Dur Min (h)': dur_min,
                'Dur Max (h)': dur_max,
            })

results_df = pd.DataFrame(all_results)

# ============================================================
# DISPLAY RESULTS BY TRIANGLE
# ============================================================
if len(results_df) > 0:
    for triangle_type in results_df['Triangle_Type'].unique():
        triangle_name = results_df[results_df['Triangle_Type'] == triangle_type]['Triangle'].iloc[0]

        print("\n" + "=" * 80)
        print(f"RESULTS: {triangle_name}")
        print("=" * 80)

        for cost_type in ['Spread-Only', 'Core']:
            subset = results_df[(results_df['Triangle_Type'] == triangle_type) &
                                (results_df['Cost Type'] == cost_type)]

            if len(subset) == 0:
                continue

            print(f"\n{cost_type}:")
            print(f"{'Entry':>6} {'Trades':>7} {'Win%':>6} {'Target':>7} {'Stop':>6} {'Gross':>10} {'NET':>10}")
            print("-" * 60)

            for _, row in subset.iterrows():
                icon = "✅" if row['Net $'] > 0 else "❌"
                print(f"{row['Entry (σ)']:>6.1f} {row['Trades']:>7.0f} {row['Win %']:>5.1f}% "
                      f"{row['Targets']:>7.0f} {row['Stops']:>6.0f} "
                      f"{row['Gross $']:>10.0f} {row['Net $']:>9.0f} {icon}")

    # ============================================================
    # COMBINED SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("🏆 COMBINED SUMMARY (Core Pricing)")
    print(f"   Lot size: {LOT_SIZE}, Capital: CHF {OANDA_CAPITAL_CHF:,}")
    print("=" * 80)

    core_results = results_df[results_df['Cost Type'] == 'Core']

    if len(core_results) > 0:
        print(f"\n{'Triangle':<30} {'Best Entry':>10} {'Trades':>8} {'Win%':>7} {'Net $/yr':>10}")
        print("-" * 70)

        total_trades = 0
        total_net = 0
        best_params = {}

        for triangle_type in core_results['Triangle_Type'].unique():
            subset = core_results[core_results['Triangle_Type'] == triangle_type]
            best = subset.loc[subset['Net $'].idxmax()]

            triangle_name = best['Triangle'][:28]
            print(f"{triangle_name:<30} {best['Entry (σ)']:>9.1f}σ {best['Trades']:>8.0f} "
                  f"{best['Win %']:>6.0f}% {best['Net $']:>10.0f}")

            total_trades += best['Trades']
            total_net += best['Net $']

            # Store best params for monitor script
            best_params[triangle_type] = {
                'entry_z': best['Entry (σ)'],
                'trades': best['Trades'],
                'win_rate': best['Win %'],
                'net': best['Net $'],
                'dur_median': best['Dur Median (h)'],
                'dur_max': best['Dur Max (h)'],
            }

        print("-" * 70)
        print(f"{'TOTAL':<30} {'':<10} {total_trades:>8.0f} {'':<7} ${total_net:>9.0f}")

        return_on_oanda = (total_net / OANDA_CAPITAL_USD) * 100
        return_on_total = (total_net / (TOTAL_CAPITAL_CHF * CHF_USD_RATE)) * 100

        print(f"\n📊 Returns:")
        print(f"   On Oanda capital (CHF {OANDA_CAPITAL_CHF:,}): {return_on_oanda:.1f}%")
        print(f"   On total capital (CHF {TOTAL_CAPITAL_CHF:,}): {return_on_total:.1f}%")

        max_loss_per_trade = LOT_SIZE * 10 * 6 * 1.5
        risk_per_trade = (max_loss_per_trade / (TOTAL_CAPITAL_CHF * CHF_USD_RATE)) * 100

        print(f"\n⚠️ Risk:")
        print(f"   Max loss per trade: ~${max_loss_per_trade:.0f}")
        print(f"   Risk per trade vs total capital: {risk_per_trade:.2f}%")

        # ============================================================
        # DURATION SUMMARY
        # ============================================================
        print("\n" + "=" * 80)
        print("⏱️ TRADE DURATION SUMMARY (Core Pricing, Best Entry)")
        print("=" * 80)
        print(f"\n{'Triangle':<20} {'Mean(h)':>10} {'Median(h)':>12} {'Min(h)':>10} {'Max(h)':>10} {'Max(days)':>12}")
        print("-" * 80)

        all_durations = []
        for triangle_type, params in best_params.items():
            subset = core_results[(core_results['Triangle_Type'] == triangle_type) &
                                  (core_results['Entry (σ)'] == params['entry_z'])]
            if len(subset) > 0:
                row = subset.iloc[0]
                print(f"{triangle_type:<20} {row['Dur Mean (h)']:>10.1f} {row['Dur Median (h)']:>12.1f} "
                      f"{row['Dur Min (h)']:>10.1f} {row['Dur Max (h)']:>10.1f} {row['Dur Max (h)']/24:>12.1f}")
                all_durations.append({
                    'mean': row['Dur Mean (h)'],
                    'median': row['Dur Median (h)'],
                    'max': row['Dur Max (h)'],
                })

        if all_durations:
            print("-" * 80)
            avg_median = np.mean([d['median'] for d in all_durations])
            max_max = max([d['max'] for d in all_durations])
            print(f"{'PORTFOLIO AVG/MAX':<20} {'':<10} {avg_median:>12.1f} {'':<10} {max_max:>10.1f} {max_max/24:>12.1f}")

            print(f"\n📊 Duration Insights:")
            print(f"   Average trade duration (median): {avg_median:.1f} hours ({avg_median/24:.1f} days)")
            print(f"   Longest trade ever: {max_max:.1f} hours ({max_max/24:.1f} days)")
            print(f"   Capital turnover: ~{365*24/avg_median:.0f}x per year")


        # ============================================================
        # PARAMETERS FOR MONITOR SCRIPT
        # ============================================================
        print("\n" + "=" * 80)
        print("📋 PARAMETERS FOR MONITOR SCRIPT")
        print("=" * 80)
        print("\nCopy this to vasicek_monitor.py:\n")
        print("TRIANGLES = {")
        for triangle_type, params in best_params.items():
            config = TRIANGLES[triangle_type]
            stats = triangle_stats.get(triangle_type, {})
            print(f"    '{triangle_type}': {{")
            print(f"        'name': '{config['name']}',")
            print(f"        'pairs': {config['pairs']},")
            print(f"        'formula': '{config['formula']}',")
            print(f"        'pip_multiplier': {config['pip_multiplier']},")
            print(f"        'train_mean': {stats.get('train_mean', 0):.2f},")
            print(f"        'train_std': {stats.get('train_std', 1):.2f},")
            print(f"        'entry_z': {params['entry_z']:.1f},")
            print(f"        'exit_z': 0.5,")
            print(f"    }},")
        print("}")

        # ============================================================
        # SCALING PROJECTION
        # ============================================================
        print("\n" + "=" * 80)
        print("📈 SCALING PROJECTION")
        print("=" * 80)

        profit_per_lot = total_net / LOT_SIZE

        print(f"\n{'Oanda Capital':<15} {'Lot Size':<10} {'Annual Net':<12} {'Return %':<10}")
        print("-" * 50)

        for capital in [5000, 7500, 10000, 12500, 15000]:
            capital_usd = capital * CHF_USD_RATE
            max_lot = (capital_usd * 0.8) / MARGIN_PER_LOT
            lot = min(max_lot, 1.5)
            annual = profit_per_lot * lot
            ret = (annual / capital_usd) * 100

            print(f"CHF {capital:<10,} {lot:<10.2f} ${annual:<11,.0f} {ret:<10.1f}%")

        # ============================================================
        # VISUALIZATION
        # ============================================================
        n_triangles = len(core_results['Triangle_Type'].unique())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Net P&L by triangle and entry (Core)
        ax1 = axes[0, 0]
        for triangle_type in core_results['Triangle_Type'].unique():
            subset = core_results[core_results['Triangle_Type'] == triangle_type]
            ax1.plot(subset['Entry (σ)'], subset['Net $'], marker='o', label=triangle_type, linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--')
        ax1.set_xlabel('Entry Threshold (σ)')
        ax1.set_ylabel('Net P&L ($)')
        ax1.set_title(f'Net P&L by Triangle (Core, {LOT_SIZE} lot)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Trade count by triangle
        ax2 = axes[0, 1]
        best_per_triangle = core_results.loc[core_results.groupby('Triangle_Type')['Net $'].idxmax()]
        ax2.barh(best_per_triangle['Triangle_Type'], best_per_triangle['Trades'], alpha=0.7)
        ax2.set_xlabel('Trades/Year')
        ax2.set_ylabel('Triangle')
        ax2.set_title('Annual Trades (Best Entry per Triangle)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Net P&L per triangle (best entry)
        ax3 = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in best_per_triangle['Net $']]
        ax3.barh(best_per_triangle['Triangle_Type'], best_per_triangle['Net $'], color=colors, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-')
        ax3.set_xlabel('Net P&L ($)')
        ax3.set_ylabel('Triangle')
        ax3.set_title('Annual Net P&L (Best Entry per Triangle)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary box
        ax4 = axes[1, 1]
        summary_text = (
            f"PORTFOLIO SUMMARY\n"
            f"─────────────────\n"
            f"Triangles: {n_triangles}\n"
            f"Total trades: {total_trades:.0f}/year\n"
            f"Trades per day: {total_trades / 365:.1f}\n"
            f"\n"
            f"Annual Net P&L: ${total_net:,.0f}\n"
            f"\n"
            f"Return on Oanda (CHF {OANDA_CAPITAL_CHF:,}): {return_on_oanda:.1f}%\n"
            f"Return on Total (CHF {TOTAL_CAPITAL_CHF:,}): {return_on_total:.1f}%\n"
            f"\n"
            f"Risk per trade: {risk_per_trade:.2f}%"
        )
        ax4.text(0.5, 0.5, summary_text,
                 ha='center', va='center', fontsize=14,
                 transform=ax4.transAxes, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen' if total_net > 0 else 'lightcoral', alpha=0.8))
        ax4.set_title('Portfolio Summary')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(r'C:\temp\git\ot_models\src\experiments\vasicek\full_triangle_sweep_2025.png', dpi=150)
        print(f"\n✓ Chart saved: full_triangle_sweep_2025.png")

        plt.show()

else:
    print("\n❌ No results generated. Check data availability.")
    print("\nDownload these pairs to c:\\git\\vasicek\\data\\<pair>\\:")
    for pair in sorted(all_pairs_needed):
        if all_pairs.get(pair) is None:
            print(f"  - {pair}")