# Predictor Improvement Roadmap

**Date:** December 28, 2025

## Current Situation

- Volatility predictor validation framework works, tells the truth
- No edge found with current features (all tests worse than naive baseline)
- Trend duration predictor needs cleanup and validation
- New direction: Vasicek mean-reversion model (from Gary Stevenson insights)

## Philosophy: The "Poor Man's" Trading Approach

### Core Principles

1. **Avoid the Quant War** - Don't compete on speed or data volume against institutional players
2. **Eat the Breadcrumbs** - Find moments of market inefficiency that large firms ignore due to their size
3. **Intrinsic Data Only** - Data must speak for itself, avoid external correlations
4. **Silence Before the Storm** - Primary signal is compression/coiling before breakouts
5. **Economic Gravity** - Prices stretched too far from mean get pulled back (Vasicek)

### Key Realizations

- **The Accuracy Paradox:** 58% accuracy was actually -9.1% edge (worse than naive baseline)
- **Root Cause of Failure:** Current features capture present state, not precursors to change
- **Model Blindness:** Missing time-of-day cycles and internal "stress" of price bars
- **Mean Reversion Edge:** Small traders can wait for high-probability (3σ) deviations

### Realistic Expectations (20,000 CHF Capital)

- **Target:** 20% net annual return (4,000 CHF)
- **Gross needed:** ~25-28% to cover spreads, commissions, swap fees
- **Edge:** Agility to avoid noisy markets, only trade high-probability setups

## Work Items

| Nr | Predictor | What | Description | To Be Done | Effort | Success (1-5) | Reason | Best in Combination |
|----|-----------|------|-------------|------------|--------|---------------|--------|---------------------|
| 1 | Volatility | Compression features | BB squeeze, ATR compression, tight bars | Add features to volatility predictor, run validation | Medium | 4 | Compression precedes breakouts - direct precursor to volatility spikes | 2, 3, 11, 12 |
| 2 | Volatility | Key level features | Distance to high/low, round numbers | Add features to volatility predictor, run validation | Medium | 3 | Price at key levels often triggers volatility, but may be noisy | 1, 3, 11 |
| 3 | Volatility | Calendar features | Hour, day of week, session (London/NY) | Add features to volatility predictor, run validation | Low | 3 | Volatility varies by session, but well-known effect may be priced in | 1, 2, 11 |
| 4 | Volatility | GARCH baseline | Use GARCH forecast as feature for RandomForest | Install arch library, implement GARCH, add as feature | High | 4 | GARCH built for volatility clustering, RF learns exceptions | 1, 2, 3, 15 |
| 5 | Volatility | IGARCH variant | GARCH with persistent shocks | Same as GARCH, different constraint | High | 3 | May fit forex better, but adds complexity | 4 |
| 6 | Trend Duration | Code cleanup | Remove config abstraction, split methods | Apply same fixes as volatility predictor | Medium | 2 | No edge gained, but enables proper testing | 7 |
| 7 | Trend Duration | Validation | Run same validation framework | Create feature file, add scenarios, run tests | Medium | 4 | Different prediction target, may have edge where volatility doesn't | 6 |
| 8 | Volatility | Code cleanup | Finish removing absurd config patterns | Replace remaining self.X_value with literals | Low | 1 | Maintainability only, no edge | - |
| 9 | Volatility, Trend | Rule-based approach | Simple rules: "BB squeeze + at support → expect move" | Define rules, backtest without ML | Medium | 3 | May outperform ML, simpler to understand and trust | 1, 2 |
| 10 | Volatility | Percentile threshold | Use high_vol_percentile instead of multiplier | Change label generation to use percentile | Low | 2 | Consistent class split across markets, but doesn't fix feature problem | 1, 2, 3, 4 |
| 11 | Volatility | Cyclical time encoding | Sine/Cosine of hour/day - "The Heartbeat" | Encode time as sin(2π*hour/24), cos(2π*hour/24) | Low | 4 | Captures market rhythm without discrete buckets, learns session patterns | 1, 2, 3 |
| 12 | Volatility | Efficiency Ratio (Kaufman) | Directness of move vs noise/path distance | ER = abs(close - close[n]) / sum(abs(close - close[1])) | Low | 4 | Measures trend quality, low ER = choppy = potential breakout setup | 1, 13, 14 |
| 13 | Volatility | Range Stability | Detect "unnatural" symmetry in bar sizes | Std dev of recent ranges / mean range | Low | 3 | Algorithmic control often shows as uniform ranges before breakout | 1, 12, 14 |
| 14 | Volatility | Bar Anatomy (Entropy) | Body-to-Range, Wick-to-Body ratios | Continuous measurements of price density and exhaustion | Low | 3 | Replaces subjective candlestick patterns with measurable physics | 12, 13 |
| 15 | Volatility | GARCH residuals | When price deviates from statistical expectation | Fit GARCH, use residuals as feature | High | 4 | Shows when actual vol differs from model expectation - anomaly detection | 4, 1 |
| 16 | Vasicek | Vasicek Model core | Mean-reversion model (Ornstein-Uhlenbeck) for forex | Implement dX = a(b-X)dt + σdW, calibrate parameters | High | 4 | Mathematical foundation for mean-reversion trading | 17, 18, 20 |
| 17 | Vasicek | Z-Score signals | Identify when price stretched too far from mean | Calculate Z = (price - mean) / σ, signal at ±2σ or ±3σ | Medium | 4 | Entry signal - trade only high-probability deviations | 16, 18 |
| 18 | Vasicek | Half-life exits | Trade duration based on mean-reversion speed | Exit after ln(2)/a periods (1-2 half-lives) | Low | 4 | Mathematically optimal exit timing | 16, 17 |
| 19 | Vasicek | Monte Carlo risk | Simulate 1000 scenarios for risk management | Generate paths, calculate VaR, protect from black swans | Medium | 3 | Ensures 20,000 CHF capital survives tail events | 16, 17, 18 |
| 20 | Vasicek | Historical calibration | Find model parameters from data | OLS regression to estimate a (speed), b (mean), σ (vol) | Medium | 5 | Required foundation - without calibration, model is useless | 16 |
| 21 | Vasicek | Pairs/spread trading | Trade spread between correlated assets | Apply Vasicek to price ratio or difference of two assets | High | 3 | Mean-reversion more reliable on spreads than single assets | 16, 17, 20 |

## Summary by Predictor

| Predictor | Items | Priority Work |
|-----------|-------|---------------|
| Volatility | 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14, 15 | 1 + 11 + 12 first (compression + time + efficiency) |
| Trend Duration | 6, 7 | 6 then 7 |
| Vasicek | 16, 17, 18, 19, 20, 21 | 20 first (calibration), then 16 + 17 + 18 |

## Feature Groups

### "Silence Before Storm" (Compression Detection)
- Item 1: BB squeeze, ATR compression
- Item 12: Efficiency Ratio (choppy = coiled)
- Item 13: Range Stability (algorithmic control)

### "Market Physics" (Intrinsic Price Behavior)
- Item 14: Bar Anatomy ratios
- Item 12: Efficiency Ratio
- Item 13: Range Stability

### "Market Rhythm" (Time-Based)
- Item 3: Calendar features (discrete)
- Item 11: Cyclical encoding (continuous)

### "Statistical Expectation" (GARCH Family)
- Item 4: GARCH baseline
- Item 5: IGARCH variant
- Item 15: GARCH residuals

### "Economic Gravity" (Vasicek Mean-Reversion)
- Item 16: Vasicek Model core
- Item 17: Z-Score signals
- Item 18: Half-life exits
- Item 19: Monte Carlo risk
- Item 20: Historical calibration
- Item 21: Pairs/spread trading

## Synergy: Combining Approaches

| Volatility Says | Vasicek Says | Action |
|-----------------|--------------|--------|
| LOW_VOL predicted | 3σ deviation | **Strong entry** - mean reversion works in calm markets |
| HIGH_VOL predicted | 3σ deviation | Wait - deviation may extend further |
| LOW_VOL predicted | No deviation | No trade - nothing to revert |
| HIGH_VOL predicted | No deviation | Prepare for breakout (compression features) |

## Suggested Sequence

### Path A: Volatility Improvement

**Phase 1: Low-Hanging Fruit** (Low effort, high potential)
- Item 11: Cyclical time encoding
- Item 12: Efficiency Ratio
- Item 1: Compression features
- Run validation, measure impact

**Phase 2: Expand if Phase 1 Shows Promise**
- Item 13: Range Stability
- Item 14: Bar Anatomy
- Item 2: Key levels

**Phase 3: Statistical Models**
- Item 4: GARCH baseline
- Item 15: GARCH residuals

### Path B: Trend Duration

- Item 6: Code cleanup
- Item 7: Validation with same framework

### Path C: Vasicek Mean-Reversion (New Direction)

**Phase 1: Foundation**
- Item 20: Historical calibration (OLS regression)
- Validate parameters on historical data

**Phase 2: Core Implementation**
- Item 16: Vasicek Model core
- Item 17: Z-Score signals
- Item 18: Half-life exits

**Phase 3: Risk Management**
- Item 19: Monte Carlo simulations
- Paper trading on demo account

**Phase 4: Advanced**
- Item 21: Pairs/spread trading

## Decision Framework

| If... | Then... |
|-------|---------|
| Volatility Phase 1 shows edge | Continue to Phase 2, defer Vasicek |
| Volatility Phase 1 fails | Start Vasicek Path C in parallel |
| Trend Duration shows edge | Combine with Volatility for regime filter |
| Vasicek calibration looks stable | Proceed to implementation |
| All ML approaches fail | Focus on Vasicek (mathematical, not ML) |