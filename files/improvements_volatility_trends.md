# Predictor Improvement Roadmap
**Date:** January 4, 2026 (Updated - Critical Bug Discovered)

## 🔥 CRITICAL STATUS - January 4, 2026

**BLOCKER:** VolatilityPredictor bug discovered - all ML validation results invalid

**Root Cause:** `_finalize_features()` return value ignored → Model trains on NaN data  

**Impact:**
- Baseline: 52.5% → 48.7% (worse than random)
- Item 22 results INVALID (showed 49.9%) - must re-test after fix
- All feature validation BLOCKED

**Fix:** 2-line code change + add ModelValidator class + regression tests

**Timeline:** Week 2 dedicated to bug fix before resuming features

---

## Current Situation

- Volatility predictor had validation framework, but catastrophic bug discovered Jan 4
- Items 11 & 12 results INVALID - bug existed since refactoring (NaN data trained model)
- **Item 22 (session overlap) BLOCKED - showed 49.9%, must re-test after bug fix**
- Trend duration predictor: 42.1% accuracy - **parked**
- **10 weeks ahead of schedule**
- **Priority:** Fix bug, add validation framework, then resume features

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
- **Mean Reversion Edge:** Small traders can wait for high-probability (3Ïƒ) deviations
- **Statistical Reality:** 53.6% Â± 2.1% std includes 50% - cannot prove better than random

### Realistic Expectations (20,000 CHF Capital)

- **Target:** 20% net annual return (4,000 CHF)
- **Gross needed:** ~25-28% to cover spreads, commissions, swap fees
- **Edge:** Agility to avoid noisy markets, only trade high-probability setups

## Work Items

| Nr | Predictor | What | Description | To Be Done | Effort | Success (1-5) | Reason | Best in Combination | Status |
|----|-----------|------|-------------|------------|--------|---------------|--------|---------------------|--------|
| 1 | Volatility | Compression features | BB squeeze, ATR compression, tight bars | Add features to volatility predictor, run validation | Medium | 4 | Compression precedes breakouts - direct precursor to volatility spikes. **Validated in Repka thesis (2013):** BB used for volatility measurement in profitable gold/cocoa strategies | 2, 3, 11, 12 | |
| 2 | Volatility | Key level features | Distance to high/low, round numbers | Add features to volatility predictor, run validation | Medium | 3 | Price at key levels often triggers volatility, but may be noisy | 1, 3, 11 | |
| 3 | Volatility | Calendar features | Hour, day of week, session (London/NY) | Add features to volatility predictor, run validation | Low | 3 | Volatility varies by session, but well-known effect may be priced in | 1, 2, 11 | |
| 4 | Volatility | GARCH baseline | Use GARCH forecast as feature for RandomForest | Install arch library, implement GARCH, add as feature | High | 4 | GARCH built for volatility clustering, RF learns exceptions | 1, 2, 3, 15 | |
| 5 | Volatility | IGARCH variant | GARCH with persistent shocks | Same as GARCH, different constraint | High | 3 | May fit forex better, but adds complexity | 4 | |
| 6 | Trend Duration | Code cleanup | Remove config abstraction, split methods | Apply same fixes as volatility predictor | Medium | 2 | No edge gained, but enables proper testing | 7 | **Done** |
| 7 | Trend Duration | Validation | Run same validation framework | Create feature file, add scenarios, run tests | Medium | 4 | Different prediction target, may have edge where volatility doesn't | 6 | **Done** - 42.1%, not useful |
| 8 | Volatility | Code cleanup | Finish removing absurd config patterns | Replace remaining self.X_value with literals | Low | 1 | Maintainability only, no edge | - | |
| 9 | Volatility, Trend | Rule-based approach | Simple rules: "BB squeeze + at support â†’ expect move" | Define rules, backtest without ML | Medium | 3 | May outperform ML, simpler to understand and trust | 1, 2 | |
| 10 | Volatility | Percentile threshold | Use high_vol_percentile instead of multiplier | Change label generation to use percentile | Low | 2 | Consistent class split across markets, but doesn't fix feature problem | 1, 2, 3, 4 | |
| 11 | Volatility | Cyclical time encoding | Sine/Cosine of hour/day - "The Heartbeat" | Encode time as sin(2Ï€*hour/24), cos(2Ï€*hour/24) | Low | 4 | Captures market rhythm without discrete buckets, learns session patterns | 1, 2, 3 | **Done** - +0.9% |
| 12 | Volatility | Efficiency Ratio (Kaufman) | Directness of move vs noise/path distance | ER = abs(close - close[n]) / sum(abs(close - close[1])) | Low | 4 | Measures trend quality, low ER = choppy = potential breakout setup | 1, 13, 14 | **Done** - +0.1% |
| 13 | Volatility | Range Stability | Detect "unnatural" symmetry in bar sizes | Std dev of recent ranges / mean range | Low | 3 | Algorithmic control often shows as uniform ranges before breakout | 1, 12, 14 | |
| 14 | Volatility | Bar Anatomy (Entropy) | Body-to-Range, Wick-to-Body ratios | Continuous measurements of price density and exhaustion | Low | 3 | Replaces subjective candlestick patterns with measurable physics | 12, 13 | |
| 15 | Volatility | GARCH residuals | When price deviates from statistical expectation | Fit GARCH, use residuals as feature | High | 4 | Shows when actual vol differs from model expectation - anomaly detection | 4, 1 | |
| 16 | Vasicek | Vasicek Model core | Mean-reversion model (Ornstein-Uhlenbeck) for forex | Implement dX = a(b-X)dt + ÏƒdW, calibrate parameters | High | 4 | Mathematical foundation for mean-reversion trading | 17, 18, 20 | |
| 17 | Vasicek | Z-Score signals | Identify when price stretched too far from mean | Calculate Z = (price - mean) / Ïƒ, signal at Â±2Ïƒ or Â±3Ïƒ | Medium | 4 | Entry signal - trade only high-probability deviations | 16, 18 | |
| 18 | Vasicek | Half-life exits | Trade duration based on mean-reversion speed | Exit after ln(2)/a periods (1-2 half-lives) | Low | 4 | Mathematically optimal exit timing | 16, 17 | |
| 19 | Vasicek | Monte Carlo risk | Simulate 1000 scenarios for risk management | Generate paths, calculate VaR, protect from black swans | Medium | 3 | Ensures 20,000 CHF capital survives tail events | 16, 17, 18 | |
| 20 | Vasicek | Historical calibration | Find model parameters from data | OLS regression to estimate a (speed), b (mean), Ïƒ (vol) | Medium | 5 | Required foundation - without calibration, model is useless | 16 | |
| 21 | Vasicek | Pairs/spread trading | Trade spread between correlated assets | Apply Vasicek to price ratio or difference of two assets | High | 3 | Mean-reversion more reliable on spreads than single assets | 16, 17, 20 | |
| 22 | Volatility | Session overlap binary | Binary flag for London/NY overlap (13:00-17:00 UTC) | Add binary feature, not sin/cos | Low | 3 | 70% of daily volume in this window - different signal than time encoding | 3, 11 | **New** |
| 23 | Volatility | Probability threshold | Use predict_proba with 70% threshold instead of 50% | Only trade high-confidence predictions | Low | 4 | Fewer trades, higher quality - filters noise | All | **New** |
| 24 | Volatility | ATR Ratio (relative) | ATR(current) / ATR(100 days) - stationary feature | Replace raw ATR with relative version | Low | 4 | RF handles relative values better than absolute | 1, 13 | **New** |
| 25 | Volatility | Event distance | Hours until next NFP/CPI/FOMC | Manual calendar or simple scraper from ForexFactory | Medium | 4 | Leading indicator - you know volatility is coming | 3, 22 | **New** |
| 26 | Volatility | VIX cross-asset | VIX level, VIX change, VIX > 20 flag | Download from Yahoo Finance, add as features | Low | 2 | Risk-off sentiment spillover - but likely arbitraged | 25 | **New** |
| 27 | Volatility | DXY extreme | DXY at multi-month high/low flag | Download from Yahoo Finance, detect extremes | Low | 2 | USD strength affects EUR/USD vol - but likely arbitraged | 26 | **New** |

## Summary by Predictor

| Predictor | Items | Priority Work | Status |
|-----------|-------|---------------|--------|
| Volatility | 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14, 15, 22-27 | 22 + 23 + 24 first (session + threshold + ATR ratio) | 53.6% - no proven edge |
| Trend Duration | 6, 7 | Complete | **Parked** - 42.1% not useful |
| Vasicek | 16, 17, 18, 19, 20, 21 | 20 first (calibration), then 16 + 17 + 18 | Not started |

## Feature Groups

### "Silence Before Storm" (Compression Detection)
- Item 1: BB squeeze, ATR compression
- Item 12: Efficiency Ratio (choppy = coiled) âœ“ Done
- Item 13: Range Stability (algorithmic control)
- Item 24: ATR Ratio (relative volatility) **New**

### "Market Physics" (Intrinsic Price Behavior)
- Item 14: Bar Anatomy ratios
- Item 12: Efficiency Ratio âœ“ Done
- Item 13: Range Stability

### "Market Rhythm" (Time-Based)
- Item 3: Calendar features (discrete)
- Item 11: Cyclical encoding (continuous) âœ“ Done
- Item 22: Session overlap binary **New**

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

### "External Context" (Cross-Asset & Calendar) **New Group**
- Item 25: Event distance (NFP/CPI/FOMC)
- Item 26: VIX cross-asset
- Item 27: DXY extreme

### "Trading Execution" (Not Prediction) **New Group**
- Item 23: Probability threshold (70% confidence filter)

## Synergy: Combining Approaches

| Volatility Says | Vasicek Says | Action |
|-----------------|--------------|--------|
| LOW_VOL predicted | 3Ïƒ deviation | **Strong entry** - mean reversion works in calm markets |
| HIGH_VOL predicted | 3Ïƒ deviation | Wait - deviation may extend further |
| LOW_VOL predicted | No deviation | No trade - nothing to revert |
| HIGH_VOL predicted | No deviation | Prepare for breakout (compression features) |

## Suggested Sequence (Updated)

### Path A: Volatility Improvement

**Phase 1: Quick Wins** (Low effort, different approach)
- Item 22: Session overlap binary
- Item 23: Probability threshold (70%)
- Item 24: ATR Ratio (relative)
- Run validation, measure impact

**Phase 2: Compression** (Original plan)
- Item 1: BB squeeze, ATR compression
- Item 13: Range Stability
- Item 14: Bar Anatomy

**Phase 3: Leading Indicators**
- Item 25: Event distance
- Item 26: VIX (skeptical but worth 1 day)

**Phase 4: Statistical Models** (High effort)
- Item 4: GARCH baseline
- Item 15: GARCH residuals

### Path B: Trend Duration - **PARKED**

- Item 6: Code cleanup âœ“ Done
- Item 7: Validation âœ“ Done - 42.1% not useful
- **Decision: Not worth further investment**

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
| Volatility Phase 1 shows edge (>57%) | Continue to Phase 2 |
| Volatility Phase 1 still ~53% | Pivot to Vasicek |
| Vasicek calibration looks stable | Proceed to implementation |
| All approaches fail | Focus purely on Vasicek (math, not ML) |

## Progress Tracking

| Item | Estimated | Actual | Result |
|------|-----------|--------|--------|
| 11 - Cyclical time | 10h | 2h | +0.9% |
| 12 - Efficiency Ratio | 10h | 2h | +0.1% |
| 6 - Trend cleanup | 10h | 3h | Done |
| 7 - Trend validation | 10h | 2h | 42.1% - not useful |

**Total buffer gained: ~30 hours (10 weeks ahead of schedule)**

## Key Insight

The fundamental problem may not be *features* but *target*:
- Predicting "will volatility regime change" might be too hard
- Alternative: Don't predict, **react** - use Vasicek to trade when price is already at extremes
- The edge isn't in prediction, it's in **patience** (wait for 3Ïƒ)
---

## Historical Validation: Repka Thesis (2013)

**Source:** "Investment Models in Financial Markets" - Master's thesis on automated trading systems

### What Worked (46.74% return in Q1 2013)

**Successful portfolio of 4 strategies:**
1. **CCTA01** (Cocoa) - Moving Averages + RSI: +$1,528.8
2. **GCTA03** (Gold) - Multiple technical indicators: +$1,360
3. **GCBO02** (Gold) - Breakout strategy: +$962
4. **CCBO01** (Cocoa) - Breakout strategy: +$823

### Technical Analysis Components Validated

| Component | Our Bot | Thesis Result |
|-----------|---------|---------------|
| **ATR-based stops** | ✓ Implemented | ✓ Used in all 4 strategies |
| **Bollinger Bands** | Item 1 (planned) | ✓ Volatility measurement |
| **Intraday-only trading** | Risk item (planned) | ✓ All positions closed daily |
| **Moving Averages** | Not used | ✓ CCTA01 (18% win rate, 7.11 RRR) |
| **RSI oscillator** | Not used | ✓ Combined with MA in CCTA01 |
| **Breakout approach** | Not used | ✓ CCBO01, GCBO02 strategies |
| **Portfolio diversification** | Multi-bot roadmap | ✓ 2 markets × 2 methods |

### Key Takeaways for Our Project

**1. Bollinger Band Squeeze Works**
- Thesis used BB for compression → expansion detection
- Confirms Item 1 rationale ("Silence Before Storm")
- **Action:** Prioritize BB squeeze implementation

**2. Win Rate ≠ Profitability**
- CCTA01: Only 18% win rate, but 7.11 reward:risk ratio = profitable
- Validates focus on edge, not accuracy percentage
- **Reminder:** Our 53.6% might still be profitable with proper money management

**3. Out-of-Sample Validation Critical**
- Thesis split: 2011-mid2012 (train), mid2012-end2012 (OOS), Q1 2013 (final test)
- Matches our approach: train/test split mandatory
- **Warning:** Author emphasizes results need live market confirmation

**4. Sensitivity Analysis on Risk**
- Tested stop-loss levels from $115-$260
- Found optimal risk per trade
- **Action:** Use our SensitivityAnalyzer similarly (already planned)

**5. Monthly Variance Reality**
- Jan 2013: +$3,320 (33.2%)
- Feb 2013: -$72 (-0.7%)
- Mar 2013: +$1,425.8 (14.3%)
- **Expectation:** Our 20% annual target will have volatile months

### What We're Doing Differently (Better)

| Aspect | Thesis (2013) | Our Bot (2026) |
|--------|---------------|----------------|
| **Overfitting detection** | Visual inspection | RobustnessAnalyzer (plateau vs peak) |
| **Optimization** | Genetic algorithms | Grid search + robustness checks |
| **Prediction** | Technical indicators only | ML (RandomForest) + Math (Vasicek) |
| **Risk management** | ATR stops only | ATR + economic calendar + grey swan detection |
| **Validation honesty** | Good (acknowledged limits) | **Excellent** (53.6% ≈ 50%, no edge proven yet) |

### Philosophical Alignment

Both cautious about backtest results:
- ✓ Proper train/test splits
- ✓ ATR-based risk management  
- ✓ Out-of-sample validation
- ✓ Realistic expectations
- ✓ Demo testing before live money

**Difference:** Thesis relied on technical indicators alone; we're adding ML + mathematical models (Vasicek) while maintaining same rigor.