# Predictor Development Roadmap - Gantt Chart

**Start Date:** January 2026  
**Capacity:** 10 hours/week  
**Goal:** Implement Vasicek triangular arbitrage (proven edge), park Volatility ML  
**Last Updated:** January 10, 2026 (Week 2 - Vasicek Priority Pivot)

## 🔥 STRATEGIC PIVOT - January 10, 2026

**Volatility ML → Vasicek Mean-Reversion**

| Approach | Result | Status |
|----------|--------|--------|
| Volatility ML | 53.6% ± 2.1% (includes 50% = no proven edge) | ⏸️ Parked |
| Vasicek/Triangular | p=0.0000, half-life 0.7 days | ✅ **PROVEN** |

**Why pivot:** Mathematical identity (EUR/USD = EUR/GBP × GBP/USD) beats ML guessing.

---

## Progress Summary

| Status | Bar | Icon | Items |
|--------|-----|------|-------|
| Done | ███ | ✅ | 6, 11, 12, 39, 20 (Vasicek validation) |
| Rejected | ⊗⊗⊗ | ❌ | 22 (Session overlap -0.8%) |
| Parked | ░░░ | ⏸️ | 7, 40, 41, 42, 1-5 (Volatility work) |
| In Progress | ▓▓▓ | 🔥 | 43-51 (Vasicek implementation) |
| Planned | ▓▓▓ | ▶️ | Paper trading, live demo |
| Buffer | | ⏳ | 10 weeks ahead of schedule |

## Test Status

```
Total: 252 tests
✅ Passing: 202 (80.2%)
🔴 Failing: 50 (19.8%) - New Vasicek components not yet implemented
⚠️ Warnings: 79

Failing breakdown:
- vasicek_model.feature: ~20 tests
- triangular_arbitrage_predictor.feature: ~15 tests
- triangular_strategy.feature: ~15 tests
```

---

## Timeline

```
2026        Jan                 Feb                 Mar                 Apr                 May
Week        1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20
            ─────────────────────────────────────────────────────────────────────────────────
COMPLETED (Week 1)
Item 11     ███                                                                             ✅ Cyclical time (+0.9%)
Item 12     ███                                                                             ✅ Efficiency Ratio (+0.1%)
Item 6      ███                                                                             ✅ Trend cleanup
Item 7      ░░░                                                                             ⏸️ Trend validation (42.1% - parked)
Item 22     ⊗⊗⊗                                                                             ❌ Session overlap (-0.8% - rejected)
Item 39     ███                                                                             ✅ Fix NaN bug (4h)
Item 20     ███                                                                             ✅ Vasicek validation (p=0.0000)
            ─────────────────────────────────────────────────────────────────────────────────
VASICEK IMPLEMENTATION (Week 2-3) 🔥 ★ CURRENT FOCUS
Item 43         ▓▓▓                                                                         🔥 TriangularSpreadCalculator (Day 1-2)
Item 50         ▓▓▓                                                                         🔥 Test fixtures (Day 1-2)
Item 44         ▓▓▓▓                                                                        🔥 VasicekModel (Day 3-4)
Item 47         ▓▓▓                                                                         🔥 vasicek_model.feature steps (Day 5)
Item 45             ▓▓▓▓                                                                    ▶️ TriangularArbitragePredictor (Week 3)
Item 48             ▓▓▓                                                                     ▶️ predictor.feature steps (Week 3)
Item 46             ▓▓▓▓▓▓                                                                  ▶️ TriangularStrategy (Week 3)
Item 49             ▓▓▓▓                                                                    ▶️ strategy.feature steps (Week 3)
Item 51             ▓▓▓                                                                     ▶️ Fix 50 failing tests (Week 3)
            ─────────────────────────────────────────────────────────────────────────────────
CONFIGURATION & DOCS (Week 4)
Config                  ▓▓▓                                                                 ▶️ predictors.json profiles
Docs                    ▓▓▓                                                                 ▶️ Architecture documentation
            ─────────────────────────────────────────────────────────────────────────────────
PAPER TRADING (Week 5-6)
Paper                       ▓▓▓▓▓▓▓                                                         ▶️ Demo account validation
            ─────────────────────────────────────────────────────────────────────────────────
LIVE DEMO (Week 7+)
Demo                                ▓▓▓▓▓▓▓                                                 ▶️ €100 live test (3 months)
            ─────────────────────────────────────────────────────────────────────────────────
PARKED (Volatility ML) ░░░
Item 42     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ⏸️ Error Analysis Dashboard
Item 40     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ⏸️ ModelValidator class
Item 41     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ⏸️ Regression guard tests
Items 1-5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ⏸️ Compression, GARCH, etc.
            ─────────────────────────────────────────────────────────────────────────────────
MILESTONE   ███─────────────────●───────●───────●───────────────────────────────────────────
            D0                  V1      V2      V3
```

**Legend:** ███ = Done ✅ | ▓▓▓ = In Progress/Planned 🔥▶️ | ░░░ = Parked ⏸️ | ⊗⊗⊗ = Rejected ❌ | ★ = Critical

---

## Detailed Schedule

| Week | Dates | Items | Focus | Hours | Status |
|------|-------|-------|-------|-------|--------|
| 1 | Jan 1-5 | 6, 7, 11, 12, 22, 39, 20 | Volatility features, NaN bug, **Vasicek validation** | 13.5h | ✅ **D0: Vasicek proven** |
| 2 | Jan 6-12 | 43, 44, 47, 50 | **Vasicek core:** Calculator, Model, fixtures, BDD steps | 10h | 🔥 **IN PROGRESS** |
| 3 | Jan 13-19 | 45, 46, 48, 49, 51 | **Vasicek complete:** Predictor, Strategy, fix 50 tests | 10h | ▶️ **V1: All tests pass** |
| 4 | Jan 20-26 | Config, Docs | Configuration profiles, architecture docs | 10h | ▶️ |
| 5-6 | Jan 27-Feb 9 | Paper | Paper trading validation | 20h | ▶️ **V2: Paper validated** |
| 7+ | Feb 10+ | Demo | €100 live demo (3 months) | - | ▶️ **V3: Live demo** |

---

## Decision Points

### D0: Vasicek Validation (Week 1 - Jan 5) ✅

**What happened:**
- Volatility ML: 53.6% ± 2.1% = no proven edge
- Vasicek ADF test: p=0.0000 = **PROVEN stationary**
- EUR/USD spread half-life: 0.7 days = fast reversion
- Architecture designed: Calculator → Model → Predictor → Strategy
- BDD features created: 3 comprehensive test files

| Result | Action |
|--------|--------|
| **Vasicek validated** | ✅ Pivot to implementation |
| **Volatility no edge** | ⏸️ Park all ML work |
| 50 new tests failing | Implement components to pass |
| 10 weeks buffer | Use for Vasicek implementation |

### V1: All Tests Pass (Week 3 - Jan 19)

| Result | Action |
|--------|--------|
| 252/252 tests passing | Proceed to paper trading |
| Calibration issues | Review OLS, check fixtures |
| Tests still failing | Debug, extend Week 3 |

### V2: Paper Trading Validated (Week 6 - Feb 9)

| Result | Action |
|--------|--------|
| Consistent signals, no bugs | Proceed to live demo |
| Execution issues | Fix before live money |
| Edge not materializing | Review calibration, thresholds |

### V3: Live Demo Success (Week 7+ - 3 months)

| Result | Action |
|--------|--------|
| 3 months profitable | Scale to €1,000 (Phase 1) |
| Breakeven or small loss | Continue demo, analyze |
| Significant loss | Review strategy, may abort |

---

## Work Items

### Priority 0: Vasicek Implementation (CURRENT FOCUS) 🔥

| Nr | Component | What | Description | Effort | Status |
|----|-----------|------|-------------|--------|--------|
| 43 | Calculator | TriangularSpreadCalculator | Pure math: spread = actual - synthetic | Low | 🔥 Week 2 |
| 44 | Model | VasicekModel | O-U process: calibrate κ, θ, σ from spread | Medium | 🔥 Week 2 |
| 45 | Predictor | TriangularArbitragePredictor | Integrate calculator + Vasicek, signals | Medium | ▶️ Week 3 |
| 46 | Strategy | TriangularStrategy | 3-leg atomic execution, risk mgmt | High | ▶️ Week 3 |
| 47 | Tests | vasicek_model.feature steps | BDD test steps for Vasicek | Medium | 🔥 Week 2 |
| 48 | Tests | triangular_arbitrage_predictor.feature steps | BDD test steps for predictor | Medium | ▶️ Week 3 |
| 49 | Tests | triangular_strategy.feature steps | BDD test steps for strategy | Medium | ▶️ Week 3 |
| 50 | Fixtures | Test data generation | Synthetic spreads + real market data | Low | 🔥 Week 2 |
| 51 | Tests | Fix 50 failing tests | Implement components to pass tests | High | ▶️ Week 2-3 |

### Priority 1: Volatility Improvements (PARKED)

| Nr | What | Description | Status |
|----|------|-------------|--------|
| 1 | Compression features | BB squeeze, ATR compression | ⏸️ Parked |
| 2 | Key level features | Distance to high/low, round numbers | ⏸️ Parked |
| 3 | Calendar features | Hour, day of week, session | ⏸️ Parked |
| 4 | GARCH baseline | GARCH forecast as feature | ⏸️ Parked |
| 5 | IGARCH variant | GARCH with persistent shocks | ⏸️ Parked |
| 40 | ModelValidator class | Prevent future NaN bugs | ⏸️ Parked |
| 41 | Regression guard tests | Catch performance degradation | ⏸️ Parked |
| 42 | Error Analysis Dashboard | Where/when predictions fail | ⏸️ Parked |

### Completed

| Nr | What | Result | Hours | Status |
|----|------|--------|-------|--------|
| 6 | Trend cleanup | Code maintainable | 3h | ✅ |
| 7 | Trend validation | 42.1% - not useful | 2h | ⏸️ Parked |
| 11 | Cyclical time encoding | +0.9% accuracy | 2h | ✅ |
| 12 | Efficiency Ratio | +0.1% accuracy | 2h | ✅ |
| 20 | Vasicek validation | p=0.0000, half-life 0.7 days | 3h | ✅ |
| 22 | Session overlap | -0.8% (degrades) | 1.5h | ❌ Rejected |
| 39 | Fix NaN bug | Baseline restored 52.5% | 4h | ✅ |

---

## Week 2 Implementation Plan 🔥

### Day 1-2: Pure Math (Items 43, 50)

```
✓ Create TriangularSpreadCalculator
  - calculate_synthetic_price(eur_gbp, gbp_usd) → synthetic EUR/USD
  - calculate_spread(actual, synthetic) → spread value
  - calculate_spread_series(df) → spread time series
  - calculate_statistics() → mean, std, min, max

✓ Generate test fixtures
  - Synthetic mean-reverting spreads (known κ, θ, σ)
  - Real EUR/USD spread data (1000+ points)
  - Non-stationary data (trending, random walk) for negative tests
```

### Day 3-4: Vasicek Model (Item 44)

```
✓ Create VasicekModel
  - calibrate(spread_series) → OLS regression for κ, θ, σ
  - calculate_z_score(current_spread) → z-score
  - predict_next_value() → expected spread
  - is_mean_reverting() → bool (ADF test)
  - get_trading_threshold(sigma_multiple) → entry/exit levels
  - calculate_half_life() → periods to mean
```

### Day 5: BDD Tests (Item 47)

```
✓ Implement all vasicek_model.feature step definitions
✓ Run tests: Expect ~30 tests to pass
✓ Fix any calibration issues
✓ Target: 232/252 passing (from 202)
```

---

## Summary by Focus Area

| Focus Area | Items | Priority | Status |
|------------|-------|----------|--------|
| **Vasicek Implementation** | 43-51 | **P0 - CRITICAL** | 🔥 50 tests failing |
| Volatility ML | 1-5, 40-42 | P1 - Secondary | ⏸️ Parked (no proven edge) |
| Trend Duration | 6, 7 | - | ✅ Complete (not useful) |
| Infrastructure | 39 | - | ✅ NaN bug fixed |

---

## Total Investment (Revised)

| Phase | Weeks | Hours | Status |
|-------|-------|-------|--------|
| Week 1 (completed) | 1 | 13.5h | ✅ Vasicek validated |
| Week 2 (Vasicek core) | 1 | 10h | 🔥 IN PROGRESS |
| Week 3 (Vasicek complete) | 1 | 10h | ▶️ |
| Week 4 (Config & Docs) | 1 | 10h | ▶️ |
| Paper trading | 2 | 20h | ▶️ |
| **Subtotal to paper** | **6** | **63.5h** | |
| Live demo | 12 | - | ▶️ (3 months) |

**Target:** Paper trading by Feb 9, Live demo by Feb 10

---

## 10-Year Plan Alignment

### Phase 1 (Years 1-5): Testing & Growing
- Starting: €20k capital
- Add: €10-15k/year savings
- Target: €150-200k by Year 5
- Return needed: 25-30% annually
- **Current:** Vasicek implementation = foundation

### Phase 2 (Year 6+): Financial Independence
- Unlock: €1M pension (early withdrawal)
- Total: €1.15-1.20M
- Deploy: 60% safe, 40% trading
- Income: €100-126k/year
- Living: €40-50k/year (Switzerland)
- Result: **QUIT JOB** ✅

---

## Risk Management Checklist

### Before Paper Trading (Week 5)

- [ ] TriangularSpreadCalculator tested
- [ ] VasicekModel calibration validated
- [ ] TriangularArbitragePredictor generating signals
- [ ] TriangularStrategy 3-leg execution working
- [ ] All 252 tests passing
- [ ] Z-score thresholds configured (2σ entry, 0.5σ exit)

### Before Live Demo (Week 7)

- [ ] Paper trading shows consistent signals
- [ ] No execution bugs
- [ ] Risk limits coded (max position, max exposure)
- [ ] Rollback mechanism tested
- [ ] Half-life monitoring active

---

## Key Insights

### Why Vasicek Over Volatility ML

| Volatility ML | Vasicek |
|---------------|---------|
| 53.6% ± 2.1% (includes 50%) | p=0.0000 (statistically proven) |
| No mathematical guarantee | EUR/USD = EUR/GBP × GBP/USD (identity) |
| Overfitting risk | Mean reversion is structural |
| Cannot prove edge | Known edge: spread MUST revert |

### The "Poor Man's" Advantage

- **Small trader can wait** for 2σ+ deviations
- **No pressure** for high frequency
- **Low competition** at these extremes
- **Mathematical edge** not speed edge

---

## Reference Materials

### Statistics and Risk Modeling - YouTube Channel
**Source:** https://www.youtube.com/@statisticsandriskmodeling5477

**Key videos:**
- Ornstein-Uhlenbeck Process Simulation: https://www.youtube.com/watch?v=dV23py1ISs0
- Vasicek Bond Pricing in Python: https://www.youtube.com/watch?v=j8Y3TCzbVa0

**Used for:**
- Items 43-50: Vasicek implementation
- OLS parameter estimation (κ, θ, σ calibration)
- Validation framework

---

**Status:** Week 2 Vasicek implementation in progress. Target: All 252 tests passing by Jan 19. 🎯