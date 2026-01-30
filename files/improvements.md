# Predictor Improvement Roadmap
**Date:** January 10, 2026 (Updated - Vasicek Priority Shift)

## 🔥 CRITICAL UPDATE - January 10, 2026

**STRATEGIC PIVOT: Volatility → Vasicek Mean-Reversion**

**What changed:**
- Volatility ML: 53.6% accuracy = no proven edge (within noise of 50%)
- Vasicek approach: Mathematical edge via triangular arbitrage
- EUR/USD spread: **p=0.0000, half-life 0.7 days** ✅ PROVEN stationary
- Strategy validated: Real mathematical relationship (not ML guessing)

**Resolution:**
- ✅ Vasicek model validated on real data
- ✅ Architecture designed (Calculator → Model → Predictor → Strategy)
- ✅ BDD features created (vasicek_model, triangular_arbitrage_predictor, triangular_strategy)
- 🔄 Implementation in progress (50 tests failing - new components not yet built)

**New priority:**
1. **Vasicek implementation** (Items 43-50) - 10-year plan Phase 1 foundation
2. Fix failing tests (Item 51)
3. Volatility improvements (Items 1-5, 42) - Secondary

**10-year plan alignment:**
- Phase 1 (Years 1-5): Test triangular arbitrage with €20k → €150-200k
- Phase 2 (Year 6+): Deploy €1M pension with proven strategies
- Result: Financial independence at €100-120k/year income

---

## Current Situation

**Vasicek Mean-Reversion (NEW FOCUS):**
- ✅ Mathematical validation: EUR/USD spread p=0.0000, half-life 0.7 days
- ✅ Architecture designed: Clean separation (Calculator/Model/Predictor/Strategy)
- ✅ BDD features created: 3 comprehensive feature files
- ⏳ Implementation needed: Components + tests
- 🎯 **This is the proven edge** - not ML guessing

**Volatility ML (SECONDARY):**
- Baseline: 52.5% accuracy
- Best config: 53.6% (time_efficiency)
- Statistical reality: 53.6% ± 2.1% includes 50% = no proven edge
- Item 22 (session overlap): REJECTED (-0.8%)
- Status: **Parked pending Vasicek completion**

**Trend Duration:**
- 42.1% accuracy - **permanently parked**

**Testing:**
- 202 tests passing ✅
- 50 tests failing 🔴 (new Vasicek components not yet implemented)
- Test infrastructure working
- Need: Implement components to fix failing tests

**Timeline:**
- **10 weeks ahead of schedule** (from volatility work)
- Using buffer for Vasicek implementation
- Target: Complete Vasicek by end of January 2026

---

## Philosophy: The "Poor Man's" Trading Approach

### Core Principles

1. **Avoid the Quant War** - Don't compete on speed or data volume against institutional players
2. **Eat the Breadcrumbs** - Find moments of market inefficiency that large firms ignore due to their size
3. **Intrinsic Data Only** - Data must speak for itself, avoid external correlations
4. **Mathematical Edges** - Triangular arbitrage: structural relationship, not pattern guessing ⭐ NEW
5. **Economic Gravity** - Prices stretched too far from mean get pulled back (Vasicek)

### Key Realizations

- **The ML Trap:** 53.6% might be noise, not signal
- **Mathematical Truth:** EUR/USD = EUR/GBP × GBP/USD (identity, not correlation)
- **Proven Stationarity:** p=0.0000 on ADF test, half-life 0.7 days
- **Small Trader Advantage:** Can wait for 2σ deviations, no pressure for high frequency
- **10-Year Plan:** Test with €20k before risking €1M pension (Years 1-5 validation)

### Realistic Expectations (€20k Starting Capital)

**Phase 1 (Years 1-5): Testing & Growing**
- Starting: €20k capital
- Add: €10-15k/year savings
- Target: €150-200k by Year 5
- Return needed: 25-30% annually
- Purpose: PROVE strategies work

**Phase 2 (Year 6+): Financial Independence**
- Unlock: €1M pension (early withdrawal)
- Total: €1.15-1.20M (Phase 1 + pension)
- Deploy: 60% safe, 40% trading
- Income: €100-126k/year
- Living: €40-50k/year (Switzerland)
- Result: **QUIT JOB** ✅

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
- GitHub code reference

---

## Work Items

### Priority 1: Vasicek Implementation (CURRENT FOCUS)

| Nr | Component | What | Description | Effort | Status |
|----|-----------|------|-------------|--------|--------|
| 43 | Calculator | TriangularSpreadCalculator | Pure math: spread = actual - synthetic | Low | **Week 2** |
| 44 | Model | VasicekModel | O-U process: calibrate κ, θ, σ from spread | Medium | **Week 2** |
| 45 | Predictor | TriangularArbitragePredictor | Integrate calculator + Vasicek, generate signals | Medium | **Week 2** |
| 46 | Strategy | TriangularStrategy | 3-leg atomic execution, risk management | High | **Week 3** |
| 47 | Tests | vasicek_model.feature steps | Implement BDD test steps for Vasicek | Medium | **Week 2** |
| 48 | Tests | triangular_arbitrage_predictor.feature steps | Implement BDD test steps for predictor | Medium | **Week 3** |
| 49 | Tests | triangular_strategy.feature steps | Implement BDD test steps for strategy | Medium | **Week 3** |
| 50 | Fixtures | Test data generation | Generate synthetic spreads + real market data | Low | **Week 2** |
| 51 | Tests | Fix 50 failing tests | Implement missing components to pass tests | High | **Week 2-3** |

### Priority 2: Volatility Improvements (SECONDARY)

| Nr | Predictor | What | Description | Effort | Status |
|----|-----------|------|-------------|--------|--------|
| 1 | Volatility | Compression features | BB squeeze, ATR compression, tight bars | Medium | On hold |
| 2 | Volatility | Key level features | Distance to high/low, round numbers | Medium | On hold |
| 3 | Volatility | Calendar features | Hour, day of week, session (London/NY) | Low | On hold |
| 4 | Volatility | GARCH baseline | Use GARCH forecast as feature | High | On hold |
| 5 | Volatility | IGARCH variant | GARCH with persistent shocks | High | On hold |
| 42 | Volatility | Error Analysis Dashboard | Where/when predictions fail | Medium | On hold |

### Infrastructure (Completed)

| Nr | Component | What | Status |
|----|-----------|------|--------|
| 6 | Trend Duration | Code cleanup | ✅ Done |
| 7 | Trend Duration | Validation | ✅ Done (42.1% - parked) |
| 11 | Volatility | Cyclical time encoding | ✅ Done (+0.9%) |
| 12 | Volatility | Efficiency Ratio | ✅ Done (+0.1%) |
| 22 | Volatility | Session overlap | ❌ Rejected (-0.8%) |
| 39 | Volatility | Fix NaN bug | ✅ Done |

---

## Summary by Focus Area

| Focus Area | Items | Priority | Status |
|------------|-------|----------|--------|
| **Vasicek Implementation** | 43-51 | **P0 - CRITICAL** | 🔴 50 tests failing |
| Volatility ML | 1-5, 42 | P1 - Secondary | ⏸️ Parked (no proven edge) |
| Trend Duration | 6, 7 | - | ✅ Complete (not useful) |
| Infrastructure | 39-41 | P1 | ✅ NaN bug fixed |

---

## Detailed Implementation Plan

### Week 2: Vasicek Core Components

**Day 1-2: Pure Math (Items 43, 50)**
```
✓ Create TriangularSpreadCalculator
  - calculate_synthetic_price()
  - calculate_spread()
  - calculate_spread_series()
  - calculate_statistics()

✓ Generate test fixtures
  - Synthetic mean-reverting spreads (known parameters)
  - Real EUR/USD spread data
  - Non-stationary data (trending, random walk)
```

**Day 3-4: Vasicek Model (Item 44)**
```
✓ Create VasicekModel
  - calibrate() - OLS regression for κ, θ, σ
  - calculate_z_score()
  - predict_next_value()
  - is_mean_reverting()
  - get_trading_threshold()

✓ Implement test steps for vasicek_model.feature
  - Calibration scenarios
  - Z-score calculations
  - Half-life tests
  - Mean reversion validation
```

**Day 5: BDD Tests (Item 47)**
```
✓ Implement all vasicek_model.feature step definitions
✓ Run tests: Expect ~30 tests to pass
✓ Fix any calibration issues
```

### Week 3: Predictor & Strategy

**Day 1-2: Predictor (Item 45)**
```
✓ Create TriangularArbitragePredictor
  - calibrate() - uses calculator + Vasicek
  - predict() - generates LONG_SPREAD/SHORT_SPREAD/CLOSE/HOLD
  - get_required_markets()
  - State management

✓ Implement test steps for triangular_arbitrage_predictor.feature
```

**Day 3-4: Strategy (Item 46)**
```
✓ Create TriangularStrategy
  - 3-leg atomic position opening
  - 3-leg atomic position closing
  - Rollback on failure
  - Z-score based stop loss
  - Trade recording

✓ Implement test steps for triangular_strategy.feature
```

**Day 5: Integration Testing (Item 51)**
```
✓ Run full test suite
✓ Fix remaining failing tests
✓ Validate all 3 components work together
✓ Target: 252 passing, 0 failing
```

### Week 4: Configuration & Documentation

**Configuration Integration**
```
✓ Add vasicek_prediction to predictors.json
  - standard profile (entry: 2.0σ, exit: 0.5σ)
  - aggressive profile (entry: 1.5σ, exit: 0.3σ)
  - conservative profile (entry: 2.5σ, exit: 0.8σ)

✓ Add test_config.json with fixtures
```

**Documentation**
```
✓ Update roadmap.md with Vasicek items
✓ Document architecture decisions
✓ Add usage examples
```

---

## Test Status Tracking

### Current Test Status (January 10, 2026)

```
Total: 252 tests
✅ Passing: 202 (80.2%)
🔴 Failing: 50 (19.8%)
⚠️ Warnings: 79

Failing tests breakdown:
- vasicek_model.feature: ~20 tests (component not implemented)
- triangular_arbitrage_predictor.feature: ~15 tests (component not implemented)
- triangular_strategy.feature: ~15 tests (component not implemented)
```

### Expected Test Status After Week 2

```
Total: 252 tests
✅ Passing: 232 (92.1%)
🔴 Failing: 20 (7.9%)

Remaining failures:
- triangular_arbitrage_predictor.feature: ~10 tests
- triangular_strategy.feature: ~10 tests
```

### Target Test Status After Week 3

```
Total: 252 tests
✅ Passing: 252 (100%)
🔴 Failing: 0 (0%)

All components implemented and tested ✅
```

---

## Decision Framework

| If... | Then... |
|-------|---------|
| Week 2 tests pass (Vasicek + Calculator) | Continue to Week 3 (Predictor + Strategy) |
| Calibration issues detected | Review OLS implementation, check fixtures |
| Week 3 tests pass (all components) | Begin paper trading preparation |
| Paper trading shows consistent edge | Move to live demo (€100) |
| Demo successful for 3 months | Scale to €1,000 (Phase 1 start) |

---

## Progress Tracking

### Completed Items

| Item | Component | Estimated | Actual | Result |
|------|-----------|-----------|--------|--------|
| 11 | Cyclical time | 10h | 2h | +0.9% ✅ |
| 12 | Efficiency Ratio | 10h | 2h | +0.1% ✅ |
| 6 | Trend cleanup | 10h | 3h | Done ✅ |
| 7 | Trend validation | 10h | 2h | 42.1% (parked) |
| 22 | Session overlap | 10h | 1.5h | -0.8% ❌ |
| 39 | Fix NaN bug | 10h | 4h | Fixed ✅ |
| 20 | Vasicek validation | 5h | 3h | p=0.0000 ✅ |

**Total buffer: ~30 hours (10 weeks ahead)**

### In Progress (Week 2)

| Item | Component | Estimated | Target |
|------|-----------|-----------|--------|
| 43 | TriangularSpreadCalculator | 5h | Day 1-2 |
| 44 | VasicekModel | 10h | Day 3-4 |
| 47 | vasicek_model.feature steps | 10h | Day 5 |
| 50 | Test fixtures | 5h | Day 1-2 |

### Planned (Week 3)

| Item | Component | Estimated | Target |
|------|-----------|-----------|--------|
| 45 | TriangularArbitragePredictor | 10h | Day 1-2 |
| 48 | predictor.feature steps | 5h | Day 2 |
| 46 | TriangularStrategy | 15h | Day 3-4 |
| 49 | strategy.feature steps | 10h | Day 4-5 |
| 51 | Fix failing tests | 10h | Day 5 |

---

## Key Insights

### Why Vasicek Over Volatility ML

**Volatility ML Problems:**
- 53.6% accuracy might be random (within 2.1% std of 50%)
- No mathematical guarantee
- Overfitting risk
- Cannot prove edge exists

**Vasicek Advantages:**
- Mathematical identity: EUR/USD = EUR/GBP × GBP/USD
- Proven stationarity: p=0.0000, half-life 0.7 days
- Known edge: spread MUST revert (arbitrage enforced)
- Predictable behavior: O-U process well-studied
- Small trader advantage: Can wait for 2σ+ deviations

### 10-Year Plan Validation

**Why Phase 1 (Years 1-5) is Critical:**
- Test with €20k before risking €1M pension ✅
- Build track record of consistent 25-30% returns ✅
- Develop execution skills ✅
- Validate strategies across market cycles ✅
- Gain psychological confidence ✅

**Phase 2 Success Probability:**
- With €150k+ self-built capital: 70% chance financial independence
- With €1M+ pension withdrawal: 90% chance financial independence
- Combined: Very high probability of €100-120k/year income

---

## Learning Journey Notes

**What worked:**
- Systematic testing (lost €0 while testing)
- Honest validation (53.6% = admitted no edge)
- Mathematical validation (ADF test p=0.0000)
- Clean architecture (separation of concerns)
- BDD test-first approach

**What didn't work (but taught valuable lessons):**
- Volatility ML: 53.6% might be noise
- Session overlap: -0.8% (more is not better)
- Trend duration: 42.1% (class imbalance)
- NaN bug: Silent failures are dangerous

**The 10-year plan principle:**
- Years 1-5: Test everything aggressively (CURRENT: Vasicek)
- Years 5-10: Deploy only proven strategies
- Never risk retirement money on unproven approaches

---

## Next Actions

### Immediate (Week 2)

1. **Day 1: Create TriangularSpreadCalculator**
   - Pure math component (no dependencies)
   - Unit tests pass
   - Document formulas

2. **Day 2: Generate Test Fixtures**
   - Synthetic spreads with known parameters
   - Real EUR/USD spread data (1000+ points)
   - Non-stationary data for negative tests

3. **Day 3-4: Implement VasicekModel**
   - OLS calibration
   - Z-score calculation
   - Mean reversion validation
   - Half-life calculation

4. **Day 5: Complete vasicek_model.feature Tests**
   - All step definitions
   - Run full test suite
   - Target: 30+ new tests passing

### Next Week (Week 3)

1. Implement TriangularArbitragePredictor
2. Implement TriangularStrategy
3. Complete all BDD tests
4. Fix remaining failing tests
5. **Target: 252/252 tests passing (100%)**

---

## Risk Mitigation

**Implementation Risks:**

| Risk | Mitigation | Status |
|------|-----------|--------|
| Calibration unstable | Use OLS method from validated YouTube source | Planned |
| Spread non-stationary | ADF test already passed (p=0.0000) | ✅ Validated |
| 3-leg execution fails | Atomic transaction with rollback | Design ready |
| Half-life too long | Monitor daily, exit if >50 periods | Threshold set |
| Tests fail | BDD-first approach, implement to pass | In progress |

**10-Year Plan Risks:**

| Risk | Mitigation | Status |
|------|-----------|--------|
| Phase 1 fails (€20k → €150k) | Only lost €20k, pension safe | Acceptable |
| Spread competition increases | Monitor half-life, find new edges | Active monitoring |
| Pension rules change | Track Swiss legislation | Planned |
| Market regime change | Multi-strategy portfolio (5-10 edges) | Design ready |

---

**Status: Week 2 implementation in progress. Target completion: End of January 2026.** 🎯

---

## Strategy Pipeline (Backup Plans)

**Purpose:** Ensure continuity if triangular arbitrage doesn't work as expected.

### Decision Tree
```
Triangular Arbitrage (Current)
    │
    ├── Works? ──────► Scale up, add more triangles
    │
    └── Doesn't work?
            │
            ├── Why: Edge too small
            │       └──► Plan B: Multi-leg FX (4-5 currencies)
            │
            ├── Why: Execution costs too high
            │       └──► Plan C: Crypto triangular (lower fees)
            │
            ├── Why: Single exchange not enough
            │       └──► Plan D: Crypto cross-exchange
            │
            ├── Why: Arbitrage fundamentally doesn't work
            │       └──► Plan E: Options volatility
            │
            └── Why: Need different approach entirely
                    └──► Plan F: Crypto market making
```

### Pipeline Items

| Nr | Strategy | Trigger | Skills Reused | New Skills Needed | Capital | Effort |
|----|----------|---------|---------------|-------------------|---------|--------|
| 60 | Multi-leg FX (4-5 currencies) | Edge too small | Vasicek, Python | Quarkus (optional) | €20k | Medium |
| 61 | Crypto triangular | FX costs too high | Vasicek math, BDD | Exchange APIs | €5k | Medium |
| 62 | Crypto cross-exchange | Single exchange not enough | Quarkus, infra | Multi-exchange sync | €5k | High |
| 63 | Options volatility | Arbitrage doesn't exist | Python, ML | Greeks, vol surface | €20k | High |
| 64 | Crypto market making | Directional edge needed | Quarkus, risk mgmt | Inventory management | €10k | High |

### Plan B: Multi-Leg FX (Item 60)

**When to try:** Triangular edge exists but too small after costs

**What it is:**
```
4-leg example:
  EUR → GBP → CHF → USD → EUR
  
Identity: EUR/GBP × GBP/CHF × CHF/USD × USD/EUR = 1.0?
If not → Arbitrage opportunity
```

**Complexity scaling:**
```
3 currencies = 3 pairs = 1 triangle
4 currencies = 6 pairs = 3 triangles
5 currencies = 10 pairs = 15 triangles
6 currencies = 15 pairs = 45 triangles
7 currencies = 21 pairs = 105 triangles
```

**Your advantage:** Monitor all combinations simultaneously with Quarkus

**Reusable:** Vasicek model, Z-score logic, state management

**New:** Multi-path calculation, Quarkus implementation

---

### Plan C: Crypto Triangular (Item 61)

**When to try:** FX broker costs kill the edge

**Why crypto:**
- Lower fees (0.1% vs 0.3% FX)
- 24/7 markets
- More volatility = bigger mispricings
- Engineering IS the moat

**Example:**
```
On Binance:
  BTC/USDT, ETH/BTC, ETH/USDT

Check: BTC/USDT × ETH/BTC = ETH/USDT?
```

**Risks:**
- Exchange bankruptcy
- Regulatory changes
- Liquidity gaps

**Reusable:** Vasicek math, signal generation, BDD framework

**New:** Exchange API connectors, crypto-specific risk management

---

### Plan D: Crypto Cross-Exchange (Item 62)

**When to try:** Single exchange arbitrage competed away

**What it is:**
```
BTC on Kraken:  $42,000
BTC on Binance: $42,050
→ Buy Kraken, sell Binance, profit $50
```

**Why engineering matters:**
- Different APIs per exchange
- Different rate limits
- Transfer delays (blockchain confirmation)
- Error handling, retry logic

**Your Quarkus advantage:**
- Robust multi-exchange connector
- Handle partial fills
- State machine for transfer tracking

**Reusable:** Risk framework, position management

**New:** Multi-exchange infrastructure, blockchain integration

---

### Plan E: Options Volatility (Item 63)

**When to try:** Arbitrage fundamentally doesn't work for you

**What it is:**
- Implied volatility vs realized volatility
- Sell expensive vol, buy cheap vol
- Time decay (theta) works for you

**Why different from FX:**
- Options have time decay
- Volatility IS the product
- Less HFT competition in options

**Needs:**
- Options account (Interactive Brokers)
- More capital (options expensive)
- Domain knowledge (Greeks)

**Reusable:** Statistical validation, Python ML skills

**New:** Options pricing, Greeks calculation, vol surface

---

### Plan F: Crypto Market Making (Item 64)

**When to try:** All arbitrage fails, need different approach

**What it is:**
- Provide liquidity on order book
- Earn bid/ask spread
- Manage inventory risk

**Why Quarkus:**
- Speed matters (seconds, not HFT)
- Handle many pairs simultaneously
- Real-time risk management

**Risks:**
- Inventory risk (holding wrong side during move)
- Adverse selection (smart money picks you off)

**Reusable:** Quarkus skills, real-time processing

**New:** Market making theory, inventory management

---

### Decision Checkpoints

| Checkpoint | When | Question | If Yes | If No |
|------------|------|----------|--------|-------|
| D1 | Paper trading (Week 6) | Edge visible in signals? | Continue to live | Try Plan B |
| D2 | €100 live (Month 3) | Profitable after costs? | Scale to €1k | Try Plan C |
| D3 | €1000 live (Month 6) | Consistent returns? | Full deploy | Evaluate Plans D-F |
| D4 | Year 1 review | 20%+ annual return? | Stay course | Major pivot |

---

### USD Black Swan Mitigation

**Concern:** USD is biggest candidate for black swan event

**Triangular arbitrage is currency-neutral:**
```
Trade 1: BUY  EUR/USD  (long EUR, short USD)
Trade 2: SELL EUR/GBP  (short EUR, long GBP)
Trade 3: SELL GBP/USD  (short GBP, long USD)

Net exposure:
  EUR: +1 -1 = 0
  GBP: +1 -1 = 0
  USD: -1 +1 = 0

You hold NOTHING. Only the spread.
```

**Real risk is execution:** If one leg fails, you HAVE exposure

**Diversification approach (future):**
```json
"triangles": [
  {"name": "USD_triangle", "target": "EURUSD", "leg1": "EURGBP", "leg2": "GBPUSD", "allocation": 0.5},
  {"name": "CHF_triangle", "target": "EURCHF", "leg1": "EURGBP", "leg2": "GBPCHF", "allocation": 0.3},
  {"name": "JPY_triangle", "target": "EURJPY", "leg1": "EURGBP", "leg2": "GBPJPY", "allocation": 0.2}
]
```

**For now:** Get one triangle working, add diversification later

---

### Engineering Edge Assessment

| Strategy | Quarkus Advantage | Worth Building? |
|----------|-------------------|-----------------|
| Triangular FX (current) | Minimal - broker is bottleneck | Stick with Python |
| Multi-leg FX | Medium - calculation speed helps | **Maybe** |
| Crypto triangular | **High** - engineering IS moat | **Consider** |
| Crypto cross-exchange | **High** - multi-system integration | **Consider** |
| Options | Medium - need domain knowledge first | Not yet |
| Market making | High - but need risk framework | **Later** |

---

### Latency Tiers

| Tier | Speed | Who Dominates | Your Chance |
|------|-------|---------------|-------------|
| 1 | Nanoseconds | HFT only | None |
| 2 | Milliseconds | Market makers, prop shops | None |
| **3** | **Seconds (100ms-10s)** | **Open competition** | **Quarkus helps** |
| 4 | Minutes/Hours | Patience-based | Python is fine |

**Your current approach (Vasicek):** Tier 4 - patience is edge, Python is fine

**Future opportunities:** Tier 3 - Quarkus becomes relevant

---