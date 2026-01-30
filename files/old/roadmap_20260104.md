# Predictor Development Roadmap - Gantt Chart

**Start Date:** January 2026  
**Capacity:** 10 hours/week  
**Goal:** Complete Volatility + Trend essentials, then Vasicek ASAP  
**Last Updated:** January 2, 2026

## Progress Summary

| Status | Bar | Icon | Items |
|--------|-----|------|-------|
| Done | ███ | ✅ | 6, 11, 12 |
| Parked | ░░░ | ⏸️ | 7 (Trend Duration - 42.1% not useful) |
| Planned | ▓▓▓ | ▶️ | 1, 13, 14, 22, 23, 24, 25, 26, 27, 28, 16-20 |
| Buffer | | ⏳ | 10 weeks ahead of schedule |

## Effort Mapping

| Effort Level | Hours | Weeks |
|--------------|-------|-------|
| Low | 5-10h | 1 week |
| Medium | 15-20h | 2 weeks |
| High | 30-40h | 3-4 weeks |

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
            ─────────────────────────────────────────────────────────────────────────────────
VOLATILITY PHASE 1 (Quick Wins)
Item 22         ▓▓▓                                                                         ▶️ Session overlap binary
Item 23         ▓▓▓                                                                         ▶️ Probability threshold (70%)
Item 24         ▓▓▓                                                                         ▶️ ATR Ratio (relative)
            ─────────────────────────────────────────────────────────────────────────────────
VOLATILITY PHASE 2 (Compression)
Item 1              ▓▓▓▓▓▓▓                                                                 ▶️ Compression (BB squeeze, ATR)
Item 13                     ▓▓▓                                                             ▶️ Range Stability
Item 14                     ▓▓▓                                                             ▶️ Bar Anatomy
Validation                      ▓▓                                                          ▶️ Test & Decide
            ─────────────────────────────────────────────────────────────────────────────────
VOLATILITY PHASE 3 (Leading Indicators)
Item 25                         ▓▓▓                                                         ▶️ Event distance (NFP/CPI/FOMC)
Item 26                         ▓                                                           ▶️ VIX cross-asset (1 day test)
Item 27                         ▓                                                           ▶️ DXY extreme (1 day test)
Item 28                         ▓                                                           ▶️ Fourier denoising
            ─────────────────────────────────────────────────────────────────────────────────
VASICEK
Item 20                             ▓▓▓▓▓▓▓                                                 ▶️ Calibration ★
Item 16                                     ▓▓▓▓▓▓▓▓▓▓▓                                     ▶️ Model core
Item 17                                                 ▓▓▓▓▓▓▓                             ▶️ Z-Score signals
Item 18                                                         ▓▓▓                         ▶️ Half-life exits
Item 19                                                             ▓▓▓▓▓▓▓                 ▶️ Monte Carlo
Event Flt                                                                   ▓▓▓             ▶️ Event filter ★
            ─────────────────────────────────────────────────────────────────────────────────
PAPER TRADE                                                                         ▓▓▓▓▓▓▓ ▶️ Demo account
            ─────────────────────────────────────────────────────────────────────────────────
MILESTONE   ███─────────────────●───────●───────●───────────────────────────────●───────●───
            D0                  D1      D1b     D2                              D3      D4
```

**Legend:** ███ = Done ✅ | ▓▓▓ = Planned ▶️ | ░░░ = Parked ⏸️ | ★ = Critical

## Detailed Schedule

| Week | Dates | Item | Predictor | Task | Hours | Status |
|------|-------|------|-----------|------|-------|--------|
| 1 | Jan 1-5 | 11, 12, 6, 7 | All | Cyclical time, Efficiency Ratio, Trend cleanup, Trend validation | 9h | ✅ **D0: 10 weeks ahead** |
| 2 | Jan 6-12 | 22, 23, 24 | Volatility | Quick wins: Session overlap, Prob threshold, ATR Ratio | 10 | ▶️ |
| 3 | Jan 13-19 | 1 | Volatility | Compression features (BB squeeze, ATR) | 10 | ▶️ |
| 4 | Jan 20-26 | 1 | Volatility | Compression features (continued) | 10 | ▶️ |
| 5 | Jan 27-Feb 2 | 13, 14 | Volatility | Range Stability + Bar Anatomy | 10 | ▶️ |
| 6 | Feb 3-9 | - | Volatility | Validation run + analysis | 10 | ▶️ **D1: Volatility Decision** |
| 7 | Feb 10-16 | 25, 26, 27, 28 | Volatility | Event distance, VIX, DXY, Fourier tests | 10 | ▶️ **D1b: External Data Decision** |
| 8 | Feb 17-23 | 20 | Vasicek | Historical calibration (OLS setup) | 10 | ▶️ |
| 9 | Feb 24-Mar 2 | 20 | Vasicek | Historical calibration (validate params) | 10 | ▶️ **D2: Calibration Valid?** |
| 10 | Mar 3-9 | 16 | Vasicek | Vasicek Model core (implement O-U) | 10 | ▶️ |
| 11 | Mar 10-16 | 16 | Vasicek | Vasicek Model core (integrate) | 10 | ▶️ |
| 12 | Mar 17-23 | 16 | Vasicek | Vasicek Model core (test) | 10 | ▶️ |
| 13 | Mar 24-30 | 17 | Vasicek | Z-Score signals (calculation) | 10 | ▶️ |
| 14 | Mar 31-Apr 6 | 17 | Vasicek | Z-Score signals (thresholds) | 10 | ▶️ |
| 15 | Apr 7-13 | 18 | Vasicek | Half-life exits | 10 | ▶️ |
| 16 | Apr 14-20 | 19 | Vasicek | Monte Carlo risk (simulation) | 10 | ▶️ |
| 17 | Apr 21-27 | 19 | Vasicek | Monte Carlo risk (VaR integration) | 10 | ▶️ **D3: Vasicek Core Ready** |
| 18 | Apr 28-May 4 | - | Vasicek | Event filter & grey swan monitor | 10 | ▶️ **D4: Risk Layer Complete** |
| 19 | May 5-11 | - | Vasicek | Paper trading setup + demo account | 10 | ▶️ |
| 20 | May 12-18 | - | Vasicek | Paper trading validation | 10 | ▶️ |

## Decision Points

### D0: Ahead of Schedule (Week 1 - Jan 1) ✅

| Result | Action |
|--------|--------|
| 4 items complete, 10 weeks buffer | Proceed with expanded volatility experiments |
| Trend Duration 42.1% | ⏸️ **Parked** - not useful for trading |

### D1: Volatility Decision (Week 6 - Feb 9)

| Result | Action |
|--------|--------|
| Edge found (>57% accuracy) | Integrate into trading system |
| Still ~53% | Use as filter only, not primary signal |

### D1b: External Data Decision (Week 7 - Feb 16)

| Result | Action |
|--------|--------|
| Event distance helps | Build economic calendar integration |
| VIX/DXY helps | Add cross-asset features |
| Fourier denoising helps | Keep as preprocessing step |
| No improvement | Confirm: stick to intrinsic data only |

### D2: Calibration Valid? (Week 9 - Mar 2)

| Result | Action |
|--------|--------|
| Parameters stable across time | Proceed to implementation |
| Parameters unstable | Try pairs trading (Item 21) or different timeframe |

### D3: Vasicek Core Ready (Week 17 - Apr 27)

| Result | Action |
|--------|--------|
| Signals valid, backtest positive | Proceed to event filter |
| Signals weak | Review calibration, try different assets |

### D4: Risk Layer Complete (Week 18 - May 4)

| Result | Action |
|--------|--------|
| Event filter implemented | Start paper trading |
| Gaps in coverage | Extend by 1 week |

## Summary

| Phase | Weeks | Dates | Focus | Status |
|-------|-------|-------|-------|--------|
| Completed | 1 | Jan 1-5 | Items 6, 7, 11, 12 | ✅ |
| Volatility quick wins | 2 | Jan 6-12 | Session, threshold, ATR ratio | ▶️ |
| Volatility compression | 3-5 | Jan 13 - Feb 2 | BB squeeze, Range, Bar anatomy | ▶️ |
| Volatility validation | 6 | Feb 3-9 | Test & decide | ▶️ D1 |
| Volatility external | 7 | Feb 10-16 | Event distance, VIX, DXY, Fourier | ▶️ D1b |
| Vasicek foundation | 8-9 | Feb 17 - Mar 2 | Calibration | ▶️ D2 |
| Vasicek implementation | 10-14 | Mar 3 - Apr 6 | Core model + signals | ▶️ |
| Vasicek exits + risk | 15-17 | Apr 7 - Apr 27 | Half-life + Monte Carlo | ▶️ D3 |
| Event filter | 18 | Apr 28 - May 4 | Grey swan protection | ▶️ D4 |
| Paper trading | 19-20 | May 5 - May 18 | Demo account validation | ▶️ |

## Total Investment (Revised)

| Phase | Weeks | Hours | Status |
|-------|-------|-------|--------|
| Completed | 1 | 9h | ✅ |
| Volatility (remaining) | 6 | 60h | ▶️ |
| Vasicek | 10 | 100h | ▶️ |
| Event filter | 1 | 10h | ▶️ |
| Paper trading | 2 | 20h | ▶️ |
| **Total** | **20** | **199h** | |

**End Date:** Mid-May 2026 (2 weeks earlier than original)

## Completed Items Summary

| Item | What | Result | Hours | Status |
|------|------|--------|-------|--------|
| 11 | Cyclical time encoding | +0.9% accuracy | 2h | ✅ |
| 12 | Efficiency Ratio | +0.1% accuracy | 2h | ✅ |
| 6 | Trend cleanup | Code maintainable | 3h | ✅ |
| 7 | Trend validation | 42.1% - not useful | 2h | ⏸️ |

**Key Learning:** 53.6% volatility accuracy is within statistical noise of 50%. Need >57% to prove edge.

## New Items Added

| Item | What | Effort | Success (1-5) | Rationale | Status |
|------|------|--------|---------------|-----------|--------|
| 22 | Session overlap binary | Low | 3 | 70% volume in London/NY overlap | ▶️ |
| 23 | Probability threshold | Low | 4 | Trade only 70%+ confidence | ▶️ |
| 24 | ATR Ratio (relative) | Low | 4 | Stationary feature, RF-friendly | ▶️ |
| 25 | Event distance | Medium | 4 | Leading indicator - know when vol comes | ▶️ |
| 26 | VIX cross-asset | Low | 2 | Likely arbitraged, but quick test | ▶️ |
| 27 | DXY extreme | Low | 2 | Likely arbitraged, but quick test | ▶️ |
| 28 | Fourier denoising | Low | 3 | Signal preprocessing - "car wash" for data | ▶️ |

## Risk Management Checklist (Before Paper Trading)

- [ ] Economic calendar blackouts implemented (NFP, FOMC, ECB, etc.)
- [ ] Grey swan indicators defined
- [ ] Hard limits coded (MAX_LEVERAGE=10, MAX_POSITION=5%, MAX_EXPOSURE=15%)
- [ ] Three trading modes operational (Normal, Elevated, Pull the plug)
- [ ] Sunday gap protection active
- [ ] Holiday blackout (Dec 24 - Jan 2) scheduled

## Multi-Bot Architecture (Future)

```
Orchestrator
    ├── Vasicek Bot (EURUSD)
    ├── Vasicek Bot (GBPUSD)  
    ├── Vasicek Bot (USDJPY)
    └── Vasicek Bot (EURGBP)
    
Shared:
    ├── Risk Manager (total exposure limit)
    ├── Event Filter (shared blackout calendar)
    ├── Capital allocator (20,000 CHF split)
    └── Data feed
```

**Benefit:** 4 pairs × 3-6 trades/year = 12-24 opportunities instead of 3-6