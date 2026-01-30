# Predictor Development Roadmap - Gantt Chart

**Start Date:** January 2026  
**Capacity:** 10 hours/week  
**Goal:** Complete Volatility + Trend essentials, then Vasicek ASAP

## Effort Mapping

| Effort Level | Hours | Weeks |
|--------------|-------|-------|
| Low | 5-10h | 1 week |
| Medium | 15-20h | 2 weeks |
| High | 30-40h | 3-4 weeks |

## Timeline

```
2026        Jan                 Feb                 Mar                 Apr                 May                 Jun
Week        1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22
            ───────────────────────────────────────────────────────────────────────────────────────────
VOLATILITY
Item 11     ███                                                                                         Cyclical time
Item 12         ███                                                                                     Efficiency Ratio
Item 1              ███████                                                                             Compression
Validation                  ██                                                                          Test & Decide
            ───────────────────────────────────────────────────────────────────────────────────────────
TREND
Item 6                          ███████                                                                 Code cleanup
Item 7                                  ███████                                                         Validation
            ───────────────────────────────────────────────────────────────────────────────────────────
VASICEK
Item 20                                         ███████                                                 Calibration ★
Item 16                                                 ███████████                                     Model core
Item 17                                                             ███████                             Z-Score signals
Item 18                                                                     ███                         Half-life exits
Item 19                                                                         ███████                 Monte Carlo
Item 22                                                                                 ███             Event filter ★
            ───────────────────────────────────────────────────────────────────────────────────────────
PAPER TRADE                                                                                     ███████ Demo account
            ───────────────────────────────────────────────────────────────────────────────────────────
MILESTONE   ────────────────────●───────────────●───────────────────────────────────────●───────●──────
                                D1              D2                                      D3      D4
```

## Detailed Schedule

| Week | Dates | Item | Predictor | Task | Hours | Milestone |
|------|-------|------|-----------|------|-------|-----------|
| 1 | Jan 6-12 | 11 | Volatility | Cyclical time encoding (sin/cos) | 10 |done|
| 2 | Jan 13-19 | 12 | Volatility | Efficiency Ratio (Kaufman) | 10 | |
| 3 | Jan 20-26 | 1 | Volatility | Compression features (BB squeeze, ATR) | 10 | |
| 4 | Jan 27-Feb 2 | 1 | Volatility | Compression features (continued) | 10 | |
| 5 | Feb 3-9 | - | Volatility | Validation run + analysis | 10 | **D1: Volatility Decision** |
| 6 | Feb 10-16 | 6 | Trend | Code cleanup (remove abstractions) | 10 | |
| 7 | Feb 17-23 | 6 | Trend | Code cleanup (split methods) | 10 | |
| 8 | Feb 24-Mar 2 | 7 | Trend | Validation framework + scenarios | 10 | |
| 9 | Mar 3-9 | 7 | Trend | Validation run + analysis | 10 | **D2: Trend Decision** |
| 10 | Mar 10-16 | 20 | Vasicek | Historical calibration (OLS setup) | 10 | |
| 11 | Mar 17-23 | 20 | Vasicek | Historical calibration (validate params) | 10 | |
| 12 | Mar 24-30 | 16 | Vasicek | Vasicek Model core (implement O-U) | 10 | |
| 13 | Mar 31-Apr 6 | 16 | Vasicek | Vasicek Model core (integrate) | 10 | |
| 14 | Apr 7-13 | 16 | Vasicek | Vasicek Model core (test) | 10 | |
| 15 | Apr 14-20 | 17 | Vasicek | Z-Score signals (calculation) | 10 | |
| 16 | Apr 21-27 | 17 | Vasicek | Z-Score signals (thresholds) | 10 | |
| 17 | Apr 28-May 4 | 18 | Vasicek | Half-life exits | 10 | |
| 18 | May 5-11 | 19 | Vasicek | Monte Carlo risk (simulation) | 10 | |
| 19 | May 12-18 | 19 | Vasicek | Monte Carlo risk (VaR integration) | 10 | **D3: Vasicek Core Ready** |
| 20 | May 19-25 | 22 | Vasicek | Event filter & grey swan monitor | 10 | **D4: Risk Layer Complete** |
| 21 | May 26-Jun 1 | - | Vasicek | Paper trading setup + demo account | 10 | |
| 22 | Jun 2-8 | - | Vasicek | Paper trading validation | 10 | |

## Decision Points

### D1: Volatility Decision (Week 5 - Feb 9)

| Result | Action |
|--------|--------|
| Edge found (>55% precision, beats baseline) | Add Items 13, 14 to backlog for later |
| No edge | Continue as planned, Volatility becomes filter only |

### D2: Trend Decision (Week 9 - Mar 9)

| Result | Action |
|--------|--------|
| Edge found | Integrate with Volatility for regime detection |
| No edge | Archive, focus fully on Vasicek |

### D3: Vasicek Core Ready (Week 19 - May 18)

| Result | Action |
|--------|--------|
| Calibration stable, signals valid | Proceed to event filter |
| Parameters unstable | Review data, try pairs trading (Item 21) |

### D4: Risk Layer Complete (Week 20 - May 25)

| Result | Action |
|--------|--------|
| Event filter implemented, limits set | Start paper trading |
| Gaps in coverage | Research additional events, extend by 1 week |

## Summary

| Phase | Weeks | Dates | Focus |
|-------|-------|-------|-------|
| Volatility essentials | 1-5 | Jan 6 - Feb 9 | New features + validation |
| Trend essentials | 6-9 | Feb 10 - Mar 9 | Cleanup + validation |
| Vasicek foundation | 10-11 | Mar 10 - Mar 23 | Calibration (★ highest priority) |
| Vasicek implementation | 12-17 | Mar 24 - May 4 | Core model + signals + exits |
| Vasicek risk | 18-19 | May 5 - May 18 | Monte Carlo + VaR |
| Event filter | 20 | May 19-25 | Economic calendar + grey swan (★ critical) |
| Paper trading | 21-22 | May 26 - Jun 8 | Demo account validation |

## Total Investment

| Phase | Weeks | Hours |
|-------|-------|-------|
| Volatility | 5 | 50h |
| Trend | 4 | 40h |
| Vasicek | 10 | 100h |
| Event filter | 1 | 10h |
| Paper trading | 2 | 20h |
| **Total** | **22** | **220h** |

**End Date:** Early June 2026 (ready for paper trading with full risk management)

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