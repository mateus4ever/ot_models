# Pairs Trading Experiment Analysis
**Date:** January 3, 2026

## Results Summary

### Simple Strategy (SPY-QQQ, 2022-2023)
- **Period:** Nov 2022 - Feb 2023 (3 months)
- **Trades:** 7 completed
- **Total PnL:** -$4,137.63 (LOSS)
- **Avg per trade:** -$591
- **Win rate:** 2/7 = 28.6%
- **Direction:** All short trades
- **Avg holding:** 11 days

### Staggered Strategy (SPY-QQQ, 2020-2025)
- **Period:** Jan 2020 - May 2025 (5.4 years)
- **Initial capital:** $20,000
- **Final capital:** $20,532
- **Total profit:** +$532
- **Return:** +2.66% over 5.4 years
- **Annualized:** ~0.5%/year
- **Trades:** 566 total (many partial entries/exits)

---

## What Went Wrong

### 1. SPY-QQQ Are Too Correlated (0.95+)
**Problem:** Both track US equities (S&P 500 vs Nasdaq 100)
- Spread barely mean-reverts
- Both move together in same direction
- No structural reason for spread to return to mean
- During market crashes (2020, 2022), spread explodes and doesn't revert

**Evidence:**
- 5 years, only +2.66% return
- Simple strategy lost money (-$4k)
- Thesis used gold + cocoa (correlation 0.146) ← 6x less correlated

### 2. Wrong Pair Selection
**Thesis approach:**
```
Gold + Cocoa correlation: 0.146 (LOW)
→ When one down, other might be up
→ Diversification benefit
```

**Your approach:**
```
SPY + QQQ correlation: 0.95+ (VERY HIGH)
→ When one down, other also down
→ No diversification
→ Spread just trends with market regime
```

### 3. No Structural Mean-Reversion Force
**Good pairs have:**
- Substitutes (Coke vs Pepsi - consumers switch)
- Same input costs (Airlines - fuel costs)
- Statistical arbitrage (ETF vs underlying)

**SPY vs QQQ have:**
- Different index composition (S&P 500 vs Nasdaq 100)
- Tech weighting difference (QQQ = 50% tech, SPY = 25%)
- During tech boom: Spread widens permanently
- No fundamental force pulling spread back

### 4. Transaction Costs Kill Small Edges
**Staggered strategy:**
- 566 trades over 5.4 years = 105 trades/year
- If 0.1% cost per trade: 105 × 0.1% = 10.5%/year in costs
- Gross return needed: 11%/year just to break even
- Actual: 0.5%/year net ← Edge destroyed by costs

**Aligns with your principle:** "Trade less, trade better"

---

## What Worked (Conceptually)

### 1. Staggered Entry Logic
**Scale in at z = 1.0, 1.5, 2.0:**
- Averages down when conviction increases
- Reduces risk of single bad entry
- Similar to Dollar Cost Averaging

**Application to trade_bot:**
- Could scale into Vasicek 3σ trades
- Entry 1: z=2.5σ (small size)
- Entry 2: z=3.0σ (full size)
- Entry 3: z=3.5σ (double down)

### 2. Multiple Exit Conditions
**Three triggers:**
1. Mean reversion (|z| < 0.3)
2. Time limit (10 days max)
3. Stop loss (2% hard stop)

**Application to trade_bot:**
- Already planning half-life exits (Vasicek Item 18)
- Add time limit (prevent zombie positions)
- Add hard stop (prevent single catastrophic loss)

### 3. Realistic Backtesting
**Included:**
- Transaction costs (0.1%)
- Slippage assumptions
- Max position size limits
- Cooldown between trades

**Application to trade_bot:**
- Must include Swissquote spread costs
- Model realistic fills
- Never assume perfect execution

### 4. Grid Search Optimization
**Sharpe ratio maximization:**
- Tested exit_threshold [0.2 - 0.4]
- Tested stop_loss [1% - 3%]
- Tested max_holding [5 - 30 days]

**Application to trade_bot:**
- Use for Vasicek parameter optimization
- Optimize z-score threshold (2σ vs 3σ)
- Optimize half-life exit multiplier

---

## What to Keep for Trade_Bot

### ✅ Keep These Concepts

| Concept | From Pairs | Apply to Trade_Bot |
|---------|------------|-------------------|
| **Z-score signals** | Entry at z=±1.5 | Vasicek: z=±3σ (more extreme) |
| **Staggered entry** | Scale in at 3 levels | Scale in for high conviction |
| **Multiple exits** | Mean-rev + time + stop | Half-life + time + stop |
| **Transaction costs** | 0.1% per trade | Model Swissquote spreads |
| **Grid optimization** | Sharpe maximization | Optimize Vasicek params |
| **Trade log** | Every entry/exit recorded | Essential for post-analysis |

### ❌ Avoid These Mistakes

| Mistake | What Happened | How to Avoid |
|---------|--------------|--------------|
| **High correlation pairs** | SPY/QQQ = 0.95 | Test EUR/USD vs GBP/USD = ? |
| **No mean-reversion force** | No structural reason | Vasicek = mathematical force |
| **Too many trades** | 105/year killed returns | Vasicek = 3-6/year target |
| **Wrong timeframe** | Daily bars = noisy | Test if 4H/1H better |

---

## Pair Selection for Trade_Bot

### Forex Pairs to Test

**High correlation (like SPY/QQQ - AVOID):**
- EUR/USD vs GBP/USD (both USD-denominated, correlated ~0.85)
- Bad for spread trading

**Lower correlation (like Gold/Cocoa - BETTER):**
- EUR/USD vs USD/JPY (inverse USD exposure)
- AUD/USD vs NZD/USD (similar economies but different)
- EUR/GBP vs EUR/USD (different currency dynamics)

**Action:** Calculate correlation matrix like thesis did (Table 4.1)

### Better Alternative: Single-Asset Vasicek

**Instead of pairs trading:**
```
Pairs:     Trade SPY-QQQ spread mean-reversion
Problem:   No structural force, 0.95 correlation

Vasicek:   Trade EUR/USD price mean-reversion
Advantage: Mathematical model, proven theory
Target:    3σ deviations, 3-6 trades/year
```

**Thesis Item 21 (Pairs trading) = HIGH effort, LOW success probability**
**Thesis Item 20 (Vasicek calibration) = MEDIUM effort, HIGH probability**

---

## Critical Questions

### 1. Did You Test EUR/USD Spread?
Your files show SPY/QQQ (stocks). Did you test forex pairs?

**If yes:** What correlation? What results?
**If no:** SPY/QQQ results don't transfer to forex

### 2. What's EUR/USD + GBP/USD Correlation?
Need to know before attempting spread trading.

**Calculate:** Load EUR/USD and GBP/USD data, compute correlation
**Hypothesis:** Probably 0.80-0.90 (too high for pairs trading)

### 3. Why Not Pure Vasicek First?
**Pairs trading:**
- Needs 2 assets with specific correlation
- Complex entry/exit logic
- High trade frequency

**Vasicek:**
- Needs 1 asset with mean-reversion
- Simple entry (3σ), simple exit (half-life)
- Low trade frequency

**Recommendation:** Test Vasicek on EUR/USD before attempting pairs

---

## Honest Assessment

### The Brutal Truth

**Your pairs experiment shows:**
1. ❌ SPY/QQQ don't mean-revert (only +2.66% in 5 years)
2. ❌ Simple strategy lost money (-$4k)
3. ❌ 105 trades/year with costs = dead on arrival
4. ✅ But you learned staggered entry, exits, realistic testing

**This validates "Trade Less, Trade Better":**
- Staggered: 105 trades/year → 0.5%/year
- Target: 3-6 trades/year → ???%/year

### What This Means

**Don't implement pairs trading for trade_bot YET:**
- SPY/QQQ results don't prove forex pairs work
- Correlation needs testing first
- Vasicek single-asset is simpler

**Do keep the techniques:**
- Staggered entry ✓
- Multiple exits ✓
- Realistic costs ✓
- Grid optimization ✓

---

## Next Steps

### Immediate (Don't Wait for Comparison Test)

1. **Calculate EUR/USD + GBP/USD correlation**
   - Load both from data/big/
   - Compute rolling correlation
   - If > 0.85: Pairs trading won't work

2. **Test Vasicek on EUR/USD first**
   - Item 20: Historical calibration
   - Simpler than pairs
   - Fewer assumptions

3. **Shelf pairs trading until proven**
   - Item 21 = LOW priority
   - Only pursue if forex correlations < 0.5

### When Comparison Test Finishes

**If ML shows no edge (still 50%):**
- Pivot to Vasicek immediately
- Use pairs trading techniques (staggered entry, etc.)
- Target 3-6 trades/year

**If ML shows edge (>57%):**
- Still test Vasicek in parallel
- Pairs trading remains LOW priority

---

## Alignment with Session Principles

### "Trade Less, Trade Better" ✓

Pairs experiment proves this:
- 105 trades/year = 0.5%/year (terrible)
- Costs killed any edge
- Validates 3-6 trades/year target

### "Market Agnostic" ✓

You tested stocks (SPY/QQQ), not forex:
- Good: Willing to test different markets
- But: Results don't transfer to forex
- Need: Test forex pairs separately

### "Execution Costs Matter" ✓

Staggered strategy:
- Theoretical edge: Maybe 5%/year
- After 105 trades × 0.1% cost: 0.5%/year
- Validates Swissquote cost concern

### "Quality Over Quantity" ✓

Perfect example:
- Simple: 7 trades, -$4k (quality was bad)
- Staggered: 566 trades, +$532 (quantity killed edge)
- Lesson: Neither worked, need BOTH quality AND low frequency

---

## Conclusion

### What You Discovered

**Pairs trading SPY/QQQ doesn't work:**
- 5.4 years, only +2.66% return
- Too correlated (0.95+)
- No structural mean-reversion
- Transaction costs killed edge

**But the techniques are valuable:**
- Staggered entry: Apply to Vasicek
- Multiple exits: Already planned (Item 18 + stops)
- Realistic testing: Must include for all strategies
- Grid optimization: Use for Vasicek params

**Recommendation:**
1. Don't pursue pairs trading yet
2. Test Vasicek on single EUR/USD first
3. If Vasicek works, THEN test forex pairs
4. Use pairs techniques (staggered, exits) in Vasicek

### This is GOOD News

You didn't waste time:
- Learned what doesn't work (high correlation pairs)
- Discovered valuable techniques (staggered, exits)
- Validated core principles (fewer trades, costs matter)
- Confirmed Vasicek should be priority

**The experiment worked - it eliminated a dead end before you coded it into the bot.**

That's exactly what experiments should do.