# Item 42: Error Analysis Dashboard

## Objective
Systematically analyze where and when volatility predictions fail to reveal missing features, optimal trading windows, and actionable filter rules.

## Current Problem
- Aggregate accuracy (53.6%) hides patterns
- Don't know if model works better at certain times/regimes
- Can't identify which feature gaps cause failures
- Missing opportunities to filter bad predictions

## What to Analyze

### 1. Time-Based Patterns
**Question:** When does the model fail?

```python
# Analyze prediction errors by:
- Hour of day (0-23 UTC)
- Day of week (Monday-Friday)
- Week of month (1-4)
- Month of year (1-12)

# Metrics for each slice:
- Accuracy (TP+TN)/(TP+TN+FP+FN)
- Precision (TP)/(TP+FP)
- Recall (TP)/(TP+FN)
- F1 Score
- Sample count

# Expected findings:
- Asian session (0-7 UTC): Lower accuracy?
- London/NY overlap (13-17 UTC): Higher accuracy?
- Fridays: Worse (weekend uncertainty)?
- Mondays: Worse (gap from weekend)?
```

### 2. Regime-Based Patterns
**Question:** In what market conditions does model fail?

```python
# Segment by actual market regime:
- Actual = HIGH_VOL, Predicted = HIGH_VOL (True Positive)
- Actual = HIGH_VOL, Predicted = LOW_VOL (False Negative)
- Actual = LOW_VOL, Predicted = HIGH_VOL (False Positive)
- Actual = LOW_VOL, Predicted = LOW_VOL (True Negative)

# Additional regime segmentation:
- Current ATR level (low/med/high percentile)
- Recent trend strength (trending vs ranging)
- Volatility expanding vs contracting

# Expected findings:
- Model only works when current vol is LOW?
- Fails to predict transitions (compression → expansion)?
- Too many false alarms in choppy markets?
```

### 3. Magnitude-Based Patterns
**Question:** Does model miss big moves?

```python
# Analyze errors by volatility magnitude:
- Small vol changes (< 25th percentile)
- Medium vol changes (25-75th percentile)
- Large vol changes (> 75th percentile)
- Extreme vol changes (> 95th percentile)

# Expected findings:
- Model catches small changes but misses big ones?
- Only works for moderate volatility?
```

### 4. Sequential Patterns
**Question:** Does model get "stuck"?

```python
# Analyze errors after:
- N consecutive correct predictions
- N consecutive wrong predictions
- Recent regime changes
- Volatility spikes

# Expected findings:
- Degrades after being correct (overconfident)?
- Gets stuck predicting same regime?
```

### 5. Feature Correlation
**Question:** Which feature values predict errors?

```python
# For each feature, analyze error rate when:
- efficiency_ratio < 0.2 (very choppy)
- efficiency_ratio > 0.8 (very trending)
- vol_20 in bottom/top quartile
- hour_sin/hour_cos specific values
- Gap from overnight session

# Expected findings:
- Errors correlate with low efficiency_ratio → Need Item 1 (compression)
- Errors at certain times → Session filter needed
```

### 6. Confidence Analysis
**Question:** Are low-confidence predictions worse?

```python
# Segment by model confidence (predict_proba):
- Very low (<0.55): How accurate?
- Low (0.55-0.65): How accurate?
- Medium (0.65-0.75): How accurate?
- High (0.75-0.85): How accurate?
- Very high (>0.85): How accurate?

# This validates Item 23 (probability threshold)
# Expected: Confidence correlates with accuracy
```

## Deliverables

### 1. ERROR_ANALYSIS_REPORT.md

```markdown
# Volatility Predictor Error Analysis
Generated: YYYY-MM-DD

## Executive Summary
- Overall accuracy: 53.6%
- Best conditions: [conditions] → 62% accuracy
- Worst conditions: [conditions] → 44% accuracy
- Key insight: Model works in X but fails in Y

## Time Patterns
### Hourly Breakdown
[Chart showing accuracy by hour]
- Best hours: 13:00-17:00 UTC (61% accuracy, N=50k samples)
- Worst hours: 2:00-6:00 UTC (47% accuracy, N=20k samples)

### Weekly Breakdown
[Chart showing accuracy by day of week]
- Monday: 51% (weekend gap effects)
- Wednesday: 56% (mid-week stability)
- Friday: 49% (pre-weekend uncertainty)

## Regime Patterns
### Current Volatility State
- Errors when currently in HIGH_VOL: 58%
- Errors when currently in LOW_VOL: 48%
→ **Model struggles to predict from calm to chaos**

### Confusion Matrix Deep Dive
- False Negatives (missed HIGH_VOL): 404,458 samples
  - When: [top failure scenarios]
  - Why: [feature analysis]
- False Positives (false alarms): 360,809 samples
  - When: [top failure scenarios]
  - Why: [feature analysis]

## Feature Gaps Revealed
1. **Compression not detected**
   - Errors cluster when efficiency_ratio < 0.3
   - → Confirms need for Item 1 (BB Squeeze)

2. **Session timing matters**
   - 14% accuracy swing between Asian and Overlap sessions
   - → Consider session-based filter

3. **Confidence calibration works**
   - <60% confidence: 48% accuracy
   - >80% confidence: 67% accuracy
   - → Item 23 (threshold) will help

## Actionable Recommendations
1. **Immediate filters (no new features):**
   - Only trade 13:00-17:00 UTC → Expected +8% accuracy
   - Only trade when confidence >70% → Expected +5% accuracy
   - Don't trade Mondays/Fridays → Expected +2% accuracy

2. **Missing features:**
   - Item 1 (BB Squeeze) - addresses compression blindness
   - Item 24 (ATR Ratio) - relative vs absolute values

3. **Strategic insight:**
   - Model has edge in specific conditions (LOW_VOL regime)
   - Hybrid with Vasicek: ML filters regime, Vasicek signals entry
```

### 2. Python Module: `error_analyzer.py`

```python
class ErrorAnalyzer:
    """Analyze prediction errors to find patterns"""

    def __init__(self, features_df, labels, predictions, confidences):
        self.features = features_df
        self.labels = labels
        self.predictions = predictions
        self.confidences = confidences
        self.errors = (labels != predictions)

    def analyze_time_patterns(self):
        """Error rates by hour, day, week, month"""
        pass

    def analyze_regime_patterns(self):
        """Error rates by volatility regime"""
        pass

    def analyze_magnitude_patterns(self):
        """Error rates by volatility magnitude"""
        pass

    def analyze_sequential_patterns(self):
        """Error rates after streaks"""
        pass

    def analyze_feature_correlation(self):
        """Which feature values correlate with errors"""
        pass

    def analyze_confidence_calibration(self):
        """Accuracy vs confidence levels"""
        pass

    def generate_report(self, output_path):
        """Generate comprehensive markdown report"""
        pass
```

### 3. BDD Test Scenario

```gherkin
@analysis
Scenario: Error analysis reveals patterns
  Given time_efficiency predictor is trained
  And validation set has predictions and actuals
  When I run error analysis
  Then report should show accuracy by hour
  And report should show accuracy by day of week
  And report should identify best/worst conditions
  And report should correlate errors with features
  And report should recommend filters
```

## Implementation Steps

### Day 1: Core Analysis
1. Create `ErrorAnalyzer` class in `src/hybrid/ml_model/`
2. Implement time-based analysis
3. Implement regime-based analysis
4. Test on time_efficiency results

### Day 2: Advanced Analysis + Report
1. Implement magnitude/sequential/feature analysis
2. Implement confidence calibration
3. Generate markdown report with charts
4. Create actionable recommendations

## Expected Insights

**Best case:**
- "Only trade 13-17 UTC + confidence >70% → 67% accuracy"
- Proves model has real edge in specific conditions
- Clear path to profitability

**Worst case:**
- "No pattern found, errors are random"
- Proves model has no edge
- Pivot to Vasicek immediately

**Most likely:**
- "Model works in LOW_VOL, fails in HIGH_VOL"
- "Missing compression signal (Item 1 needed)"
- "Session timing matters (filter rules needed)"
- Validates hybrid approach: ML + Vasicek

## Success Criteria

**Item 42 is complete when:**
1. ✅ ERROR_ANALYSIS_REPORT.md generated for all 6 configs
2. ✅ At least 3 actionable filter rules identified
3. ✅ Feature gaps identified with specific Item recommendations
4. ✅ Confidence calibration validates Item 23 approach
5. ✅ Best/worst trading conditions clearly documented

## Effort Estimate
- **Core implementation:** 8 hours
- **Report generation:** 4 hours
- **Analysis & interpretation:** 4 hours
- **Total:** 16 hours = 2 days

## Dependencies
- Requires working predictor (Item 39 fixed ✓)
- Requires validation results (just generated ✓)
- No new features needed
- Can run immediately

## Value Proposition

**Why this is worth 2 days:**

1. **May reveal free +10% accuracy** via filters (no new features)
2. **Validates/invalidates next features** (Item 1, 23, 24)
3. **Proves or disproves edge exists** in specific conditions
4. **Informs Vasicek strategy** (when to use each model)
5. **Professional ML methodology** (error analysis is standard practice)

**This could be the most valuable 2 days of the entire project.**