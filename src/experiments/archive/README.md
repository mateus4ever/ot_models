# Experimental Options Research Archive
**Date Range:** 2023-2024  
**Status:** Research complete, waiting for production trigger

## What's Here

### ARCH/GARCH (Volatility Modeling)
- Volatility clustering tests
- GARCH calibration attempts
- **Key learning:** Vol clusters, time-varying volatility
- **Useful for:** Items 4, 5, 15 (GARCH features)

### Black-Scholes (European Options)
- Option pricing implementation
- Implied volatility extraction (Newton-Raphson)
- Greeks calculation
- **Key learning:** Vol → Price conversion, IV extraction
- **Useful for:** Item 35 (foundation)

### Bjerksund-Stensland (American Options)
- Extracted from C++ source code
- American option pricing
- Early exercise analysis
- **Key learning:** When early exercise matters
- **Useful for:** Item 35 (if trading American options)

### Heston (Stochastic Volatility)
- 12 implementation attempts
- Monte Carlo simulations
- Fourier transform pricing
- **Key learning:** FFT for surface pricing, portfolio decomposition
- **Useful for:** Items 36, 37, 38

### Option Volatility Experiments
- IV vs RV comparison
- Volatility smile analysis
- Surface visualization
- **Key learning:** How to compare predicted vs market vol
- **Useful for:** Item 35 (strategy validation)

## Status

- ❌ **Don't use for:** Spot price prediction (doesn't work)
- ✅ **Use for:** Options pricing, volatility trading (when ready)

## When To Use

**Trigger:** Volatility predictor achieves >57% accuracy

**Then:** Extract code from this archive for Items 35-38

## Notes

Implementation was "garbage" for original purpose (spot prediction).
Mathematical insights are gold for correct purpose (options trading).
Archive preserved for future use.