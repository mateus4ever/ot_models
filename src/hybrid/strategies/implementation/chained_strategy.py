from typing import Dict, Optional

from src.hybrid.strategies.implementation import BaseStrategy

""""
=== HYBRID STRATEGY CONCEPTS TO PRESERVE ===

1. REGIME-BASED SIGNAL PROCESSING
   - Trending Up (regime=1): Only long signals, max(0, signal)
   - Trending Down (regime=2): Only short signals, min(0, signal)
   - High Volatility (regime=3): Reduced signal strength (multiplier)
   - Ranging (regime=0): Mean reversion signals

2. VOLATILITY-BASED POSITION SIZING
   - ML volatility prediction adjusts position size
   - High vol + high confidence = reduced position (high_vol_position_multiplier)
   - Configurable threshold: high_vol_confidence_threshold

3. DURATION-BASED MULTIPLIERS
   - Predicted trend duration affects position size
   - Categories: very_short, short, medium, long
   - Confidence-scaled: base_multiplier * confidence + blend * (1 - confidence)

4. SIGNAL COMBINATION
   - Weighted average of multiple signals (KAMA, Kalman)
   - Configurable weights per signal source
   - Confidence scoring from multiple ML components

5. REGIME ADJUSTMENTS (from config)
   regime_adjustments:
     trending: 1.0
     ranging: 0.5
     high_volatility: 0.3

6. KEY FORMULAS
   final_signal = signal_strength * confidence_mult * volatility_mult
   position_size = base_size * regime_mult * confidence_mult * vol_mult * duration_mult * abs(signal)
   position_size = min(position_size, max_position_size)

7. ML COMPONENTS TO INTEGRATE LATER
   - RegimeDetector (rule-based, not ML)
   - VolatilityPredictor (ML)
   - DurationPredictor (ML)
   - TechnicalSignalGenerator (KAMA, Kalman, RSI, BB, MACD)

=== END TAKEAWAYS ===
"""

"""
Strategy with signal chains for entry/exit

=== CHAIN LOGIC (TODO) ===
Entry chain: AND logic - all signals must agree
Exit chain: OR logic - first signal that triggers wins
Filter chain: AND logic - all must pass before entry

Future: configurable logic (AND/OR/weighted voting)
See optimization notes for signal combination optimization
"""

class ChainedStrategy(BaseStrategy):
    def __init__(self, name, config):
        super().__init__(name, config)
        # Override single signals with chains
        self.entry_chain = []  # Multiple entry signals (AND logic)
        self.exit_chain = []  # Multiple exit signals (OR logic)
        self.filter_chain = []  # Pre-entry filters

        # From old HybridStrategy
        self.predictors = []

    def add_entry_signal(self, signal):
        """ChainedStrategy: add to entry chain"""
        self.entry_chain.append(signal)

    def add_exit_signal(self, signal):
        """ChainedStrategy: add to exit chain"""
        self.exit_chain.append(signal)

    def add_filter_signal(self, signal):
        """Add filter to filter chain"""
        self.filter_chain.append(signal)

    def get_optimizable_parameters(self) -> Dict:
        params = {}

        # Get signals config
        signals_config = self.config.get_section('signals', {})

        # Load from active signals (based on strategy config)
        strategy_config = self.config.get_section('strategy', {})
        active_signals = strategy_config.get('active_signals', [])

        for signal_name in active_signals:
            # Find signal in nested structure
            signal_def = self._find_signal_definition(signals_config, signal_name)
            if signal_def and 'optimizable_parameters' in signal_def:
                # Prefix params to avoid collisions
                for param_name, param_def in signal_def['optimizable_parameters'].items():
                    params[f"{signal_name}_{param_name}"] = param_def

        # Load from active position sizer
        mm_config = self.config.get_section('money_management', {})
        active_sizer = mm_config.get('position_sizing')
        if active_sizer in mm_config.get('position_sizers', {}):
            sizer_params = mm_config['position_sizers'][active_sizer].get('optimizable_parameters', {})
            for param_name, param_def in sizer_params.items():
                params[f"sizer_{param_name}"] = param_def

        # Load from active risk manager
        active_risk_mgr = mm_config.get('risk_management')
        if active_risk_mgr in mm_config.get('risk_managers', {}):
            risk_params = mm_config['risk_managers'][active_risk_mgr].get('optimizable_parameters', {})
            for param_name, param_def in risk_params.items():
                params[f"risk_{param_name}"] = param_def

        return params

    def _find_signal_definition(self, signals_config: Dict, signal_name: str) -> Optional[Dict]:
        """Find signal definition in nested structure"""
        for category in ['trend_following', 'mean_reversion', 'momentum']:
            if signal_name in signals_config.get(category, {}):
                return signals_config[category][signal_name]
        return None