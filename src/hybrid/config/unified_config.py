"""
Unified Configuration System
Loads and manages all configuration from multiple JSON files
Provides backward compatibility and easy access to all parameters
ZERO HARDCODED VALUES SUPPORT FOR ALL COMPONENTS
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class UnifiedConfig:
    """
    Unified configuration system that loads from multiple JSON files and provides
    structured access to all configuration parameters
    ZERO HARDCODED VALUES - ALL PARAMETERS CONFIGURABLE
    """

    def __init__(self, config_path: Optional[str] = None, environment: str = "prod"):
        """
        Initialize UnifiedConfig with automatic path detection

        Args:
            config_path: Optional path to config file/directory. If None, auto-detects.
            environment: Environment to load (dev/prod). Default is "prod".
        """
        self.environment = environment
        self.config_path = config_path or self._find_config_path()

        # Determine if we're using single file or multi-file configuration
        if os.path.isfile(self.config_path):
            # Single file mode (backward compatibility)
            self.config = self._load_single_config()
            self.multi_file_mode = False
        else:
            # Multi-file mode
            self.config = self._load_multi_file_config()
            self.multi_file_mode = True

        # Apply preset if specified
        self._apply_preset()

        # Cache frequently accessed config sections
        self._cache_config_sections()

    def _find_config_path(self) -> str:
        """
        Auto-detect config file path relative to project root
        """
        # Try to find config file starting from current location
        current_path = Path(__file__).resolve()

        # Look for project root markers and config file/directory
        for parent in [current_path.parent] + list(current_path.parents):
            # Check for multi-file config directory first
            config_dir = parent / "config"
            if config_dir.exists() and config_dir.is_dir():
                # Check if we have the expected config files
                base_config = config_dir / "base.json"
                if base_config.exists():
                    return str(config_dir)

            # Check for single file (backward compatibility)
            config_file = parent / "config" / "default.json"
            if config_file.exists():
                return str(config_file)

        # Fallback paths
        fallback_paths = [
            "config",  # Directory
            "config/default.json",  # Single file
            "../config",
            "../config/default.json",
            "../../config",
            "../../config/default.json",
            "../../../config",
            "../../../config/default.json"
        ]

        for path in fallback_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "Could not find config directory or default.json config file. "
            "Please ensure config files exist in a 'config' directory."
        )

    def _load_single_config(self) -> Dict[str, Any]:
        """
        Load configuration from single JSON file (backward compatibility)
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            logger.info(f"Loaded single-file configuration from: {self.config_path}")
            return config

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _load_multi_file_config(self) -> Dict[str, Any]:
        """
        Load configuration from multiple JSON files and merge them
        """
        config_dir = Path(self.config_path)
        merged_config = {}

        # Define the load order (base files first, then environment overrides)
        config_files = [
            "base.json",
            "risk.json",
            "technical.json",
            "display.json",
            "files.json",
            "presets.json"
        ]

        # Load base configuration files
        for config_file in config_files:
            file_path = config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        file_config = json.load(f)
                    self._deep_update(merged_config, file_config)
                    logger.debug(f"Loaded config from: {file_path}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {file_path}: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    raise
            else:
                logger.warning(f"Config file not found: {file_path}")

        # Load environment-specific overrides
        env_file = config_dir / "environments" / f"{self.environment}.json"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    env_config = json.load(f)
                self._deep_update(merged_config, env_config)
                logger.info(f"Applied {self.environment} environment overrides from: {env_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {env_file}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading {env_file}: {e}")
                raise
        else:
            logger.info(f"No environment config found for: {self.environment}")

        if not merged_config:
            raise FileNotFoundError(
                f"No configuration files found in directory: {config_dir}"
            )

        logger.info(f"Loaded multi-file configuration from: {config_dir} (environment: {self.environment})")
        return merged_config

    def _apply_preset(self):
        """
        Apply preset configuration if specified
        """
        # Check for preset in environment variable or command line args
        import sys

        preset_name = None

        # Check command line arguments for preset
        for arg in sys.argv:
            if arg in ['single', 'compare', 'swing', 'scalping', 'conservative', 'aggressive', 'forex_position']:
                preset_name = f"forex_{arg}" if arg in ['swing', 'scalping'] else arg
                break

        # Apply preset if found
        if preset_name and preset_name in self.config.get('presets', {}):
            preset_config = self.config['presets'][preset_name]
            self._deep_update(self.config, preset_config)
            logger.info(f"Applied preset: {preset_name}")

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """
        Deep update nested dictionary
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _cache_config_sections(self):
        """
        Cache frequently accessed configuration sections for performance
        """
        # Numeric formatting configuration
        self.numeric_formatting = self.config.get('numeric_formatting', {})

        # Array indexing configuration
        self.array_indexing = self.config.get('array_indexing', {})

        # Mathematical operations configuration
        self.mathematical_operations = self.config.get('mathematical_operations', {})

        # Constants
        self.constants = self.config.get('constants', {})

        # Backtesting configuration
        self.backtesting = self.config.get('backtesting', {})

        # Risk management configuration
        self.risk_management = self.config.get('risk_management', {})

        # General configuration
        self.general = self.config.get('general', {})

    # ========================================
    # SECTION ACCESS METHODS
    # ========================================

    def get_section(self, section_name: str, default: Any = None) -> Any:
        """
        Get a configuration section by name

        Args:
            section_name: Name of the configuration section
            default: Default value if section not found

        Returns:
            Configuration section or default value
        """
        return self.config.get(section_name, default)

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values

        Args:
            updates: Dictionary of configuration updates
        """
        self._deep_update(self.config, updates)
        self._cache_config_sections()  # Refresh cached sections

    def save_config(self, filepath: Optional[str] = None):
        """
        Save current configuration to file

        Args:
            filepath: Optional path to save to. If None, uses current config_path.
        """
        if self.multi_file_mode:
            # For multi-file mode, save to a single combined file
            save_path = filepath or str(Path(self.config_path) / f"combined_config_{self.environment}.json")
        else:
            # For single-file mode, save to original file
            save_path = filepath or self.config_path

        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=self.numeric_formatting.get('json_indent', 2))

        logger.info(f"Configuration saved to: {save_path}")

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration setup
        """
        return {
            'config_path': self.config_path,
            'multi_file_mode': self.multi_file_mode,
            'environment': self.environment,
            'sections_loaded': list(self.config.keys()),
            'total_sections': len(self.config)
        }

    def validate_config(self) -> Dict[str, list]:
        """
        Validate configuration and return any issues found

        Returns:
            Dictionary with validation results
        """
        issues = {
            'missing_sections': [],
            'invalid_values': [],
            'warnings': []
        }

        # Check required sections
        required_sections = [
            'general', 'regime_detection', 'volatility_prediction',
            'technical_analysis', 'risk_management', 'backtesting',
            'mathematical_operations', 'array_indexing', 'numeric_formatting'
        ]

        for section in required_sections:
            if section not in self.config:
                issues['missing_sections'].append(section)

        # Check critical values
        if self.config.get('general', {}).get('random_state') is None:
            issues['warnings'].append("No random_state set - results may not be reproducible")

        # Check regime detection method
        regime_method = self.config.get('regime_detection', {}).get('method')
        if regime_method not in ['rule_based', 'ml_based']:
            issues['invalid_values'].append(f"Invalid regime detection method: {regime_method}")

        # Check mathematical operations
        math_ops = self.config.get('mathematical_operations', {})
        if math_ops.get('zero') is None or math_ops.get('unity') is None:
            issues['missing_sections'].append("mathematical_operations must define 'zero' and 'unity'")

        # Check array indexing
        array_idx = self.config.get('array_indexing', {})
        if array_idx.get('first_index') is None or array_idx.get('second_index') is None:
            issues['missing_sections'].append("array_indexing must define index values")

        # Check backtesting parameters
        backtest = self.config.get('backtesting', {})
        required_backtest = ['initial_capital', 'transaction_cost', 'slippage', 'risk_free_rate']
        for param in required_backtest:
            if backtest.get(param) is None:
                issues['missing_sections'].append(f"backtesting.{param} is required")

        # Check risk management parameters
        risk = self.config.get('risk_management', {})
        required_risk = ['stop_loss_pct', 'take_profit_pct', 'max_holding_periods', 'max_position_size']
        for param in required_risk:
            if risk.get(param) is None:
                issues['missing_sections'].append(f"risk_management.{param} is required")

        return issues

    def __repr__(self) -> str:
        """String representation of the config"""
        mode = "multi-file" if self.multi_file_mode else "single-file"
        return f"UnifiedConfig(config_path='{self.config_path}', mode='{mode}', environment='{self.environment}', sections={len(self.config)})"


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def get_config(config_path: Optional[str] = None, environment: str = "prod") -> UnifiedConfig:
    """
    Convenience function to get UnifiedConfig instance

    Args:
        config_path: Optional path to config file/directory
        environment: Environment to load (dev/prod)

    Returns:
        UnifiedConfig instance
    """
    return UnifiedConfig(config_path, environment)


def load_config_from_preset(preset_name: str, config_path: Optional[str] = None,
                            environment: str = "prod") -> UnifiedConfig:
    """
    Load configuration with specific preset applied

    Args:
        preset_name: Name of preset to apply
        config_path: Optional path to config file/directory
        environment: Environment to load (dev/prod)

    Returns:
        UnifiedConfig instance with preset applied
    """
    config = UnifiedConfig(config_path, environment)

    if preset_name in config.config.get('presets', {}):
        preset_config = config.config['presets'][preset_name]
        config._deep_update(config.config, preset_config)
        config._cache_config_sections()  # Refresh cached sections
        logger.info(f"Applied preset: {preset_name}")
    else:
        logger.warning(f"Preset '{preset_name}' not found in configuration")

    return config


def validate_all_configs(config_path: Optional[str] = None, environment: str = "prod") -> Dict[str, Any]:
    """
    Validate that all required configuration parameters are present
    for the entire system to work with ZERO hardcoded values

    Args:
        config_path: Optional path to config file/directory
        environment: Environment to validate

    Returns:
        Comprehensive validation report
    """
    config = UnifiedConfig(config_path, environment)
    validation_results = config.validate_config()

    # Additional system-wide validations
    system_validation = {
        'config_completeness': 'PASS',
        'zero_hardcoded_compliance': 'PASS',
        'critical_errors': [],
        'warnings': validation_results.get('warnings', []),
        'config_info': config.get_config_info()
    }

    # Check for critical missing sections that would cause runtime failures
    critical_sections = [
        'mathematical_operations.zero',
        'mathematical_operations.unity',
        'array_indexing.first_index',
        'array_indexing.second_index',
        'backtesting.calculations.min_trades_for_metrics',
        'backtesting.calculations.days_per_year',
        'backtesting.calculations.loop_start_index',
        'technical_analysis.signal_generation.zero_threshold',
        'volatility_prediction.feature_generation.zero_threshold'
    ]

    for section_path in critical_sections:
        sections = section_path.split('.')
        current = config.config

        try:
            for section in sections:
                current = current[section]
            if current is None:
                system_validation['critical_errors'].append(f"Missing critical config: {section_path}")
        except (KeyError, TypeError):
            system_validation['critical_errors'].append(f"Missing critical config: {section_path}")

    if system_validation['critical_errors']:
        system_validation['config_completeness'] = 'FAIL'
        system_validation['zero_hardcoded_compliance'] = 'FAIL'

    # Combine results
    full_validation = {
        'basic_validation': validation_results,
        'system_validation': system_validation,
        'summary': {
            'total_issues': len(validation_results.get('missing_sections', [])) +
                            len(validation_results.get('invalid_values', [])) +
                            len(system_validation['critical_errors']),
            'compliance_status': system_validation['zero_hardcoded_compliance'],
            'ready_for_production': system_validation['config_completeness'] == 'PASS' and
                                    len(system_validation['critical_errors']) == 0,
            'config_mode': 'multi-file' if config.multi_file_mode else 'single-file'
        }
    }

    return full_validation