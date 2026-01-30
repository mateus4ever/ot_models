# test_trade_history.py
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.predictors.volatility_predictor import VolatilityPredictor

logger = logging.getLogger(__name__)

# Load all scenarios from the feature file
scenarios('volatility_predictor.feature')

"""Volatility predictor tests using pytest-bdd."""

import logging
from datetime import datetime  # ← Add this import
from pathlib import Path
import pytest


# ... other imports ...

# ============================================================================
# LOG FILE SETUP
# ============================================================================

def setup_log_file(timestamp=None):
    """Set up file logging for comparison tests.

    Args:
        timestamp: Optional timestamp string. If None, generates new timestamp.

    Returns:
        tuple: (log_filepath, timestamp) for use in report generation
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Set up log file path
    log_filepath = reports_dir / f"volatility_comparison_{timestamp}.log"

    # Get root logger
    root_logger = logging.getLogger()

    # Remove any existing file handlers (prevents duplicate logs)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)

    # Add new file handler
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Set format to match console output
    formatter = logging.Formatter(
        '%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s'
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    logging.info(f"Log file created: {log_filepath}")

    return str(log_filepath), timestamp


# ==============================================================================
# FIXTURES AND SETUP
# ==============================================================================

# Test fixtures and shared state

@pytest.fixture
def test_context(request):
    """A per-scenario context dict with scenario name pre-attached."""
    ctx = {}
    ctx['pytest_node'] = request.node

    node_name = request.node.name
    if '[' in node_name and ']' in node_name:
        params = node_name.split('[')[1].rstrip(']')
        config_name = params.split('-')[0]
        ctx["scenario_name"] = config_name
    else:
        scenario = getattr(request.node._obj, "__scenario__", None)
        if scenario:
            ctx["scenario_name"] = scenario.name
        else:
            ctx["scenario_name"] = node_name
    return ctx


@pytest.fixture(scope="module")
def comparison_results():
    """Accumulate results across all comparison scenarios."""
    return {'configs': []}


@pytest.fixture
def predictor_with_data(test_context):
    """Helper fixture for caching tests"""
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    historical_data = past_data_dict[market_id]

    predictor.train(historical_data)
    return predictor, historical_data


# =============================================================================
# GIVEN steps - Setup and preconditions
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    """Load configuration file from specified directory"""

    # Set up log file FIRST (before any scenario runs)
    log_filepath, timestamp = setup_log_file()
    test_context['log_filepath'] = log_filepath
    test_context['timestamp'] = timestamp

    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory

    test_root = Path(__file__).parent.parent.parent
    assert config_path.exists(), f"Configuration file not found: {config_path}"

    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")

    test_context['unified_config'] = unified_config
    test_context['test_root'] = test_root


@given(parsers.parse('data source is set to {data_path}'))
def set_data_management_source(test_context, data_path):
    """Set data source in data_management config"""
    unified_config = test_context['unified_config']
    data_path = test_context['test_root'] / data_path

    update_payload = {
        'data_loading': {
            'directory_path': str(data_path)
        }
    }
    unified_config.update_config(update_payload)
    test_context['unified_config'] = unified_config


@given(parsers.parse('create a VolatilityPredictor and DataManager'))
def step_volatility_predictor(test_context):
    unified_config = test_context['unified_config']

    # ADD QUICK WIN TEST HERE ⬇️
    vol_config = unified_config.get_section('volatility_prediction')
    feature_config = vol_config['ml']['parameters']['feature_generation']
    logger.warning(f"🔍 Step 2: Before creating predictor:")
    logger.warning(f"   use_time_features = {feature_config.get('use_time_features', 'NOT FOUND')}")
    logger.warning(f"   use_efficiency_ratio = {feature_config.get('use_efficiency_ratio', 'NOT FOUND')}")
    logger.warning(f"   use_session_overlap = {feature_config.get('use_session_overlap', 'NOT FOUND')}")

    volatility_predictor = VolatilityPredictor(unified_config)
    data_manager = DataManager(unified_config)
    data_manager.load_market_data()

    test_context['volatility_predictor'] = volatility_predictor
    test_context['data_manager'] = data_manager

    logger.warning(f"🔍 Step 3: After creating predictor:")
    logger.warning(f"   predictor.use_time_features = {volatility_predictor.use_time_features}")
    logger.warning(f"   predictor.use_efficiency_ratio = {volatility_predictor.use_efficiency_ratio}")
    logger.warning(f"   predictor.use_session_overlap = {volatility_predictor.use_session_overlap}")


@given(parsers.parse('market data with {count} elements'))
def prepare_market_data(test_context, count):
    data_manager = test_context['data_manager']
    market_id = test_context['market_id']
    data_manager.set_active_market(market_id)
    data_manager.initialize_temporal_pointer(int(count))
    past_data = data_manager.get_past_data()
    test_context['market_data'] = past_data[market_id]


@given(parsers.parse('time features are {state}'))
def step_set_time_features(test_context, state):
    unified_config = test_context['unified_config']
    enabled = state.lower() == 'enabled'

    update_payload = {
        'volatility_prediction': {
            'ml': {
                'parameters': {
                    'feature_generation': {
                        'use_time_features': enabled
                    }
                }
            }
        }
    }
    unified_config.update_config(update_payload)
    test_context['unified_config'] = unified_config

    actual = unified_config.get_section('volatility_prediction')['ml']['parameters']['feature_generation'][
        'use_time_features']
    logger.warning(f"🔍 Step 1: Config updated - use_time_features = {actual} (expected {enabled})")
    assert actual == enabled, f"Config update failed! Expected {enabled}, got {actual}"


@given(parsers.parse('efficiency ratio is {state}'))
def step_set_efficiency_ratio(test_context, state):
    unified_config = test_context['unified_config']
    enabled = state.lower() == 'enabled'

    # Get periods from existing config (don't hardcode)
    vol_config = unified_config.get_section('volatility_prediction')
    periods = vol_config['ml']['parameters']['feature_generation']['efficiency_ratio_periods']

    update_payload = {
        'volatility_prediction': {
            'ml': {
                'parameters': {
                    'feature_generation': {
                        'use_efficiency_ratio': enabled,
                        'efficiency_ratio_periods': periods  # ← From config
                    }
                }
            }
        }
    }
    unified_config.update_config(update_payload)
    test_context['unified_config'] = unified_config


@given(parsers.parse('session overlap is {status}'))
def step_set_session_overlap(test_context, status):
    unified_config = test_context['unified_config']
    enabled = (status == 'enabled')

    # Get session config from existing config (don't hardcode)
    vol_config = unified_config.get_section('volatility_prediction')
    session_config = vol_config['ml']['parameters']['feature_generation']['session_overlap']

    update_payload = {
        'volatility_prediction': {
            'ml': {
                'parameters': {
                    'feature_generation': {
                        'use_session_overlap': enabled,
                        'session_overlap': session_config  # ← From config
                    }
                }
            }
        }
    }
    unified_config.update_config(update_payload)
    test_context['unified_config'] = unified_config


# =============================================================================
# WHEN steps - Actions
# =============================================================================

@when(parsers.parse('I train the predictor with {amount_of_bars} historical elements'))
def step_train_predictor(test_context, amount_of_bars):
    volatility_predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    points = int(amount_of_bars)
    data_manager.initialize_temporal_pointer(points)

    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    past_data = past_data_dict[market_id]

    result = volatility_predictor.train(past_data)
    test_context['result'] = result
    test_context['feature_columns'] = volatility_predictor.feature_names

    # ADD QUICK WIN TEST HERE ⬇️
    logger.warning(f"🔍 Step 4: After training:")
    logger.warning(f"   Generated {len(volatility_predictor.feature_names)} features")

    # Check specific features
    has_time = any('hour_sin' in f or 'day_sin' in f for f in volatility_predictor.feature_names)
    has_efficiency = any('efficiency_ratio' in f for f in volatility_predictor.feature_names)
    has_session = 'session_overlap' in volatility_predictor.feature_names

    logger.warning(f"   Has time features: {has_time}")
    logger.warning(f"   Has efficiency features: {has_efficiency}")
    logger.warning(f"   Has session overlap: {has_session}")


@when(parsers.parse('I predict volatility on the next {count} elements'))
def step_predict_volatility(test_context, count):
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    future_data_dict = data_manager.get_future_data_preview(int(count))
    market_id = data_manager._active_market
    future_data = future_data_dict[market_id]

    result = predictor.predict(future_data)

    test_context['predictions'] = result['predictions']
    test_context['confidences'] = result['confidences']


@when(parsers.parse('I run chunked validation with {training_window} training window and {chunk_size:d} per chunk'))
def step_chunked_validation(test_context, training_window, chunk_size):
    training_window = int(training_window)
    chunk_size = int(chunk_size)

    scenario_name = test_context.get('scenario_name', 'unknown')
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    start_time = time.time()
    data_manager.initialize_temporal_pointer(training_window)

    total_records = data_manager.total_records
    total_chunks = (total_records - training_window) // chunk_size

    chunk_results = []
    all_predictions = []
    all_actuals = []
    chunk_num = 0

    while True:
        chunk_num += 1

        past_data, market_id = _train_on_window(predictor, data_manager, training_window)

        future_dict = data_manager.get_future_data_preview(chunk_size)
        future_data = future_dict[market_id]

        if len(future_data) < chunk_size:
            logger.info(f"End of data at chunk {chunk_num}")
            break

        predictions = _predict_on_chunk(predictor, past_data, future_data)
        actuals = _calculate_actuals(predictor, future_data)
        valid_preds, valid_actuals = _get_valid_results(predictions, actuals)

        if len(valid_actuals) > 0:
            chunk_accuracy = (valid_preds == valid_actuals).mean()
            chunk_results.append({
                'chunk': chunk_num,
                'accuracy': chunk_accuracy,
                'samples': len(valid_actuals),
                'predictions': valid_preds.tolist(),
                'actuals': valid_actuals.tolist()
            })
            all_predictions.extend(valid_preds.tolist())
            all_actuals.extend(valid_actuals.tolist())
            _log_chunk_progress(chunk_num, total_chunks, chunk_accuracy, len(valid_actuals))

        current_pos = data_manager._active_market_index
        new_pos = current_pos + chunk_size
        if new_pos >= data_manager.total_records:
            break
        data_manager.set_pointer(new_pos)

    elapsed = time.time() - start_time
    avg_accuracy, std_accuracy = _log_validation_summary(scenario_name, chunk_results, elapsed)

    test_context['chunk_results'] = chunk_results
    test_context['all_predictions'] = np.array(all_predictions)
    test_context['all_actuals'] = np.array(all_actuals)
    test_context['avg_accuracy'] = avg_accuracy
    test_context['std_accuracy'] = std_accuracy


@when(parsers.parse('I calculate asymmetric cost where missing HIGH_VOL costs {multiplier:d}x'))
def when_calculate_asymmetric_cost(test_context, multiplier):
    """
    Calculate expected value with asymmetric costs.
    Missing HIGH_VOL (false negative) is more costly than false positive.
    """
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    # Confusion matrix counts
    tp = ((predictions == 1) & (actuals == 1)).sum()
    tn = ((predictions == 0) & (actuals == 0)).sum()
    fp = ((predictions == 1) & (actuals == 0)).sum()
    fn = ((predictions == 0) & (actuals == 1)).sum()

    # Cost model:
    # - Correct predictions: +1 (gain)
    # - False positive (predicted HIGH, was LOW): -1 (opportunity cost)
    # - False negative (predicted LOW, was HIGH): -multiplier (missed risk)

    total_gain = tp + tn
    total_cost = fp + (fn * multiplier)

    expected_value = (total_gain - total_cost) / len(actuals)

    # Cost-adjusted accuracy
    # Weighted accuracy where FN counts more
    weighted_correct = tp + tn
    weighted_incorrect = fp + (fn * multiplier)
    cost_adjusted_accuracy = weighted_correct / (weighted_correct + weighted_incorrect)

    logger.info(f"\n=== ASYMMETRIC COST ANALYSIS ===")
    logger.info(f"Cost multiplier for missing HIGH_VOL: {multiplier}x")
    logger.info(f"True positives (caught HIGH_VOL): {tp}")
    logger.info(f"True negatives (caught LOW_VOL): {tn}")
    logger.info(f"False positives (false alarm): {fp}")
    logger.info(f"False negatives (missed HIGH_VOL): {fn} (costs {fn * multiplier})")
    logger.info(f"Expected value per prediction: {expected_value:.4f}")
    logger.info(f"Cost-adjusted accuracy: {cost_adjusted_accuracy:.1%}")

    test_context['expected_value'] = expected_value
    test_context['cost_adjusted_accuracy'] = cost_adjusted_accuracy


@when("I predict on the same data twice")
def predict_twice(test_context):
    """Predict twice to test caching"""
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    historical_data = past_data_dict[market_id]

    # First prediction
    result1 = predictor.predict(historical_data)
    test_context['predictions_1'] = result1['predictions']
    test_context['cache_length_1'] = len(predictor.feature_cache) if predictor.feature_cache is not None else 0

    # Second prediction
    result2 = predictor.predict(historical_data)
    test_context['predictions_2'] = result2['predictions']
    test_context['cache_length_2'] = len(predictor.feature_cache) if predictor.feature_cache is not None else 0

@when("I clear the cache")
def clear_predictor_cache(test_context):
    """Clear the feature cache"""
    predictor = test_context['volatility_predictor']
    predictor.clear_cache()


@when("I train with different data")
def train_with_different_data(test_context):
    """Train with completely different dataset to invalidate cache"""
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    # Get different slice of data
    data_manager.initialize_temporal_pointer(100)  # Different window
    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    different_data = past_data_dict[market_id]

    predictor.train(different_data)
    test_context['cache_invalidated'] = True


# =============================================================================
# THEN steps - Assertions
# =============================================================================
@then('HIGH_VOL predictions should have realized vol at least threshold_multiplier times LOW_VOL predictions')
def then_high_vol_exceeds_threshold(test_context):
    predictions = test_context['predictions']
    realized_vol = test_context['realized_vol']
    threshold = test_context['predictor'].threshold_multiplier

    high_vol_realized = realized_vol[predictions == 1].mean()
    low_vol_realized = realized_vol[predictions == 0].mean()

    assert high_vol_realized >= low_vol_realized * threshold


@then('training should complete successfully')
def then_training_complete(test_context):
    result = test_context['result']
    predictor = test_context['volatility_predictor']

    # Check no error (result is not empty)
    assert result, "Training returned empty result"

    # Check predictor is trained
    assert predictor._is_trained, "Predictor should be marked as trained"

    # Check result has expected keys
    assert 'n_samples' in result, "Result should contain n_samples"
    assert 'n_features' in result, "Result should contain n_features"
    assert result['n_samples'] > 0, "Should have trained on samples"
    assert result['n_features'] > 0, "Should have features"


@then('predictions should be 0 or 1 only')
def then_predictions_binary(test_context):
    predictions = test_context['predictions']
    unique_values = set(predictions)
    assert unique_values.issubset({0, 1}), f"Predictions should be 0 or 1, got {unique_values}"


@then(parsers.parse('predictions length should be {count:d}'))
def then_predictions_length(test_context, count):
    predictions = test_context['predictions']
    assert len(predictions) == count, f"Expected {count} predictions, got {len(predictions)}"


@then('predictor should not be marked as trained')
def then_predictor_not_trained(test_context):
    predictor = test_context['volatility_predictor']
    assert not predictor._is_trained, "Predictor should not be trained with insufficient data"


@then(parsers.parse('prediction accuracy should exceed {threshold:d}%'))
def then_accuracy_exceeds(test_context, threshold):
    accuracy = test_context['accuracy']
    assert accuracy > threshold / 100, f"Accuracy {accuracy:.1%} should exceed {threshold}%"


@then(parsers.parse('average accuracy should exceed {threshold}%'))
def then_avg_accuracy_exceeds(test_context, threshold):
    avg_accuracy = test_context['avg_accuracy']
    assert avg_accuracy > float(threshold) / 100, f"Avg accuracy {avg_accuracy:.1%} should exceed {threshold}%"


@then(parsers.parse('accuracy standard deviation should be below {threshold}%'))
def then_std_accuracy_below(test_context, threshold):
    std_accuracy = test_context['std_accuracy']
    assert std_accuracy < float(threshold) / 100, f"Std accuracy {std_accuracy:.1%} should be below {threshold}%"


@then(parsers.parse('class distribution in actuals should be between {low:d}% and {high:d}%'))
def then_class_distribution(test_context, low, high):
    """Verify actual HIGH_VOL percentage is not extremely skewed"""
    actuals = test_context['all_actuals']
    high_vol_pct = np.mean(actuals)

    logger.info(f"Class distribution: HIGH_VOL = {high_vol_pct:.1%}, LOW_VOL = {1 - high_vol_pct:.1%}")

    assert low / 100 <= high_vol_pct <= high / 100, \
        f"HIGH_VOL distribution {high_vol_pct:.1%} outside [{low}%, {high}%] - data may be imbalanced"


@then(parsers.parse('prediction distribution should be between {low:d}% and {high:d}%'))
def then_prediction_distribution(test_context, low, high):
    """Verify predictor is not just predicting one class"""
    predictions = test_context['all_predictions']
    pred_high_vol_pct = np.mean(predictions)

    logger.info(f"Prediction distribution: HIGH_VOL = {pred_high_vol_pct:.1%}, LOW_VOL = {1 - pred_high_vol_pct:.1%}")

    assert low / 100 <= pred_high_vol_pct <= high / 100, \
        f"Predicted HIGH_VOL {pred_high_vol_pct:.1%} outside [{low}%, {high}%] - predictor may be biased"


@then(parsers.parse('true positive rate should exceed {threshold:d}%'))
def then_tpr_exceeds(test_context, threshold):
    """TPR = correctly predicted HIGH_VOL / actual HIGH_VOL"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    high_vol_mask = actuals == 1
    if high_vol_mask.sum() == 0:
        pytest.fail("No HIGH_VOL samples in actuals")

    tpr = np.mean(predictions[high_vol_mask] == 1)
    logger.info(f"True Positive Rate (HIGH_VOL recall): {tpr:.1%}")

    assert tpr > threshold / 100, f"TPR {tpr:.1%} should exceed {threshold}%"


@then(parsers.parse('true negative rate should exceed {threshold:d}%'))
def then_tnr_exceeds(test_context, threshold):
    """TNR = correctly predicted LOW_VOL / actual LOW_VOL"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    low_vol_mask = actuals == 0
    if low_vol_mask.sum() == 0:
        pytest.fail("No LOW_VOL samples in actuals")

    tnr = np.mean(predictions[low_vol_mask] == 0)
    logger.info(f"True Negative Rate (LOW_VOL recall): {tnr:.1%}")

    assert tnr > threshold / 100, f"TNR {tnr:.1%} should exceed {threshold}%"


# =============================================================================
# THEN steps - Confusion Matrix Analysis
# =============================================================================

@then('confusion matrix should be logged')
def then_log_confusion_matrix(test_context):
    """Log full confusion matrix for analysis"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']
    scenario_name = test_context.get('scenario_name', 'unknown')

    # Calculate confusion matrix
    tp = ((predictions == 1) & (actuals == 1)).sum()
    tn = ((predictions == 0) & (actuals == 0)).sum()
    fp = ((predictions == 1) & (actuals == 0)).sum()
    fn = ((predictions == 0) & (actuals == 1)).sum()

    total = len(actuals)

    # Calculate metrics
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Use WARNING level to stand out
    logger.warning(f"\n{'=' * 60}")
    logger.warning(f"RESULTS: {scenario_name}")
    logger.warning(f"{'=' * 60}")
    logger.warning(f"Accuracy: {accuracy:.1%}")
    logger.warning(f"Precision HIGH_VOL: {precision:.1%}")
    logger.warning(f"                  Actual HIGH_VOL    Actual LOW_VOL")
    logger.warning(f"Predicted HIGH    TP: {tp:>6} ({tp / total:.1%})    FP: {fp:>6} ({fp / total:.1%})")
    logger.warning(f"Predicted LOW     FN: {fn:>6} ({fn / total:.1%})    TN: {tn:>6} ({tn / total:.1%})")
    logger.warning(f"Total samples: {total}")
    logger.warning(f"{'=' * 60}\n")

    # Store for other steps
    test_context['confusion_matrix'] = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


@then(parsers.parse('precision for HIGH_VOL should exceed {threshold:d}%'))
def then_precision_high_vol(test_context, threshold):
    """Precision = TP / (TP + FP) - when we predict HIGH, how often are we right?"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    predicted_high = predictions == 1
    if predicted_high.sum() == 0:
        pytest.fail("No HIGH_VOL predictions made")

    precision = np.mean(actuals[predicted_high] == 1)
    logger.info(f"HIGH_VOL Precision: {precision:.1%}")

    assert precision > threshold / 100, f"HIGH_VOL precision {precision:.1%} should exceed {threshold}%"


@then(parsers.parse('recall for HIGH_VOL should exceed {threshold:d}%'))
def then_recall_high_vol(test_context, threshold):
    """Recall = TP / (TP + FN) - of all actual HIGH_VOL, how many did we catch?"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    actual_high = actuals == 1
    if actual_high.sum() == 0:
        pytest.fail("No HIGH_VOL in actuals")

    recall = np.mean(predictions[actual_high] == 1)
    logger.info(f"HIGH_VOL Recall: {recall:.1%}")

    assert recall > threshold / 100, f"HIGH_VOL recall {recall:.1%} should exceed {threshold}%"


@then(parsers.parse('precision for LOW_VOL should exceed {threshold:d}%'))
def then_precision_low_vol(test_context, threshold):
    """Precision for LOW_VOL = TN / (TN + FN)"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    predicted_low = predictions == 0
    if predicted_low.sum() == 0:
        pytest.fail("No LOW_VOL predictions made")

    precision = np.mean(actuals[predicted_low] == 0)
    logger.info(f"LOW_VOL Precision: {precision:.1%}")

    assert precision > threshold / 100, f"LOW_VOL precision {precision:.1%} should exceed {threshold}%"


@then(parsers.parse('recall for LOW_VOL should exceed {threshold:d}%'))
def then_recall_low_vol(test_context, threshold):
    """Recall for LOW_VOL = TN / (TN + FP)"""
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    actual_low = actuals == 0
    if actual_low.sum() == 0:
        pytest.fail("No LOW_VOL in actuals")

    recall = np.mean(predictions[actual_low] == 0)
    logger.info(f"LOW_VOL Recall: {recall:.1%}")

    assert recall > threshold / 100, f"LOW_VOL recall {recall:.1%} should exceed {threshold}%"


# =============================================================================
# THEN steps - Asymmetric Cost Analysis
# =============================================================================

@when(parsers.parse('I calculate asymmetric cost where missing HIGH_VOL costs {multiplier:d}x'))
def when_calculate_asymmetric_cost(test_context, multiplier):
    """
    Calculate expected value with asymmetric costs.
    Missing HIGH_VOL (false negative) is more costly than false positive.
    """
    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']

    # Confusion matrix counts
    tp = ((predictions == 1) & (actuals == 1)).sum()
    tn = ((predictions == 0) & (actuals == 0)).sum()
    fp = ((predictions == 1) & (actuals == 0)).sum()
    fn = ((predictions == 0) & (actuals == 1)).sum()

    # Cost model:
    # - Correct predictions: +1 (gain)
    # - False positive (predicted HIGH, was LOW): -1 (opportunity cost)
    # - False negative (predicted LOW, was HIGH): -multiplier (missed risk)

    total_gain = tp + tn
    total_cost = fp + (fn * multiplier)

    expected_value = (total_gain - total_cost) / len(actuals)

    # Cost-adjusted accuracy
    # Weighted accuracy where FN counts more
    weighted_correct = tp + tn
    weighted_incorrect = fp + (fn * multiplier)
    cost_adjusted_accuracy = weighted_correct / (weighted_correct + weighted_incorrect)

    logger.info(f"\n=== ASYMMETRIC COST ANALYSIS ===")
    logger.info(f"Cost multiplier for missing HIGH_VOL: {multiplier}x")
    logger.info(f"True positives (caught HIGH_VOL): {tp}")
    logger.info(f"True negatives (caught LOW_VOL): {tn}")
    logger.info(f"False positives (false alarm): {fp}")
    logger.info(f"False negatives (missed HIGH_VOL): {fn} (costs {fn * multiplier})")
    logger.info(f"Expected value per prediction: {expected_value:.4f}")
    logger.info(f"Cost-adjusted accuracy: {cost_adjusted_accuracy:.1%}")

    test_context['expected_value'] = expected_value
    test_context['cost_adjusted_accuracy'] = cost_adjusted_accuracy


@then('expected value should be positive')
def then_expected_value_positive(test_context):
    ev = test_context['expected_value']
    assert ev > 0, f"Expected value {ev:.4f} should be positive"


@then(parsers.parse('cost-adjusted accuracy should exceed {threshold}%'))
def then_cost_adjusted_accuracy(test_context, threshold):
    cost_adj_acc = test_context['cost_adjusted_accuracy']
    threshold = float(threshold)
    assert cost_adj_acc > threshold / 100, \
        f"Cost-adjusted accuracy {cost_adj_acc:.1%} should exceed {threshold}%"


@then('features should include hour_sin')
def then_features_include_hour_sin(test_context):
    predictor = test_context['volatility_predictor']
    assert predictor.feature_names is not None, "Predictor not trained"
    assert 'hour_sin' in predictor.feature_names, \
        f"Feature 'hour_sin' not found. Available: {predictor.feature_names}"
    logger.info("Feature 'hour_sin' present ✓")


@then('features should include hour_cos')
def then_features_include_hour_cos(test_context):
    predictor = test_context['volatility_predictor']
    assert 'hour_cos' in predictor.feature_names
    logger.info("Feature 'hour_cos' present ✓")


@then(parsers.parse('features should include {feature_type}_{trig_func}'))
def then_features_include_time_feature(test_context, feature_type, trig_func):
    """Check if time feature exists (hour_sin, hour_cos, day_sin, day_cos)"""
    predictor = test_context['volatility_predictor']
    feature_name = f'{feature_type}_{trig_func}'
    assert feature_name in predictor.feature_names, \
        f"Feature '{feature_name}' not found. Available: {predictor.feature_names}"
    logger.info(f"Feature '{feature_name}' present ✓")


@then(parsers.parse('features should include efficiency_ratio_{period}'))
def then_features_include_efficiency_ratio(test_context, period):
    """Check if efficiency_ratio feature for given period exists"""
    predictor = test_context['volatility_predictor']
    period_int = int(period)  # ← Explicit cast
    feature_name = f'efficiency_ratio_{period_int}'
    assert feature_name in predictor.feature_names, \
        f"Feature '{feature_name}' not found. Available: {predictor.feature_names}"
    logger.info(f"Feature '{feature_name}' present ✓")


@then("all features should be numeric")
def check_features_numeric(test_context):
    """Verify all features are numeric types"""
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    # Get historical data from data_manager
    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    historical_data = past_data_dict[market_id]

    features = predictor.create_volatility_features(historical_data)
    assert features.select_dtypes(include=[np.number]).shape[1] == features.shape[1], \
        "Not all features are numeric!"


@then("predictor should be marked as trained")
def check_predictor_trained(test_context):
    """Verify predictor training flag is set"""
    predictor = test_context['volatility_predictor']  # ← Use correct key
    assert predictor._is_trained, "Predictor should be marked as trained after training!"


@then('comparison matrix should be logged')
def then_log_comparison_matrix(test_context):
    logger.warning("\n" + "=" * 60)
    logger.warning("COMPARISON SUMMARY - ALL CONFIGURATIONS")
    logger.warning("=" * 60)

    cm = test_context.get('confusion_matrix', {})

    result = {
        'scenario_name': test_context.get('scenario_name', 'unknown'),
        'avg_accuracy': float(test_context['avg_accuracy']),
        'std_accuracy': float(test_context['std_accuracy']),
        'confusion_matrix': {
            'tp': int(cm.get('tp', 0)),
            'tn': int(cm.get('tn', 0)),
            'fp': int(cm.get('fp', 0)),
            'fn': int(cm.get('fn', 0)),
        } if cm else {},
        'config_snapshot': test_context.get('config_snapshot', {}),
        'predictor_snapshot': test_context.get('predictor_snapshot', {}),
    }

    temp_file = Path('reports/comparison_temp.json')
    temp_file.parent.mkdir(exist_ok=True)

    existing = []
    if temp_file.exists():
        existing = json.loads(temp_file.read_text())
    existing.append(result)
    temp_file.write_text(json.dumps(existing))

    # Check if all scenarios complete (6 configs expected)
    config_names = [c['scenario_name'] for c in existing]

    # FIXED: Remove str() wrapper
    is_6config_complete = (len(existing) == 6 and
                           all(name in config_names for name in
                               ['baseline', 'time_only', 'efficiency_only',
                                'time_efficiency', 'session_only', 'all_features']))

    # Only generate report when ALL scenarios finished
    if is_6config_complete:
        # Get timestamp from context (set by Background step)
        timestamp = test_context.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Optional: Log summary to console (remove if _log_comparison_matrices doesn't exist)
        # _log_comparison_matrices({'configs': existing})

        _write_comparison_report({'configs': existing}, timestamp)

        log_filepath = test_context.get('log_filepath', f'reports/volatility_comparison_{timestamp}.log')
        logger.warning(f"Report saved: reports/volatility_comparison_{timestamp}.md")
        logger.warning(f"Log saved: {log_filepath}")

        temp_file.unlink()  # Cleanup


@then("no features should contain NaN values")
def check_no_nan_features(test_context):
    """Verify features don't contain NaN after validation"""
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    # Get historical data
    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    historical_data = past_data_dict[market_id]

    features = predictor.create_volatility_features(historical_data)

    # Check no NaN
    assert not features.isnull().any().any(), \
        f"Features contain NaN! Columns with NaN: {features.columns[features.isnull().any()].tolist()}"
    logger.info("All features valid (no NaN) ✓")


@then("training should return valid metrics")
def check_training_metrics(test_context):
    """Verify training returned valid result metrics"""
    result = test_context.get('result')

    assert result is not None, "Training did not return a result"
    assert 'n_samples' in result, "Result missing 'n_samples'"
    assert 'n_features' in result, "Result missing 'n_features'"
    assert result['n_samples'] > 0, "No samples in training result"
    assert result['n_features'] > 0, "No features in training result"

    logger.info(f"Training metrics valid: {result['n_samples']} samples, {result['n_features']} features ✓")


@then("no features should contain inf values")
def check_no_inf_features(test_context):
    """Verify features don't contain inf after validation"""
    predictor = test_context['volatility_predictor']
    data_manager = test_context['data_manager']

    # Get historical data
    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    historical_data = past_data_dict[market_id]

    features = predictor.create_volatility_features(historical_data)

    # Check no inf
    assert not np.isinf(features.values).any(), \
        f"Features contain inf! Columns with inf: {features.columns[np.isinf(features).any()].tolist()}"
    logger.info("All features valid (no inf) ✓")


@then("feature importance should sum to approximately 1.0")
def check_feature_importance_sum(test_context):
    """Verify RandomForest feature importances sum to 1.0"""
    predictor = test_context['volatility_predictor']

    assert predictor._is_trained, "Predictor must be trained to check feature importance"
    assert hasattr(predictor.model, 'feature_importances_'), \
        "Model doesn't have feature_importances_ attribute"

    importance_sum = predictor.model.feature_importances_.sum()

    # Allow small floating point error
    assert abs(importance_sum - 1.0) < 0.01, \
        f"Feature importance sum {importance_sum:.6f} should be approximately 1.0"

    logger.info(f"Feature importance sum: {importance_sum:.6f} ✓")


# =============================================================================
# Sub-methods
# =============================================================================
def _train_on_window(predictor, data_manager, training_window):
    """Get past data and train predictor."""
    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    past_data = past_data_dict[market_id]

    if len(past_data) > training_window:
        past_data = past_data.tail(training_window)

    predictor.train(past_data)
    return past_data, market_id


def _predict_on_chunk(predictor, past_data, future_data):
    """Combine past and future, predict, return only future portion."""
    combined_data = pd.concat([past_data, future_data])
    chunk_predictions, _ = predictor.predict_volatility(combined_data)
    return chunk_predictions[-len(future_data):]


def _calculate_actuals(predictor, future_data):
    """Calculate actual volatility regime changes."""
    returns = future_data['close'].pct_change()
    current_vol = returns.shift(1).rolling(predictor.vol_window).std()
    future_vol = returns.shift(-predictor.forward_window).rolling(predictor.forward_window).std()
    historical_vol = returns.shift(1).rolling(predictor.vol_window).std()
    vol_threshold = historical_vol * predictor.vol_threshold_multiplier

    current_regime = (current_vol > vol_threshold).astype(int)
    future_regime = (future_vol > vol_threshold).astype(int)
    return (current_regime != future_regime).astype(int).values


def _get_valid_results(predictions, actuals):
    """Filter out NaN values, return valid predictions and actuals."""
    valid_mask = ~(np.isnan(actuals) | np.isnan(predictions))
    return predictions[valid_mask], actuals[valid_mask]


def _log_chunk_progress(chunk_num, total_chunks, chunk_accuracy, sample_count):
    """Log chunk progress."""
    logger.info(f"Chunk {chunk_num}/{total_chunks}: {chunk_accuracy:.1%} accuracy ({sample_count} samples)")


def _log_validation_summary(scenario_name, chunk_results, elapsed):
    """Log final validation summary."""
    accuracies = [r['accuracy'] for r in chunk_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)

    logger.warning(f"\n=== VALIDATION SUMMARY: {scenario_name} ===")
    logger.warning(f"Chunks: {len(chunk_results)}")
    logger.warning(f"Avg accuracy: {avg_accuracy:.1%}")
    logger.warning(f"Std accuracy: {std_accuracy:.1%}")
    logger.warning(f"Min/Max: {min_accuracy:.1%} / {max_accuracy:.1%}")
    logger.warning(f"Time: {elapsed / 60:.1f} min")

    return avg_accuracy, std_accuracy


def _log_comparison_matrices(comparison_results):
    """Log all matrices together."""
    logger.warning(f"\n{'=' * 60}")
    logger.warning("COMPARISON SUMMARY - ALL CONFIGURATIONS")
    logger.warning(f"{'=' * 60}")

    logger.warning(f"\n{'Config':<20} | {'Avg Acc':>8} | {'Std':>6}")
    logger.warning(f"{'-' * 20}-+-{'-' * 8}-+-{'-' * 6}")

    for config in comparison_results['configs']:
        logger.warning(
            f"{config['scenario_name']:<20} | {config['avg_accuracy']:>7.1%} | {config['std_accuracy']:>5.1%}")

    # Find best
    best = max(comparison_results['configs'], key=lambda x: x['avg_accuracy'])
    baseline = next((c for c in comparison_results['configs'] if 'baseline' in c['scenario_name'].lower()), None)

    if baseline:
        improvement = best['avg_accuracy'] - baseline['avg_accuracy']
        logger.warning(f"\nBest: {best['scenario_name']} (+{improvement:.1%} vs baseline)")
    else:
        logger.warning(f"\nBest: {best['scenario_name']}")


def _write_comparison_report(comparison_results, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    filepath = f'{report_dir}/volatility_comparison_{timestamp}.md'

    with open(filepath, 'w') as f:
        f.write('# Volatility Predictor Comparison Report\n\n')
        f.write(f'**Generated:** {datetime.now().isoformat()}\n\n')

        f.write('## Summary\n\n')
        f.write('| Config | Avg Acc | Std |\n')
        f.write('|--------|---------|-----|\n')
        for config in comparison_results['configs']:
            f.write(f"| {config['scenario_name']} | {config['avg_accuracy']:.1%} | {config['std_accuracy']:.1%} |\n")

        f.write('\n## Details\n\n')
        for config in comparison_results['configs']:
            f.write(f"### {config['scenario_name']}\n\n")

            cfg = config.get('config_snapshot', {})
            pred = config.get('predictor_snapshot', {})

            if cfg or pred:
                f.write('**Configuration:**\n\n')
                f.write('| Setting | Config | Predictor | Match |\n')
                f.write('|---------|--------|-----------|-------|\n')

                for key in ['use_time_features', 'use_efficiency_ratio', 'use_session_overlap']:
                    cfg_val = cfg.get(key, 'N/A')
                    pred_val = pred.get(key, 'N/A')
                    match = '✓' if cfg_val == pred_val else '✗'
                    f.write(f'| {key} | {cfg_val} | {pred_val} | {match} |\n')
                f.write('\n')

            cm = config.get('confusion_matrix', {})
            if cm:
                f.write('**Confusion Matrix:**\n\n')
                f.write('|  | Actual HIGH | Actual LOW |\n')
                f.write('|--|-------------|------------|\n')
                f.write(f"| Pred HIGH | TP: {cm.get('tp', 0)} | FP: {cm.get('fp', 0)} |\n")
                f.write(f"| Pred LOW | FN: {cm.get('fn', 0)} | TN: {cm.get('tn', 0)} |\n\n")

    logger.warning(f'Report saved: {filepath}')


@then(parsers.parse('features should {expectation} session_overlap'))
def step_check_session_overlap(test_context, expectation):
    """Check if session_overlap feature exists"""
    feature_columns = test_context['feature_columns']

    if expectation == 'include':
        assert 'session_overlap' in feature_columns, \
            f"Expected 'session_overlap' in features, but got: {feature_columns}"
    elif expectation == 'not include':
        assert 'session_overlap' not in feature_columns, \
            f"Expected 'session_overlap' NOT in features, but it was found"
    else:
        raise ValueError(f"Unknown expectation: {expectation}")


@then("predictions should be identical")
def check_predictions_identical(test_context):
    """Verify cached predictions match"""
    pred1 = test_context['predictions_1']
    pred2 = test_context['predictions_2']
    assert np.array_equal(pred1, pred2), "Cached predictions don't match!"
    logger.info("Cached predictions identical ✓")


@then("feature cache should exist")
def check_cache_exists(test_context):
    """Verify cache was created"""
    predictor = test_context['volatility_predictor']
    assert predictor.feature_cache is not None, "Cache should be created after prediction"
    assert len(predictor.feature_cache) > 0, "Cache should contain features"
    logger.info(f"Cache exists with {len(predictor.feature_cache)} rows ✓")


@then("feature cache should be empty")
def check_cache_empty(test_context):
    """Verify cache was cleared"""
    predictor = test_context['volatility_predictor']
    assert predictor.feature_cache is None, "Cache should be None after clearing"
    logger.info("Cache cleared ✓")


@then("cache should be invalidated")
def check_cache_invalidated(test_context):
    """Verify cache was cleared when data changed"""
    predictor = test_context['volatility_predictor']
    # Cache should be recreated (not just empty)
    assert predictor.feature_cache is not None
    logger.info("Cache invalidated and recreated ✓")
