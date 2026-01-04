import logging
from pathlib import Path

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from src.hybrid.config.unified_config import UnifiedConfig
from src.hybrid.data import DataManager
from src.hybrid.predictors.trend_duration_predictor import TrendDurationPredictor

logger = logging.getLogger(__name__)
scenarios('trend_duration_prediction.feature')

# ==============================================================================
# FIXTURES AND SETUP
# ==============================================================================
@pytest.fixture
def test_context(request):
    """A per-scenario context dict with scenario name pre-attached."""
    ctx = {}
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

# =============================================================================
# GIVEN steps
# =============================================================================

@given(parsers.parse('config files are available in {config_directory}'))
def load_configuration_file(test_context, config_directory):
    root_path = Path(__file__).parent.parent.parent.parent
    config_path = root_path / config_directory
    test_root = Path(__file__).parent.parent.parent

    assert config_path.exists(), f"Configuration file not found: {config_path}"

    unified_config = UnifiedConfig(config_path=str(config_path), environment="test")
    test_context['unified_config'] = unified_config
    test_context['test_root'] = test_root

@given(parsers.parse('data source is set to {data_path}'))
def set_data_management_source(test_context, data_path):
    unified_config = test_context['unified_config']
    data_path = test_context['test_root'] / data_path

    update_payload = {
        'data_loading': {
            'directory_path': str(data_path)
        }
    }
    unified_config.update_config(update_payload)

@given('create a TrendDurationPredictor and DataManager')
def step_trend_duration_predictor(test_context):
    unified_config = test_context['unified_config']

    predictor = TrendDurationPredictor(unified_config)
    data_manager = DataManager(unified_config)
    data_manager.load_market_data()

    test_context['predictor'] = predictor
    test_context['data_manager'] = data_manager
# =============================================================================
# WHEN steps
# =============================================================================
@when(parsers.parse('I train the duration predictor with {amount} historical elements'))
def step_train_duration_predictor(test_context, amount):
    predictor = test_context['predictor']
    data_manager = test_context['data_manager']

    points = int(amount)
    data_manager.initialize_temporal_pointer(points)

    past_data_dict = data_manager.get_past_data()
    market_id = data_manager._active_market
    past_data = past_data_dict[market_id]

    result = predictor.train(past_data)
    test_context['result'] = result

@when(parsers.parse('I predict duration on the next {count} elements'))
def step_predict_duration(test_context, count):
    predictor = test_context['predictor']
    data_manager = test_context['data_manager']

    future_data_dict = data_manager.get_future_data_preview(int(count))
    market_id = data_manager._active_market
    future_data = future_data_dict[market_id]

    predictions = predictor.predict_duration(future_data)
    test_context['predictions'] = predictions

@when(parsers.parse('I run duration chunked validation with {training_window} training window and {chunk_size:d} per chunk'))
def step_duration_chunked_validation(test_context, training_window, chunk_size):
    import time
    import numpy as np

    training_window = int(training_window)
    chunk_size = int(chunk_size)

    scenario_name = test_context.get('scenario_name', 'unknown')
    predictor = test_context['predictor']
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

        # Get past data with fixed window
        past_data_dict = data_manager.get_past_data()
        market_id = data_manager._active_market
        past_data = past_data_dict[market_id]

        if len(past_data) > training_window:
            past_data = past_data.tail(training_window)

        predictor.train(past_data)

        # Get next chunk for prediction
        future_dict = data_manager.get_future_data_preview(chunk_size)
        future_data = future_dict[market_id]

        if len(future_data) < chunk_size:
            logger.info(f"End of data at chunk {chunk_num}")
            break

        # Predict on future data
        predictions = predictor.predict_duration(future_data)

        # Calculate actuals (duration labels for future data)
        actuals = predictor.create_duration_labels(future_data)

        # Align lengths
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        # Filter valid
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        valid_preds = predictions[valid_mask]
        valid_actuals = actuals[valid_mask]

        if len(valid_actuals) > 0:
            chunk_accuracy = (valid_preds == valid_actuals).mean()
            chunk_results.append({
                'chunk': chunk_num,
                'accuracy': chunk_accuracy,
                'samples': len(valid_actuals),
            })
            all_predictions.extend(valid_preds.tolist())
            all_actuals.extend(valid_actuals.tolist())
            logger.info(f"Chunk {chunk_num}/{total_chunks}: {chunk_accuracy:.1%} accuracy ({len(valid_actuals)} samples)")

        # Move pointer forward
        current_pos = data_manager._active_market_index
        new_pos = current_pos + chunk_size
        if new_pos >= data_manager.total_records:
            break
        data_manager.set_pointer(new_pos)

    elapsed = time.time() - start_time

    # Summary
    accuracies = [r['accuracy'] for r in chunk_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    logger.warning(f"\n=== DURATION VALIDATION SUMMARY: {scenario_name} ===")
    logger.warning(f"Chunks: {len(chunk_results)}")
    logger.warning(f"Avg accuracy: {avg_accuracy:.1%}")
    logger.warning(f"Std accuracy: {std_accuracy:.1%}")
    logger.warning(f"Time: {elapsed / 60:.1f} min")

    test_context['chunk_results'] = chunk_results
    test_context['all_predictions'] = np.array(all_predictions)
    test_context['all_actuals'] = np.array(all_actuals)
    test_context['avg_accuracy'] = avg_accuracy
    test_context['std_accuracy'] = std_accuracy


# =============================================================================
# THEN steps
# =============================================================================
@then('duration predictions should be 0, 1, 2, or 3 only')
def then_duration_predictions_valid(test_context):
    predictions = test_context['predictions']
    unique_values = set(predictions)
    valid_values = {0, 1, 2, 3}
    assert unique_values.issubset(valid_values), f"Predictions should be 0-3, got {unique_values}"

@then(parsers.parse('predictions length should be {count}'))
def then_predictions_length(test_context, count):
    predictions = test_context['predictions']
    count = int(count)
    assert len(predictions) == count, f"Expected {count} predictions, got {len(predictions)}"
@then('predictor should not be marked as trained')
def then_predictor_not_trained(test_context):
    predictor = test_context['predictor']
    assert not predictor.is_trained, "Predictor should not be trained with insufficient data"
@then(parsers.parse('features should include {feature_name}'))
def then_features_include(test_context, feature_name):
    predictor = test_context['predictor']
    assert predictor.feature_names is not None, "Predictor not trained"
    assert feature_name in predictor.feature_names, \
        f"Feature '{feature_name}' not found. Available: {predictor.feature_names}"
    logger.info(f"Feature '{feature_name}' present ✓")

@then(parsers.parse('features should include {feature_name}'))
def then_features_include(test_context, feature_name):
    predictor = test_context['predictor']
    assert predictor.feature_names is not None, "Predictor not trained"
    assert feature_name in predictor.feature_names, \
        f"Feature '{feature_name}' not found. Available: {predictor.feature_names}"
    logger.info(f"Feature '{feature_name}' present ✓")

@then('duration confusion matrix should be logged')
def then_duration_confusion_matrix(test_context):
    import numpy as np

    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']
    scenario_name = test_context.get('scenario_name', 'unknown')

    # Multi-class confusion: count per category
    categories = [0, 1, 2, 3]  # very_short, short, medium, long
    category_names = ['very_short', 'short', 'medium', 'long']

    total = len(actuals)
    correct = (predictions == actuals).sum()
    accuracy = correct / total if total > 0 else 0

    logger.warning(f"\n{'=' * 60}")
    logger.warning(f"DURATION RESULTS: {scenario_name}")
    logger.warning(f"{'=' * 60}")
    logger.warning(f"Overall Accuracy: {accuracy:.1%}")
    logger.warning(f"Total samples: {total}")
    logger.warning(f"\nPer-category accuracy:")

    for cat_id, cat_name in zip(categories, category_names):
        actual_mask = actuals == cat_id
        if actual_mask.sum() > 0:
            cat_accuracy = (predictions[actual_mask] == cat_id).mean()
            cat_count = actual_mask.sum()
            logger.warning(f"  {cat_name}: {cat_accuracy:.1%} ({cat_count} samples)")

    logger.warning(f"{'=' * 60}\n")

    test_context['confusion_matrix'] = {
        'accuracy': accuracy,
        'total': total,
        'correct': correct
    }

@then('duration confusion matrix should be logged')
def then_duration_confusion_matrix(test_context):
    import json

    predictions = test_context['all_predictions']
    actuals = test_context['all_actuals']
    scenario_name = test_context.get('scenario_name', 'unknown')

    categories = [0, 1, 2, 3]
    category_names = ['very_short', 'short', 'medium', 'long']

    total = len(actuals)
    correct = int((predictions == actuals).sum())
    accuracy = correct / total if total > 0 else 0

    # Per-category stats
    category_stats = {}
    for cat_id, cat_name in zip(categories, category_names):
        actual_mask = actuals == cat_id
        if actual_mask.sum() > 0:
            cat_accuracy = float((predictions[actual_mask] == cat_id).mean())
            cat_count = int(actual_mask.sum())
            category_stats[cat_name] = {'accuracy': cat_accuracy, 'count': cat_count}

    logger.warning(f"\n{'=' * 60}")
    logger.warning(f"DURATION RESULTS: {scenario_name}")
    logger.warning(f"{'=' * 60}")
    logger.warning(f"Overall Accuracy: {accuracy:.1%}")
    logger.warning(f"Total samples: {total}")
    logger.warning(f"\nPer-category accuracy:")
    for cat_name, stats in category_stats.items():
        logger.warning(f"  {cat_name}: {stats['accuracy']:.1%} ({stats['count']} samples)")
    logger.warning(f"{'=' * 60}\n")

    # Store result to temp file
    result = {
        'scenario_name': scenario_name,
        'avg_accuracy': float(test_context['avg_accuracy']),
        'std_accuracy': float(test_context['std_accuracy']),
        'overall_accuracy': float(accuracy),
        'category_stats': category_stats,
    }

    temp_file = Path('reports/.duration_comparison_temp.json')
    temp_file.parent.mkdir(exist_ok=True)

    existing = []
    if temp_file.exists():
        existing = json.loads(temp_file.read_text())
    existing.append(result)
    temp_file.write_text(json.dumps(existing))

    # Count expected configs from Examples table - currently just 1 (baseline)
    expected_configs = 1
    if len(existing) == expected_configs:
        _log_duration_comparison(existing)
        _write_duration_report(existing)
        temp_file.unlink()


def _log_duration_comparison(results):
    logger.warning(f"\n{'=' * 60}")
    logger.warning("DURATION COMPARISON SUMMARY")
    logger.warning(f"{'=' * 60}")

    logger.warning(f"\n{'Config':<20} | {'Avg Acc':>8} | {'Std':>6}")
    logger.warning(f"{'-' * 20}-+-{'-' * 8}-+-{'-' * 6}")

    for config in results:
        logger.warning(
            f"{config['scenario_name']:<20} | {config['avg_accuracy']:>7.1%} | {config['std_accuracy']:>5.1%}")


def _write_duration_report(results):
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'reports/duration_comparison_{timestamp}.md'

    with open(filepath, 'w') as f:
        f.write('# Trend Duration Predictor Report\n\n')
        f.write(f'**Generated:** {datetime.now().isoformat()}\n\n')

        f.write('## Summary\n\n')
        f.write('| Config | Avg Acc | Std | Overall Acc |\n')
        f.write('|--------|---------|-----|-------------|\n')
        for r in results:
            f.write(
                f"| {r['scenario_name']} | {r['avg_accuracy']:.1%} | {r['std_accuracy']:.1%} | {r['overall_accuracy']:.1%} |\n")

        f.write('\n## Per-Category Accuracy\n\n')
        for r in results:
            f.write(f"### {r['scenario_name']}\n\n")
            f.write('| Category | Accuracy | Samples |\n')
            f.write('|----------|----------|--------|\n')
            for cat_name, stats in r['category_stats'].items():
                f.write(f"| {cat_name} | {stats['accuracy']:.1%} | {stats['count']} |\n")
            f.write('\n')

    logger.warning(f'Report saved: {filepath}')