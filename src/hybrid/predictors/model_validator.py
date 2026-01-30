import numpy as np


class ModelValidator:
    """Reusable validation for all ML models - prevents silent failures"""

    @staticmethod
    def validate_dataframe(df, name="DataFrame"):
        """Validate DataFrame has no NaN/inf"""
        assert not df.isnull().any().any(), f"{name} contains NaN!"
        assert not np.isinf(df).any().any(), f"{name} contains inf!"
        assert len(df) > 0, f"{name} is empty!"

    @staticmethod
    def validate_predictions(predictions, name="Predictions"):
        """Validate model output"""
        assert len(predictions) > 0, f"{name} is empty!"
        assert not np.isnan(predictions).any(), f"{name} contains NaN!"
        assert not np.isinf(predictions).any(), f"{name} contains inf!"

    @staticmethod
    def validate_range(values, min_val, max_val, name="Values"):
        """Validate values are in expected range"""
        assert np.all(values >= min_val), f"{name} below minimum {min_val}"
        assert np.all(values <= max_val), f"{name} above maximum {max_val}"

    @staticmethod
    def validate_binary(predictions, name="Binary predictions"):
        """Validate binary classifier output"""
        unique_vals = np.unique(predictions)
        assert len(unique_vals) <= 2, f"{name} has more than 2 values: {unique_vals}"
        assert np.all(np.isin(predictions, [0, 1])), f"{name} contains values other than 0/1"