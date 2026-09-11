"""
Unit Tests for Machine Learning Pipeline & Data Leakage Prevention
Verifies both Traditional and Modern 2025 Scalers.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.travel_dataset import load_travel_dataset
from src.models.ml_pipeline import TravelExpenditurePipeline, AdaptiveWinsorizedScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


def test_data_leakage_prevention():
    """Verify that scaler parameters (center_, scale_) are computed SOLELY from X_train,
    and NOT influenced by X_val or X_test.
    """
    df = load_travel_dataset()
    pipeline = TravelExpenditurePipeline(df=df, random_state=42)
    pipeline.execute_split_comparison()

    X_train = pipeline.splits["3_way"]["X_train"]
    X_val = pipeline.splits["3_way"]["X_val"]
    X_test = pipeline.splits["3_way"]["X_test"]

    # 1. Test Traditional RobustScaler
    ref_scaler = RobustScaler()
    ref_scaler.fit(X_train)

    leaked_scaler = RobustScaler()
    leaked_scaler.fit(np.vstack([X_train, X_test]))

    pipeline.execute_scaling_comparison()
    pipeline_scaler = pipeline.scalers["RobustScaler [전통]"]["scaler"]

    np.testing.assert_allclose(
        pipeline_scaler.center_,
        ref_scaler.center_,
        err_msg="RobustScaler center_ deviates from pure train fit! Data leakage detected.",
    )
    np.testing.assert_allclose(
        pipeline_scaler.scale_,
        ref_scaler.scale_,
        err_msg="RobustScaler scale_ deviates from pure train fit! Data leakage detected.",
    )
    assert not np.allclose(pipeline_scaler.center_, leaked_scaler.center_), (
        "Pipeline scaler matches leaked scaler! Leakage check failed."
    )

    # 2. Test Modern 2025 Adaptive Winsorized Scaler
    modern_ref = AdaptiveWinsorizedScaler()
    modern_ref.fit(X_train)
    modern_pipeline = pipeline.scalers["Adaptive Winsorized [2025 하이브리드]"]["scaler"]

    np.testing.assert_allclose(
        modern_pipeline.lower_bounds_,
        modern_ref.lower_bounds_,
        err_msg="Modern Scaler lower bounds deviate! Data leakage detected.",
    )
    np.testing.assert_allclose(
        modern_pipeline.upper_bounds_,
        modern_ref.upper_bounds_,
        err_msg="Modern Scaler upper bounds deviate! Data leakage detected.",
    )

    print("[PASS] Data leakage prevention test passed for both Traditional and 2025 Modern Scalers!")


def test_test_set_isolation():
    """Ensure Test set shape and content remain untouched during Step 2, 3, 4."""
    df = load_travel_dataset()
    pipeline = TravelExpenditurePipeline(df=df, random_state=42)
    pipeline.execute_split_comparison()

    orig_test_shape = pipeline.splits["3_way"]["X_test"].shape
    orig_test_first_row = pipeline.splits["3_way"]["X_test"][0].copy()

    pipeline.execute_scaling_comparison()
    pipeline.execute_hyperparameter_tuning()

    assert pipeline.splits["3_way"]["X_test"].shape == orig_test_shape
    np.testing.assert_array_equal(pipeline.splits["3_way"]["X_test"][0], orig_test_first_row)
    print("[PASS] Test set isolation verified. Test set was not accessed during tuning.")


def test_evaluation_metrics():
    """Verify regression metrics are valid and strictly positive for errors."""
    df = load_travel_dataset()
    pipeline = TravelExpenditurePipeline(df=df, random_state=42)
    pipeline.execute_split_comparison()
    pipeline.execute_scaling_comparison()
    pipeline.execute_hyperparameter_tuning()
    metrics = pipeline.execute_final_evaluation()

    assert metrics["MAE"] > 0, "MAE must be positive"
    assert metrics["MSE"] > 0, "MSE must be positive"
    assert metrics["RMSE"] > 0, "RMSE must be positive"
    assert 0.0 < metrics["R2"] <= 1.0, f"R2 must be in reasonable range, got {metrics['R2']}"
    print(f"[PASS] Evaluation metrics verified: MAE={metrics['MAE']:.1f}, R2={metrics['R2']:.4f}")


if __name__ == "__main__":
    print("Running unit tests...")
    test_data_leakage_prevention()
    test_test_set_isolation()
    test_evaluation_metrics()
    print("All unit tests passed successfully!")
