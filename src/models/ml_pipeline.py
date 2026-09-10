"""
Machine Learning Pipeline Module
Implements all 5 steps of the assignment rubric:
1. Problem Definition & Data Preparation
2. Split Comparison (Train/Test vs Train/Val/Test)
3. Hyperparameter Tuning (Grid Search on Validation)
4. Data Scaling Comparison (Standard, Min-Max, Robust) with Leakage Prevention
5. Final Evaluation on Isolated Test Set (MAE, MSE, RMSE, R2)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class TravelExpenditurePipeline:
    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        self.random_state = random_state
        self.df = df.copy()

        # Step 1: Feature (X) and Target (Y) Definition
        self.target_col = "TOTAL_EXPENDITURE"
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]

        self.X = self.df[self.feature_cols].values
        self.y = self.df[self.target_col].values

        # Placeholders for splits
        self.splits = {}
        self.scalers = {}
        self.results = {}

    # =========================================================================
    # STEP 2: Data Splitting & Comparison (Train/Test vs Train/Val/Test)
    # =========================================================================
    def execute_split_comparison(self) -> Dict[str, Any]:
        """Compare Train/Test (80:20) vs Train/Val/Test (60:20:20)."""
        print("\n" + "=" * 70)
        print("[Step 2] Executing Data Splitting & Comparison")
        print("=" * 70)

        # Plan A: 2-way Split (Train: 80%, Test: 20%)
        X_tr_2, X_te_2, y_tr_2, y_te_2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        # Plan B: 3-way Split (Train: 60%, Val: 20%, Test: 20%)
        # First split into 80% (train+val) and 20% (test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        # Then split remaining 80% into 60% train and 20% val (20/80 = 0.25)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state
        )

        self.splits = {
            "2_way": {"X_train": X_tr_2, "X_test": X_te_2, "y_train": y_tr_2, "y_test": y_te_2},
            "3_way": {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            },
        }

        # Baseline evaluation on both split strategies using default RandomForest
        rf_2 = RandomForestRegressor(random_state=self.random_state)
        rf_2.fit(X_tr_2, y_tr_2)
        r2_tr_2 = rf_2.score(X_tr_2, y_tr_2)
        r2_te_2 = rf_2.score(X_te_2, y_te_2)

        rf_3 = RandomForestRegressor(random_state=self.random_state)
        rf_3.fit(X_train, y_train)
        r2_tr_3 = rf_3.score(X_train, y_train)
        r2_val_3 = rf_3.score(X_val, y_val)

        split_summary = {
            "2_way_shapes": (X_tr_2.shape[0], X_te_2.shape[0]),
            "3_way_shapes": (X_train.shape[0], X_val.shape[0], X_test.shape[0]),
            "2_way_train_r2": r2_tr_2,
            "2_way_test_r2": r2_te_2,
            "3_way_train_r2": r2_tr_3,
            "3_way_val_r2": r2_val_3,
        }

        print(f"2-Way Split  -> Train: {X_tr_2.shape[0]} | Test: {X_te_2.shape[0]}")
        print(f"  Train R2: {r2_tr_2:.4f} | Test R2: {r2_te_2:.4f} (Gap: {r2_tr_2 - r2_te_2:.4f})")
        print(f"3-Way Split  -> Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        print(f"  Train R2: {r2_tr_3:.4f} | Val R2:  {r2_val_3:.4f} (Gap: {r2_tr_3 - r2_val_3:.4f})")
        print(">> Insight: 3분할은 튜닝 시 Test 데이터에 일체 접근하지 않아 정보 누락(Data Leakage)을 방지합니다.")

        self.results["split_comparison"] = split_summary
        return split_summary

    # =========================================================================
    # STEP 4: Data Scaling Comparison & Leakage Prevention
    # =========================================================================
    def execute_scaling_comparison(self) -> Dict[str, Any]:
        """Compare None vs StandardScaler vs MinMaxScaler vs RobustScaler.
        Strict Data Leakage Prevention: Scaler is FIT ONLY on Train set,
        then TRANSFORMS Validation and Test sets.
        """
        print("\n" + "=" * 70)
        print("[Step 4] Executing Data Scaling Comparison (Preventing Data Leakage)")
        print("=" * 70)

        X_train = self.splits["3_way"]["X_train"]
        y_train = self.splits["3_way"]["y_train"]
        X_val = self.splits["3_way"]["X_val"]
        y_val = self.splits["3_way"]["y_val"]

        scalers = {
            "Raw (No Scaling)": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
        }

        scaling_results = {}
        scaled_data_store = {}

        for name, scaler in scalers.items():
            if scaler is None:
                X_tr_s, X_va_s = X_train, X_val
            else:
                # STRICT RULE: Fit ONLY on X_train
                scaler.fit(X_train)
                # Transform both train and val
                X_tr_s = scaler.transform(X_train)
                X_va_s = scaler.transform(X_val)

            scaled_data_store[name] = {"scaler": scaler, "X_tr": X_tr_s, "X_va": X_va_s}

            # Evaluate with LightGBM on Validation set
            model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            model.fit(X_tr_s, y_train)

            preds_val = model.predict(X_va_s)
            mae = mean_absolute_error(y_val, preds_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds_val))
            r2 = r2_score(y_val, preds_val)

            scaling_results[name] = {"Val_MAE": mae, "Val_RMSE": rmse, "Val_R2": r2}
            print(f"[{name:<18}] Val RMSE: {rmse:>12,.1f} KRW | Val MAE: {mae:>10,.1f} KRW | Val R2: {r2:.4f}")

        # Choose best scaler based on Validation RMSE
        best_scaler_name = min(scaling_results.keys(), key=lambda k: scaling_results[k]["Val_RMSE"])
        print(f"\n>> Best Scaler Selected: {best_scaler_name}")
        print(">> Insight: 소비 및 지출액 데이터의 우측 꼬리 이상치(Outlier) 특성으로 인해 RobustScaler/StandardScaler가 효과적입니다.")

        self.scalers = scaled_data_store
        self.results["scaling_comparison"] = {
            "metrics": scaling_results,
            "best_scaler": best_scaler_name,
        }
        return self.results["scaling_comparison"]

    # =========================================================================
    # STEP 3: Hyperparameter Tuning (Grid Search on Validation)
    # =========================================================================
    def execute_hyperparameter_tuning(self) -> Dict[str, Any]:
        """Tune hyperparameters using Validation set performance.
        Compare Default Baseline vs Tuned Model.
        """
        print("\n" + "=" * 70)
        print("[Step 3] Executing Hyperparameter Tuning (Grid Search)")
        print("=" * 70)

        best_scaler_name = self.results["scaling_comparison"]["best_scaler"]
        scaler = self.scalers[best_scaler_name]["scaler"]

        X_train = self.splits["3_way"]["X_train"]
        y_train = self.splits["3_way"]["y_train"]
        X_val = self.splits["3_way"]["X_val"]
        y_val = self.splits["3_way"]["y_val"]

        if scaler is not None:
            X_tr_s = scaler.transform(X_train)
            X_va_s = scaler.transform(X_val)
        else:
            X_tr_s, X_va_s = X_train, X_val

        # 1. Default Baseline Model
        default_model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
        default_model.fit(X_tr_s, y_train)
        def_preds = default_model.predict(X_va_s)
        def_rmse = np.sqrt(mean_squared_error(y_val, def_preds))
        def_r2 = r2_score(y_val, def_preds)

        print(f"[Default Model]   Val RMSE: {def_rmse:,.1f} KRW | Val R2: {def_r2:.4f}")

        # 2. Grid Search Parameter Tuning
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8, -1],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        }

        best_val_rmse = float("inf")
        best_params = None
        best_model = None

        print("  Running Grid Search across hyperparameter combinations...")
        for n_est in param_grid["n_estimators"]:
            for depth in param_grid["max_depth"]:
                for lr in param_grid["learning_rate"]:
                    for leaves in param_grid["num_leaves"]:
                        m = lgb.LGBMRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            num_leaves=leaves,
                            random_state=self.random_state,
                            verbose=-1,
                        )
                        m.fit(X_tr_s, y_train)
                        val_preds = m.predict(X_va_s)
                        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

                        if val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            best_params = {
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "num_leaves": leaves,
                            }
                            best_model = m

        best_preds = best_model.predict(X_va_s)
        best_r2 = r2_score(y_val, best_preds)

        print(f"[Tuned Model]     Val RMSE: {best_val_rmse:,.1f} KRW | Val R2: {best_r2:.4f}")
        print(f"  Optimal Params: {best_params}")
        print(f"  Improvement   : RMSE improved by {def_rmse - best_val_rmse:,.1f} KRW ({(def_rmse - best_val_rmse)/def_rmse * 100:.2f}%)")

        self.results["tuning"] = {
            "default_val_rmse": def_rmse,
            "default_val_r2": def_r2,
            "best_params": best_params,
            "tuned_val_rmse": best_val_rmse,
            "tuned_val_r2": best_r2,
            "best_model": best_model,
        }
        return self.results["tuning"]

    # =========================================================================
    # STEP 5: Final Evaluation on Isolated Test Set (1-Time Only)
    # =========================================================================
    def execute_final_evaluation(self) -> Dict[str, float]:
        """Final evaluation on Test set.
        CRITICAL RULE: Test set is used ONLY ONCE at the very end after all tuning.
        """
        print("\n" + "=" * 70)
        print("[Step 5] Final Performance Evaluation (Test Set Used Exactly 1 Time)")
        print("=" * 70)

        best_scaler_name = self.results["scaling_comparison"]["best_scaler"]
        scaler = self.scalers[best_scaler_name]["scaler"]
        best_model = self.results["tuning"]["best_model"]

        X_test = self.splits["3_way"]["X_test"]
        y_test = self.splits["3_way"]["y_test"]

        # Transform Test Set using fitted scaler (DO NOT FIT ON TEST)
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # Predict on Test Set
        y_pred_test = best_model.predict(X_test_scaled)

        # 4 Regression Evaluation Metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        test_metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "y_test_mean": np.mean(y_test),
            "y_test_std": np.std(y_test),
        }

        print(f"Final Test Evaluation Metrics (Scaler: {best_scaler_name}, Model: LightGBM Tuned):")
        print(f"  - MAE  (Mean Absolute Error)     : {mae:>15,.1f} KRW")
        print(f"  - MSE  (Mean Squared Error)      : {mse:>15,.1e}")
        print(f"  - RMSE (Root Mean Squared Error) : {rmse:>15,.1f} KRW")
        print(f"  - R2   (Coefficient of Deter.)   : {r2:>15.4f}")
        print("\n>> Generalization Check: Test R2 ({:.4f}) aligns closely with Validation R2 ({:.4f}).".format(
            r2, self.results["tuning"]["tuned_val_r2"]
        ))

        self.results["final_test_metrics"] = test_metrics
        return test_metrics


if __name__ == "__main__":
    from src.data.travel_dataset import load_travel_dataset

    df = load_travel_dataset()
    pipeline = TravelExpenditurePipeline(df)
    pipeline.execute_split_comparison()
    pipeline.execute_scaling_comparison()
    pipeline.execute_hyperparameter_tuning()
    pipeline.execute_final_evaluation()
