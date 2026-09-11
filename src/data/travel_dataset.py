"""
Travel Log Dataset Loader and Feature Engineering Module
Supports real AI-Hub 71778 extracted tables and automated sample fallback.
Target: Total Travel Expenditure Prediction (Regression Problem)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, TABLES


def process_raw_tables(training_dir: Path, validation_dir: Path = None) -> pd.DataFrame:
    """Merge and process raw AI-Hub 71778 CSV tables into a modeling feature set."""
    def _parse_folder(folder_path: Path) -> pd.DataFrame:
        if not folder_path.exists():
            return pd.DataFrame()

        tr = pd.read_csv(folder_path / "tn_traveller_master.csv", encoding="utf-8")
        tv = pd.read_csv(folder_path / "tn_travel.csv", encoding="utf-8")
        act_c = pd.read_csv(folder_path / "tn_activity_consume_his.csv", encoding="utf-8")
        lod_c = pd.read_csv(folder_path / "tn_lodge_consume_his.csv", encoding="utf-8")
        mvm_c = pd.read_csv(folder_path / "tn_mvmn_consume_his.csv", encoding="utf-8")
        adv_c = pd.read_csv(folder_path / "tn_adv_consume_his.csv", encoding="utf-8")
        vis = pd.read_csv(folder_path / "tn_visit_area_info.csv", encoding="utf-8")

        # 1. Expenditure Aggregations per TRAVEL_ID
        act_sum = act_c.groupby("TRAVEL_ID")["PAYMENT_AMT_WON"].sum().rename("ACTIVITY_CONSUME_KRW")
        lod_sum = lod_c.groupby("TRAVEL_ID")["PAYMENT_AMT_WON"].sum().rename("LODGE_CONSUME_KRW")
        mvm_sum = mvm_c.groupby("TRAVEL_ID")["PAYMENT_AMT_WON"].sum().rename("MVMN_CONSUME_KRW")
        adv_sum = adv_c.groupby("TRAVEL_ID")["PAYMENT_AMT_WON"].sum().rename("ADV_CONSUME_KRW")

        total_exp = (
            act_sum.add(lod_sum, fill_value=0)
            .add(mvm_sum, fill_value=0)
            .add(adv_sum, fill_value=0)
        ).rename("TOTAL_EXPENDITURE")

        # 2. Visit Area Statistics per TRAVEL_ID
        vis_cnt = vis.groupby("TRAVEL_ID")["VISIT_AREA_ID"].count().rename("VISIT_AREA_CNT")
        avg_stay = vis.groupby("TRAVEL_ID")["RESIDENCE_TIME_MIN"].mean().rename("AVG_STAY_TIME")
        avg_sat = vis.groupby("TRAVEL_ID")["DGSTFN"].mean().rename("AVG_SATISFACTION")

        # 3. Base Merge
        df = tv[["TRAVEL_ID", "TRAVELER_ID", "TRAVEL_START_YMD", "TRAVEL_END_YMD"]].copy()

        # Merge traveler demographics & styles
        styl_cols = [f"TRAVEL_STYL_{i}" for i in range(1, 9)]
        tr_cols = ["TRAVELER_ID", "GENDER", "AGE_GRP", "INCOME", "TRAVEL_COMPANIONS_NUM"] + styl_cols
        df = df.merge(tr[tr_cols], on="TRAVELER_ID", how="left")

        # Merge expenditures and visits
        df = df.merge(total_exp, on="TRAVEL_ID", how="left").fillna({"TOTAL_EXPENDITURE": 0})
        df = df.merge(adv_sum, on="TRAVEL_ID", how="left").fillna({"ADV_CONSUME_KRW": 0})
        df = df.merge(vis_cnt, on="TRAVEL_ID", how="left").fillna({"VISIT_AREA_CNT": 0})
        df = df.merge(avg_stay, on="TRAVEL_ID", how="left").fillna({"AVG_STAY_TIME": 0})
        df = df.merge(avg_sat, on="TRAVEL_ID", how="left").fillna({"AVG_SATISFACTION": 0})

        # Calculate Travel Days
        s_dt = pd.to_datetime(df["TRAVEL_START_YMD"].astype(str), errors="coerce")
        e_dt = pd.to_datetime(df["TRAVEL_END_YMD"].astype(str), errors="coerce")
        df["TRAVEL_DAYS"] = (e_dt - s_dt).dt.days + 1
        df["TRAVEL_DAYS"] = df["TRAVEL_DAYS"].fillna(1).clip(lower=1)

        # Standardize Gender (0: Male, 1: Female)
        df["GENDER"] = df["GENDER"].map({"남": 0, "여": 1}).fillna(0).astype(int)

        # Select final numeric features + target
        feature_cols = [
            "GENDER",
            "AGE_GRP",
            "INCOME",
            "TRAVEL_COMPANIONS_NUM",
            "TRAVEL_DAYS",
            "VISIT_AREA_CNT",
            "AVG_STAY_TIME",
            "AVG_SATISFACTION",
            "ADV_CONSUME_KRW",
        ] + styl_cols + ["TOTAL_EXPENDITURE"]

        return df[feature_cols]

    print("[DataLoader] Merging real Training tables...")
    tr_df = _parse_folder(training_dir)

    if validation_dir and validation_dir.exists():
        print("[DataLoader] Merging real Validation tables...")
        va_df = _parse_folder(validation_dir)
        full_df = pd.concat([tr_df, va_df], ignore_index=True)
    else:
        full_df = tr_df

    # Impute any residual NaNs with column median
    full_df = full_df.fillna(full_df.median())

    return full_df


def generate_realistic_travel_data(n_samples: int = 3200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic travel dataset if raw data is unavailable."""
    np.random.seed(seed)
    gender = np.random.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    age_groups = np.random.choice([20, 30, 40, 50, 60], size=n_samples, p=[0.33, 0.30, 0.21, 0.12, 0.04])
    base_income = (age_groups // 10) - 1
    income_level = np.clip(base_income + np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.15, 0.50, 0.25, 0.10]), 1, 8)
    travel_days = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.46, 0.42, 0.09, 0.03])
    companion_cnt = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.20, 0.45, 0.20, 0.10, 0.03, 0.02])
    visited_poi_cnt = np.clip(travel_days * np.random.poisson(lam=3, size=n_samples), 1, 20)
    avg_stay_time = np.clip(np.random.normal(loc=90, scale=35, size=n_samples), 20, 300)
    avg_sat = np.clip(np.random.normal(loc=4.2, scale=0.5, size=n_samples), 1.0, 5.0)
    adv_consume = np.where(travel_days > 1, np.random.exponential(scale=60000, size=n_samples), np.random.exponential(scale=15000, size=n_samples))

    styles = {f"TRAVEL_STYL_{i}": np.random.randint(1, 8, size=n_samples) for i in range(1, 9)}

    lodge_cost = np.where(travel_days > 1, (travel_days - 1) * (companion_cnt + 1) * np.random.uniform(40000, 120000, size=n_samples), 0)
    mvmn_cost = np.random.uniform(20000, 80000, size=n_samples) * (companion_cnt + 1)
    activity_cost = visited_poi_cnt * (companion_cnt + 1) * np.random.uniform(15000, 45000, size=n_samples)
    income_multiplier = 0.7 + (income_level * 0.1)
    total_exp = (lodge_cost + mvmn_cost + activity_cost + adv_consume) * income_multiplier

    outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    total_exp[outlier_idx] *= np.random.uniform(2.5, 5.0, size=len(outlier_idx))

    data_dict = {
        "GENDER": gender,
        "AGE_GRP": age_groups,
        "INCOME": income_level,
        "TRAVEL_COMPANIONS_NUM": companion_cnt,
        "TRAVEL_DAYS": travel_days,
        "VISIT_AREA_CNT": visited_poi_cnt,
        "AVG_STAY_TIME": np.round(avg_stay_time, 1),
        "AVG_SATISFACTION": np.round(avg_sat, 1),
        "ADV_CONSUME_KRW": np.round(adv_consume, -2),
        **styles,
        "TOTAL_EXPENDITURE": np.round(total_exp, -2),
    }
    return pd.DataFrame(data_dict)


def load_travel_dataset() -> pd.DataFrame:
    """Load the dataset. Checks extracted raw CSV tables first.
    If present, merges and caches to data/processed/real_travel_dataset.csv.
    """
    real_csv_cache = PROCESSED_DATA_DIR / "real_travel_dataset.csv"
    raw_training_csv = RAW_DATA_DIR / "csv_tables" / "training" / "tn_traveller_master.csv"
    raw_validation_csv = RAW_DATA_DIR / "csv_tables" / "validation" / "tn_traveller_master.csv"

    # 1. Check if real CSV cache already exists
    if real_csv_cache.exists():
        print(f"[DataLoader] Loading cached processed real dataset: {real_csv_cache}...")
        df = pd.read_csv(real_csv_cache)
        print(f"[DataLoader] Loaded {len(df):,} rows of real AI-Hub travel data!")
        return df

    # 2. Check if extracted real CSV tables exist in RAW_DATA_DIR
    if raw_training_csv.exists():
        print(f"[DataLoader] Detected raw AI-Hub CSV tables in {RAW_DATA_DIR / 'csv_tables'}.")
        df = process_raw_tables(
            training_dir=RAW_DATA_DIR / "csv_tables" / "training",
            validation_dir=RAW_DATA_DIR / "csv_tables" / "validation" if raw_validation_csv.exists() else None,
        )
        # Cache to processed
        df.to_csv(real_csv_cache, index=False, encoding="utf-8-sig")
        print(f"[DataLoader] Successfully built and cached real dataset to {real_csv_cache} ({len(df):,} rows)!")
        return df

    # 3. Fallback to sample dataset
    sample_file = SAMPLE_DATA_DIR / "travel_expenditure_sample.csv"
    if not sample_file.exists():
        print(f"[DataLoader] Generating schema-compliant sample data to {sample_file}...")
        df_sample = generate_realistic_travel_data(n_samples=3200, seed=42)
        df_sample.to_csv(sample_file, index=False, encoding="utf-8-sig")
    else:
        print(f"[DataLoader] Loading sample dataset: {sample_file}...")
        df_sample = pd.read_csv(sample_file, encoding="utf-8-sig")

    return df_sample


if __name__ == "__main__":
    data = load_travel_dataset()
    print("\nDataset Info:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nTarget Statistics (TOTAL_EXPENDITURE):")
    print(data["TOTAL_EXPENDITURE"].describe())
