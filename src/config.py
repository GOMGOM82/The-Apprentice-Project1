"""
Central Configuration Module for Travel Log Research Project
Base Directory: D:/paper/b
Temp Directory: D:/paper/b/temp
"""

from pathlib import Path
import shutil
import os

# Base directory
BASE_DIR = Path(r"D:\paper\b").resolve()

# Dedicated Temporary Directory (Mandatory for temporary storage)
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Documentation & Workspace
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# AI-Hub Dataset 71778 Metadata
DATASET_ID = "71778"
DATASET_NAME = "국내 여행로그 데이터(동부권) (2023)"
REGION_CODE = "F"  # F = 동부권 (E=수도권, G=서부권, H=제주및도서권)

# Standard Table Names in 71778
TABLES = {
    "TRAVELLER_MASTER": "tn_traveller_master_여행객 Master_F.csv",
    "TRAVEL": "tn_travel_여행_F.csv",
    "VISIT_AREA_INFO": "tn_visit_area_info_방문지정보_F.csv",
    "ACTIVITY_HIS": "tn_activity_his_활동내역_F.csv",
    "MOVE_HIS": "tn_move_his_이동내역_F.csv",
    "COMPANION_INFO": "tn_companion_info_동반자정보_F.csv",
    "ADV_CONSUME_HIS": "tn_adv_consume_his_사전소비내역_F.csv",
    "LODGE_CONSUME_HIS": "tn_lodge_consume_his_숙박소비내역_F.csv",
    "MVMN_CONSUME_HIS": "tn_mvmn_consume_his_이동수단소비내역_F.csv",
    "ACTIVITY_CONSUME_HIS": "tn_activity_consume_his_활동소비내역_F.csv",
    "TOUR_PHOTO": "tn_tour_photo_관광사진_F.csv",
    "POI_MASTER": "tn_poi_master_POIMaster.csv",
    "GPS_COORD_PREFIX": "tn_gps_coord_",
    "CODE_SGG": "tc_sgg_시군구코드.csv",
    "CODE_A": "tc_codea_코드A.csv",
    "CODE_B": "tc_codeb_코드B.csv",
}


def get_temp_path(filename: str) -> Path:
    """Generate and return an absolute path inside D:/paper/b/temp."""
    return TEMP_DIR / filename


def clean_temp_dir() -> int:
    """Safely remove all temporary files inside TEMP_DIR.
    Returns the number of deleted items.
    """
    count = 0
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return 0

    for item in TEMP_DIR.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1
        except Exception as e:
            print(f"Warning: Failed to delete {item}: {e}")
    return count


if __name__ == "__main__":
    print(f"[Config] Project Base Dir: {BASE_DIR}")
    print(f"[Config] Temp Dir: {TEMP_DIR} (Exists: {TEMP_DIR.exists()})")
    print(f"[Config] Dataset: {DATASET_NAME} (ID: {DATASET_ID})")

