"""
Environment and Workspace Verification Test
Verifies directory structure, temp folder functionality, and config module.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    TEMP_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    DOCS_DIR,
    NOTEBOOKS_DIR,
    get_temp_path,
    clean_temp_dir,
)


def test_directories():
    assert BASE_DIR.exists(), f"Base directory does not exist: {BASE_DIR}"
    assert TEMP_DIR.exists(), f"Temp directory does not exist: {TEMP_DIR}"
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert SAMPLE_DATA_DIR.exists()
    assert DOCS_DIR.exists()
    assert NOTEBOOKS_DIR.exists()
    print("[PASS] All required directories verified.")


def test_temp_file_operations():
    # Test creating temporary file via get_temp_path
    test_file = get_temp_path("test_temp_check.tmp")
    test_file.write_text("temporary data check", encoding="utf-8")
    assert test_file.exists(), "Failed to create temporary file in temp directory"
    assert test_file.parent == TEMP_DIR, "Temporary file is not in TEMP_DIR"

    content = test_file.read_text(encoding="utf-8")
    assert content == "temporary data check"

    # Test clean_temp_dir
    cleaned_count = clean_temp_dir()
    assert not test_file.exists(), "Failed to clean temporary file"
    assert cleaned_count >= 1, "Clean count should be at least 1"
    print(f"[PASS] Temp file creation and cleanup verified (cleaned {cleaned_count} items).")


if __name__ == "__main__":
    print(f"Running verification on {BASE_DIR}...")
    test_directories()
    test_temp_file_operations()
    print("Environment setup is fully ready.")

