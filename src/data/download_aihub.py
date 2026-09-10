"""
AI-Hub Dataset 71778 Automated Downloader
Downloads files from AI-Hub Open API directly into D:/paper/b/data/raw
Uses D:/paper/b/temp for temporary tar downloads and part merging.
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import tarfile
from pathlib import Path
import re

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    TEMP_DIR,
    RAW_DATA_DIR,
    DATASET_ID,
    get_temp_path,
    clean_temp_dir,
)

# AI-Hub 71778 File Catalog
FILE_CATALOG = {
    # Tabular / Metadata / Core Data
    "TL_csv": {"fileSn": "539795", "name": "TL_csv.zip", "size": "4 MB", "category": "tabular"},
    "VL_csv": {"fileSn": "539798", "name": "VL_csv.zip", "size": "583 KB", "category": "tabular"},
    "TL_gps": {"fileSn": "539796", "name": "TL_gps_data.zip", "size": "187 MB", "category": "tabular"},
    "VL_gps": {"fileSn": "539799", "name": "VL_gps_data.zip", "size": "21 MB", "category": "tabular"},
    "Other": {"fileSn": "539793", "name": "Other.zip (POI Master)", "size": "386 MB", "category": "tabular"},
    # Multimodal / Sublabels
    "SbL": {"fileSn": "549766", "name": "SbL.zip (Caption JSON)", "size": "14 GB", "category": "sublabel"},
    # Photo Raw Data (Large)
    "VS_photo": {"fileSn": "539797", "name": "VS_photo.zip", "size": "8 GB", "category": "photo"},
    "TS_photo": {"fileSn": "539794", "name": "TS_photo.zip", "size": "66 GB", "category": "photo"},
}

BASE_API_URL = "https://api.aihub.or.kr/down/0.6"


def merge_parts(target_dir: Path):
    """Merge split part files (e.g. file.zip.part0, file.zip.part1 -> file.zip)."""
    part_files = list(target_dir.glob("*.part*"))
    if not part_files:
        return

    # Group by prefix
    prefixes = set()
    for pf in part_files:
        # Match up to .part
        m = re.match(r"^(.*)\.part\d+$", pf.name)
        if m:
            prefixes.add(m.group(1))

    for prefix in prefixes:
        matched = sorted(target_dir.glob(f"{prefix}.part*"), key=lambda p: int(re.search(r"\d+$", p.name).group() if re.search(r"\d+$", p.name) else 0))
        merged_file = target_dir / prefix
        print(f"  [Merge] Merging {len(matched)} parts into {merged_file.name}...")
        with open(merged_file, "wb") as out_f:
            for part in matched:
                with open(part, "rb") as in_f:
                    while chunk := in_f.read(1024 * 1024 * 16):
                        out_f.write(chunk)
                part.unlink()  # Remove merged part
        print(f"  [Merge] Finished merging {merged_file.name}")


def download_file(file_key: str, apikey: str, dest_dir: Path) -> bool:
    """Download a specific fileSn from AI-Hub API."""
    info = FILE_CATALOG.get(file_key)
    if not info:
        print(f"[Error] Unknown file key: {file_key}")
        return False

    file_sn = info["fileSn"]
    name = info["name"]
    size = info["size"]
    print(f"\n>>> Starting download: [{file_key}] {name} (~{size})")

    download_url = f"{BASE_API_URL}/{DATASET_ID}.do?fileSn={file_sn}"
    tar_path = get_temp_path(f"download_{file_sn}.tar")

    req = urllib.request.Request(
        download_url,
        headers={
            "apikey": apikey,
            "User-Agent": "Mozilla/5.0",
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            # Check response header or first bytes
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024 * 8  # 8MB chunk

            with open(tar_path, "wb") as out_f:
                while chunk := response.read(block_size):
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        percent = downloaded / total_bytes * 100
                        print(f"\r  Downloading: {downloaded / (1024**2):.1f} MB / {total_bytes / (1024**2):.1f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  Downloading: {downloaded / (1024**2):.1f} MB", end="", flush=True)
            print()

        # Check if response was an error message
        if tar_path.stat().st_size < 500:
            content = tar_path.read_text(encoding="utf-8", errors="ignore")
            if "인증실패" in content or "권한" in content or "오류" in content:
                print(f"[FAIL] AI-Hub Error: {content.strip()}")
                tar_path.unlink(missing_ok=True)
                return False

        # Extract tar
        print(f"  [Extract] Extracting {tar_path.name} to {dest_dir}...")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(dest_dir)

        # Remove temp tar file
        tar_path.unlink(missing_ok=True)

        # Merge parts if any exist in destination
        for root_dir in [dest_dir] + list(dest_dir.rglob("*")):
            if root_dir.is_dir():
                merge_parts(root_dir)

        print(f"[SUCCESS] Download & extract completed: {name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"[FAIL] HTTP Error: {e.code} - {e.reason}")
        tar_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        tar_path.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="AI-Hub Dataset 71778 Downloader")
    parser.add_argument("--apikey", type=str, default=os.getenv("AIHUB_APIKEY", ""), help="AI-Hub Open API Key")
    parser.add_argument(
        "--target",
        type=str,
        default="tabular",
        choices=["tabular", "sublabel", "photo", "all"],
        help="Download target: tabular (~598MB), sublabel (~14GB), photo (~74GB), all (~89GB)",
    )
    parser.add_argument("--file", type=str, help="Specific file key to download (e.g. TL_csv, VL_csv)")

    args = parser.parse_args()

    if not args.apikey:
        print("\n" + "=" * 60)
        print("[!] AI-Hub API Key가 필요합니다.")
        print("  - AI-Hub(aihub.or.kr) 로그인 -> 마이페이지 -> 오픈 API 인증키 발급")
        print("  - 실행 방법:")
        print("    python src/data/download_aihub.py --apikey <발급받은_인증키> --target tabular")
        print("    (또는 환경변수 AIHUB_APIKEY 설정)")
        print("=" * 60 + "\n")
        return

    dest_dir = RAW_DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        download_file(args.file, args.apikey, dest_dir)
        return

    # Select target keys
    if args.target == "tabular":
        keys = ["TL_csv", "VL_csv", "TL_gps", "VL_gps", "Other"]
    elif args.target == "sublabel":
        keys = ["SbL"]
    elif args.target == "photo":
        keys = ["VS_photo", "TS_photo"]
    elif args.target == "all":
        keys = list(FILE_CATALOG.keys())

    print(f"\n[AI-Hub] Downloading target group: {args.target} ({len(keys)} files)")
    for k in keys:
        success = download_file(k, args.apikey, dest_dir)
        if not success:
            print(f"[Warning] Failed to download {k}. Stopping or check API key/permissions.")
            break


if __name__ == "__main__":
    main()

