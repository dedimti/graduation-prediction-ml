"""
download_dataset.py
-------------------
Downloads the UCI 'Predict Students Dropout and Academic Success' dataset
and saves it to data/student_data.csv
"""

import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "student_data.csv")


def download_via_ucimlrepo():
    """Try downloading via the ucimlrepo package (preferred)."""
    try:
        from ucimlrepo import fetch_ucirepo
        print("Fetching dataset via ucimlrepo...")
        dataset = fetch_ucirepo(id=697)
        X = dataset.data.features
        y = dataset.data.targets
        import pandas as pd
        df = pd.concat([X, y], axis=1)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"  Saved: {OUTPUT_PATH}  ({len(df)} rows, {len(df.columns)} cols)")
        return True
    except Exception as e:
        print(f"  ucimlrepo failed: {e}")
        return False


def download_via_url():
    """Fallback: direct URL download."""
    import urllib.request
    URL = (
        "https://archive.ics.uci.edu/static/public/697/"
        "predict+students+dropout+and+academic+success.zip"
    )
    ZIP_PATH = os.path.join(DATA_DIR, "raw.zip")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading from UCI URL...")
    try:
        urllib.request.urlretrieve(URL, ZIP_PATH)
        import zipfile
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV found inside ZIP")
            zf.extract(csv_names[0], DATA_DIR)
            src = os.path.join(DATA_DIR, csv_names[0])
            if src != OUTPUT_PATH:
                os.rename(src, OUTPUT_PATH)
        os.remove(ZIP_PATH)
        print(f"  Saved: {OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"  URL download failed: {e}")
        return False


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at: {OUTPUT_PATH}")
        return

    ok = download_via_ucimlrepo() or download_via_url()

    if not ok:
        print("\n[ERROR] Could not download automatically.")
        print("Manual steps:")
        print("  1. Visit https://archive.ics.uci.edu/dataset/697/")
        print("  2. Download the CSV file")
        print(f"  3. Save it as: {OUTPUT_PATH}")
        sys.exit(1)

    print("\nDataset ready. Run:  python main.py")


if __name__ == "__main__":
    main()
