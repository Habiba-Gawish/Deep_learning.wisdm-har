"""Download the WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.

Source : https://archive.ics.uci.edu/dataset/507
DOI    : 10.24432/C5HK59
Licence: CC BY 4.0
Size   : ~295 MB (zip), 51 subjects, 18 activities, 20 Hz

Usage (Command Prompt, from the project root):
    python data\\download.py

If you already have the zip, put it at data\\raw\\wisdm.zip (or the extracted
folder inside data\\raw\\) and the script will skip the download.
Raw data is NOT committed to git (see .gitignore).
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/507/"
    "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)
RAW_DIR = Path(__file__).resolve().parent / "raw"
ZIP_PATH = RAW_DIR / "wisdm.zip"


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100 * done / total_size
    sys.stdout.write(f"\rDownloading... {pct:5.1f}% ({done / 1e6:.0f}/{total_size / 1e6:.0f} MB)")
    sys.stdout.flush()


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"Zip already exists: {ZIP_PATH}")
        return
    print(f"Downloading from {URL}")
    try:
        urllib.request.urlretrieve(URL, ZIP_PATH, _progress)
    except Exception as e:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()  # remove partial file
        sys.exit(
            f"\nDownload failed: {e}\n"
            f"Download it manually from the UCI page and save it as:\n  {ZIP_PATH}"
        )
    print("\nDownload finished.")


def _extract_zip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def extract():
    """Extract the zip (and any nested zip) into data/raw/."""
    marker = RAW_DIR / ".extracted"
    if marker.exists():
        print("Already extracted.")
        return
    print("Extracting...")
    _extract_zip(ZIP_PATH, RAW_DIR)
    # The archive may contain another zip inside it; extract it too.
    for nested in list(RAW_DIR.rglob("*.zip")):
        if nested.resolve() == ZIP_PATH.resolve():
            continue
        _extract_zip(nested, nested.parent)
        nested.unlink()
    marker.touch()
    print(f"Extracted to {RAW_DIR}")


def summarize():
    """Print what ended up in data/raw so the next step (preprocess.py) can rely on it."""
    print("\nContents of data\\raw:")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_dir() and len(p.relative_to(RAW_DIR).parts) <= 3:
            n = sum(1 for f in p.iterdir() if f.is_file())
            print(f"  {p.relative_to(RAW_DIR)}\\  ({n} files)")
    for name in ("activity_key.txt",):
        found = list(RAW_DIR.rglob(name))
        print(f"  {name}: {'found' if found else 'NOT FOUND'}")


def main():
    download()
    extract()
    summarize()


if __name__ == "__main__":
    main()