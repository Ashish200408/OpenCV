"""
=============================================================================
download_model.py — Downloads MobileNet-SSD model files into models/
=============================================================================
Run once before starting the main application:
    python download_model.py
=============================================================================
"""

import os
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Model URLs ───────────────────────────────────────────────────────────────
# MobileNet-SSD trained on the COCO dataset (Caffe framework)
# These files are hosted on the official OpenCV GitHub repository.
MODEL_FILES = {
    # Note: the file is called 'deploy.prototxt' in the original repo;
    # we save it under the name our detector.py expects.
    "MobileNetSSD_deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
        "master/deploy.prototxt"
    ),
    "MobileNetSSD_deploy.caffemodel": (
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/"
        "mobilenet_iter_73000.caffemodel"
    ),
}

MODELS_DIR = Path("models")


# ── Helpers ──────────────────────────────────────────────────────────────────

class _ProgressBar(urllib.request.Request):
    """Simple progress hook used when tqdm is unavailable."""
    pass


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    bar_len = 40
    filled = int(bar_len * percent / 100)
    bar = "█" * filled + "─" * (bar_len - filled)
    sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  ")
    sys.stdout.flush()
    if downloaded >= total_size:
        print()


def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar."""
    print(f"  >>  {dest.name}")
    print(f"     {url}")
    if HAS_TQDM:
        class _TqdmHook:
            def __init__(self):
                self.pbar = None
            def __call__(self, b, bsize, tsize):
                if self.pbar is None:
                    self.pbar = tqdm(total=tsize, unit="B",
                                     unit_scale=True, ncols=70)
                self.pbar.update(b * bsize - self.pbar.n)
            def close(self):
                if self.pbar:
                    self.pbar.close()
        hook = _TqdmHook()
        urllib.request.urlretrieve(url, dest, reporthook=hook)
        hook.close()
    else:
        urllib.request.urlretrieve(url, dest, reporthook=_reporthook)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MobileNet-SSD Model Downloader")
    print("=" * 60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for filename, url in MODEL_FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [OK] {filename}  ({size_mb:.1f} MB) -- already downloaded")
            continue
        try:
            download_file(url, dest)
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [OK] Saved  {filename}  ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"\n  [FAIL] Failed to download {filename}: {exc}")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("  All model files are ready. Run: python main.py")
    else:
        print("  Some downloads failed. Check your internet connection.")
        print("  Alternatively, download the files manually (see README.md).")
    print("=" * 60)


if __name__ == "__main__":
    main()
