"""Paths, disk checks, and small file helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
REPORTS_DIR = ROOT / "reports"
TRANSACTIONS = DATA_DIR / "transactions"
CUSTOMERS = DATA_DIR / "customers"
PRODUCTS = DATA_DIR / "products"
METADATA = DATA_DIR / "metadata.json"

# Leave this much free so a shuffle spill cannot fill the disk.
DISK_FLOOR_BYTES = 500 * 1024 * 1024
TARGET_BYTES = int(1.05 * 1024**3)


def ensure_dirs() -> None:
    for path in (DATA_DIR, OUTPUT_DIR, REPORTS_DIR, REPORTS_DIR / "sql", REPORTS_DIR / "screenshots"):
        path.mkdir(parents=True, exist_ok=True)


def disk_free(path: Path | None = None) -> int:
    return shutil.disk_usage(path or ROOT).free


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fmt_bytes(num: int) -> str:
    return f"{num / 1024**3:.3f} GiB ({num / 1e9:.3f} GB, {num:,} bytes)"


def runtime_root() -> Path:
    """Prefer a symlink without spaces. Spark's local dir is picky about them."""
    link = Path("/tmp/retail-pipeline")
    try:
        if link.is_symlink() and link.resolve() == ROOT:
            return link
        if link.exists() or link.is_symlink():
            return ROOT
        link.symlink_to(ROOT)
        return link
    except OSError:
        return ROOT
