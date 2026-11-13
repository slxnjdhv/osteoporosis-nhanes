"""Step 1: download the NHANES .xpt files listed in config.py from the CDC website."""
from __future__ import annotations

import time
import urllib.request

from config import COMPONENTS, CYCLES, RAW_DIR

URL_PATTERNS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{name}.XPT",
]


def is_xpt(blob: bytes) -> bool:
    return blob[:80].startswith(b"HEADER RECORD")


def fetch(name: str, cycle: str) -> bytes | None:
    year = cycle.split("-")[0]
    for pattern in URL_PATTERNS:
        url = pattern.format(year=year, cycle=cycle, name=name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read()
            if is_xpt(blob):
                return blob
        except Exception:
            pass
    return None


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for suffix, cycle in CYCLES.items():
        for comp in COMPONENTS:
            name = f"{comp}_{suffix}"
            path = RAW_DIR / f"{name}.xpt"
            if path.exists() and path.stat().st_size > 0:
                print(f"[skip] {name} already downloaded")
                continue
            blob = fetch(name, cycle)
            if blob is None:
                print(f"[MISSING] {name} ({cycle}) not found on CDC site")
                missing.append(name)
                continue
            path.write_bytes(blob)
            print(f"[ok] {name}  {len(blob) / 1e6:.1f} MB")
            time.sleep(0.5)
    if missing:
        print("\nNot downloaded:", ", ".join(missing))
        print("These variables will be missing (NaN) for those cycles.")


if __name__ == "__main__":
    main()
