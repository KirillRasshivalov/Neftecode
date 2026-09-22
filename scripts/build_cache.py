"""Build the parquet cache in `data/cache/` from the organisers' raw package in `data/raw/`.

Everything downstream reads the cache, never the raw files (CLAUDE.md §2 rule 9, §11).
The cache is a **raw parse**: sentinels are not stripped and nothing is cleaned —
`scripts/realdata.py` does that, in one place. The parsers are the ones the Phase 1
data audit used, so a rebuilt cache reproduces every number in this repository.

Put into `data/raw/`:

    242000_tags.csv                        24-2000 telemetry
    ЛИМС*.xlsx                             lab results      («ЛИМСы 01.01.2023 - н.в_.xlsx»)
    *ПАК*.xlsx                             online analyzers («Выгрузка ПАК 01.01.2023 - н.в_.xlsx»)

then run once:

    python -m scripts.build_cache           # skips files already built
    python -m scripts.build_cache --force   # rebuilds everything

The AVT telemetry is not built: nothing in the decision path reads it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import realdata as rd

RAW_DIR = rd.REPO_ROOT / "data" / "raw"

TELEMETRY_CSV = "242000_tags.csv"
LIMS_PATTERN = "ЛИМС*.xlsx"
PAK_PATTERN = "*ПАК*.xlsx"


def find(pattern: str, raw_dir: Path = RAW_DIR) -> Path:
    """The one raw file matching `pattern`; a clear error when there is none or several."""
    matches = sorted(p for p in raw_dir.glob(pattern) if not p.name.startswith("~$"))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern!r} in {raw_dir}. Put the organisers' package there "
            f"(see the docstring of scripts/build_cache.py)."
        )
    if len(matches) > 1:
        raise FileExistsError(f"Several files match {pattern!r} in {raw_dir}: {[m.name for m in matches]}")
    return matches[0]


def parse_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def parse_lims(path: Path) -> pd.DataFrame:
    """LIMS wide blocks to long form.

    Layout, verified in the audit: row 0 the sampling point (merged over its block),
    row 1 the parameter, row 2 the unit, row 3 the stated count, rows 4 onwards pairs of
    (timestamp, value) columns.
    """
    raw = pd.read_excel(path, sheet_name="Лист1", header=None)
    points = raw.iloc[0].ffill()
    params, units, counts = raw.iloc[1], raw.iloc[2], raw.iloc[3]
    blocks = []
    for c in range(0, raw.shape[1], 2):
        param = params.iat[c]
        if pd.isna(param):
            continue
        ts = pd.to_datetime(raw.iloc[4:, c], errors="coerce")
        val = pd.to_numeric(raw.iloc[4:, c + 1], errors="coerce")
        keep = ts.notna() & val.notna()
        if not keep.any():
            continue
        blocks.append(pd.DataFrame({
            "point": str(points.iat[c]).strip(),
            "param": str(param).strip(),
            "unit": "" if pd.isna(units.iat[c]) else str(units.iat[c]).strip(),
            "stated_count": np.nan if pd.isna(counts.iat[c + 1]) else float(counts.iat[c + 1]),
            "timestamp": ts[keep].to_numpy(),
            "value": val[keep].to_numpy(),
        }))
    out = pd.concat(blocks, ignore_index=True).sort_values(["point", "param", "timestamp"])
    return out.reset_index(drop=True)


def parse_pak(path: Path) -> pd.DataFrame:
    """Online-analyzer wide blocks to long form: (signal, unit, timestamp, value)."""
    raw = pd.read_excel(path, sheet_name="Лист1", header=None)
    names, units = raw.iloc[0], raw.iloc[1]
    blocks = []
    for c in range(0, raw.shape[1], 3):
        name = names.iat[c]
        if pd.isna(name):
            continue
        ts = pd.to_datetime(raw.iloc[2:, c], errors="coerce")
        val = pd.to_numeric(raw.iloc[2:, c + 1], errors="coerce")
        keep = ts.notna() & val.notna()
        blocks.append(pd.DataFrame({
            "signal": str(name).strip(),
            "unit": "" if pd.isna(units.iat[c]) else str(units.iat[c]).strip(),
            "timestamp": ts[keep].to_numpy(),
            "value": val[keep].to_numpy(),
        }))
    out = pd.concat(blocks, ignore_index=True).sort_values(["signal", "timestamp"])
    return out.reset_index(drop=True)


#: cache name -> (how to find the raw file, how to parse it)
SOURCES = {
    "tags_242000": (lambda raw_dir: raw_dir / TELEMETRY_CSV, parse_telemetry),
    "lims_long": (lambda raw_dir: find(LIMS_PATTERN, raw_dir), parse_lims),
    "pak_long": (lambda raw_dir: find(PAK_PATTERN, raw_dir), parse_pak),
}


def build(raw_dir: Path = RAW_DIR, cache_dir: Path = rd.CACHE_DIR, force: bool = False) -> dict[str, str]:
    """Build every missing cache file; returns what happened to each."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (locate, parse) in SOURCES.items():
        target = cache_dir / f"{name}.parquet"
        if target.exists() and not force:
            report[name] = "already built"
            continue
        source = locate(raw_dir)
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing — put the organisers' package into {raw_dir}")
        frame = parse(source)
        frame.to_parquet(target, index=False)
        report[name] = f"built from {source.name}: {len(frame):,} rows"
    return report


def main(argv: list[str] | None = None) -> None:
    rd.use_utf8_stdout()
    parser = argparse.ArgumentParser(prog="build_cache", description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild files that already exist")
    args = parser.parse_args(argv)
    for name, what in build(force=args.force).items():
        print(f"{name:<12} {what}")


if __name__ == "__main__":
    main()
