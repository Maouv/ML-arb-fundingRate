"""
data_loader_real.py
-------------------
Replaces load_fr_data() in data_loader.py for real Binance FR data.

Input: directory of CSVs named {SYMBOL}-fundingRate.csv
       columns: calc_time (unix ms), funding_interval_hours, last_funding_rate

calc_time    : Unix milliseconds UTC
last_funding_rate : DECIMAL form (0.0001 = 0.01%)
funding_interval_hours : should all be 8 — rows where != 8 are dropped

Output: standard long-format DataFrame with columns:
    timestamp (UTC datetime), symbol (str), funding_rate (decimal)
    funding_rate_pct (percent = funding_rate * 100)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_real_fr_data(
    data_dir: str | Path,
    pattern: str = "*-fundingRate.csv",
    drop_non_8h: bool = True,
    dedup_tolerance_ms: int = 5000,  # tolerate up to 5s jitter in calc_time
) -> pd.DataFrame:
    """
    Load all {SYMBOL}-fundingRate.csv from data_dir into standard long format.

    Parameters
    ----------
    data_dir : path to folder containing CSVs
    pattern  : glob pattern to match files
    drop_non_8h : drop rows where funding_interval_hours != 8
    dedup_tolerance_ms : snap timestamps to nearest 8h boundary to remove jitter dupes

    Returns
    -------
    DataFrame with columns: timestamp, symbol, funding_rate, funding_rate_pct
    Sorted by (symbol, timestamp). Index reset.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}\n"
            f"Files present: {[f.name for f in data_dir.iterdir()][:10]}"
        )

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for f in files:
        # Extract symbol from filename: BTCUSDT-fundingRate.csv → BTCUSDT
        symbol = f.stem.replace("-fundingRate", "")

        try:
            df = pd.read_csv(f)
        except Exception as e:
            skipped.append(f"{f.name}: read error ({e})")
            continue

        # ── Validate columns ──
        required = {"calc_time", "last_funding_rate"}
        missing = required - set(df.columns)
        if missing:
            skipped.append(f"{f.name}: missing columns {missing}")
            continue

        if len(df) < 10:
            skipped.append(f"{f.name}: too few rows ({len(df)})")
            continue

        # ── Drop non-8h rows ──
        if drop_non_8h and "funding_interval_hours" in df.columns:
            before = len(df)
            df = df[df["funding_interval_hours"] == 8].copy()
            dropped = before - len(df)
            if dropped > 0:
                print(f"  [{symbol}] Dropped {dropped} non-8h rows")

        # ── Parse timestamp ──
        # calc_time is unix milliseconds with jitter (e.g. 1640995200006)
        # Snap to nearest 8h boundary to normalize jitter
        df["timestamp"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
        df["timestamp"] = _snap_to_8h_boundary(df["timestamp"])

        # ── Dedup: keep first occurrence per snapped timestamp ──
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        if len(df) < before:
            print(f"  [{symbol}] Deduped {before - len(df)} duplicate timestamps")

        # ── Rename and select ──
        df = df.rename(columns={"last_funding_rate": "funding_rate"})
        df["symbol"] = symbol
        df["funding_rate_pct"] = df["funding_rate"] * 100

        # ── Sanity check FR values ──
        fr_abs_max = df["funding_rate"].abs().max()
        if fr_abs_max > 0.75:
            print(f"  [{symbol}] WARNING: max |FR| = {fr_abs_max:.4f} — unusually large")
        if df["funding_rate"].abs().max() < 1e-8:
            skipped.append(f"{f.name}: all FR values are zero")
            continue

        frames.append(df[["timestamp", "symbol", "funding_rate", "funding_rate_pct"]])

    if not frames:
        raise ValueError(
            f"No valid data loaded. Skipped files:\n" + "\n".join(skipped)
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # ── Summary ──
    print(f"\n[load_real] {len(combined):,} rows loaded")
    print(f"  Symbols : {combined['symbol'].nunique()} coins")
    print(f"  Range   : {combined['timestamp'].min().date()} → {combined['timestamp'].max().date()}")
    print(f"  FR stats (pct): mean={combined['funding_rate_pct'].mean():.4f}% "
          f"| std={combined['funding_rate_pct'].std():.4f}% "
          f"| max={combined['funding_rate_pct'].max():.4f}%")
    print(f"  Entry-worthy (|FR| >= 0.05%): "
          f"{(combined['funding_rate_pct'].abs() >= 0.05).mean():.1%} of rows")

    if skipped:
        print(f"\n  Skipped {len(skipped)} files:")
        for s in skipped[:5]:
            print(f"    {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    return combined


def _snap_to_8h_boundary(ts: pd.Series) -> pd.Series:
    """
    Snap timestamps to nearest 8h UTC boundary.
    Handles jitter like 1640995200006ms → 2022-01-01 00:00:00 UTC.

    8h boundaries: 00:00, 08:00, 16:00 UTC
    """
    # Floor to 8h
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    seconds = (ts - epoch).dt.total_seconds()
    snapped_seconds = (seconds / (8 * 3600)).round() * (8 * 3600)
    return epoch + pd.to_timedelta(snapped_seconds, unit="s")


def validate_coverage(
    df: pd.DataFrame,
    expected_interval_hours: int = 8,
    max_gap_tolerance: int = 3,  # allow up to 3 missing settlements before flagging
) -> pd.DataFrame:
    """
    Check per-symbol timestamp coverage.
    Returns DataFrame with gap stats per coin — useful to identify
    symbols with missing history before they pollute features.

    Parameters
    ----------
    max_gap_tolerance : consecutive missing settlements before flagging
    """
    rows = []
    expected_delta = pd.Timedelta(hours=expected_interval_hours)

    for symbol, grp in df.groupby("symbol"):
        ts = grp["timestamp"].sort_values()
        deltas = ts.diff().dropna()

        n_expected = int((ts.max() - ts.min()) / expected_delta) + 1
        n_actual   = len(ts)
        missing    = n_expected - n_actual

        max_gap_settlements = int(deltas.max() / expected_delta) if len(deltas) > 0 else 0
        start = ts.min().date()
        end   = ts.max().date()

        rows.append({
            "symbol": symbol,
            "start": start,
            "end": end,
            "n_actual": n_actual,
            "n_expected": n_expected,
            "missing_pct": round(missing / max(n_expected, 1) * 100, 2),
            "max_gap_settlements": max_gap_settlements,
            "flagged": missing > max_gap_tolerance or max_gap_settlements > max_gap_tolerance,
        })

    coverage = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)

    flagged = coverage[coverage["flagged"]]
    if len(flagged) > 0:
        print(f"\n[coverage] {len(flagged)} symbols flagged for gaps:")
        print(flagged[["symbol", "missing_pct", "max_gap_settlements"]].head(10).to_string(index=False))
    else:
        print(f"\n[coverage] All {len(coverage)} symbols have clean coverage")

    return coverage


def align_universe(
    df: pd.DataFrame,
    min_history_days: int = 180,
    min_coverage_pct: float = 95.0,
) -> pd.DataFrame:
    """
    Filter to coins with sufficient history and coverage.
    Coins with sparse data will corrupt rolling features.

    Parameters
    ----------
    min_history_days  : drop coins with < N days of data
    min_coverage_pct  : drop coins with > (100 - N)% missing settlements
    """
    coverage = validate_coverage(df)

    # Filter by history length
    coverage["days"] = (
        pd.to_datetime(coverage["end"]) - pd.to_datetime(coverage["start"])
    ).dt.days

    valid = coverage[
        (coverage["days"] >= min_history_days) &
        (coverage["missing_pct"] <= (100 - min_coverage_pct))
    ]["symbol"].tolist()

    before = df["symbol"].nunique()
    df_filtered = df[df["symbol"].isin(valid)].copy()
    after = df_filtered["symbol"].nunique()

    print(f"\n[align] Universe: {before} → {after} coins "
          f"(dropped {before - after} with insufficient history)")

    return df_filtered

