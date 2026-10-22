"""Load raw CSVs, drop duplicated timestamps, split into continuous segments, assign CV time blocks."""
from pathlib import Path

import numpy as np
import pandas as pd

CH = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
FILES = ("press_data_normal.csv", "outlier_data.csv")
SAMPLE_DT = 0.1      # seconds (10 Hz)
GAP_S = 0.15         # dt above this starts a new continuous segment
BLOCK_S = 30.0       # CV group = 30 s wall-clock block


def segment(d: pd.DataFrame, file_idx: int = 0) -> pd.DataFrame:
    """Return copy with `seg` (continuous run id) and `block` (30 s time block) columns."""
    d = d.sort_values("TimeStamp").drop_duplicates("TimeStamp").reset_index(drop=True)
    gap = d["TimeStamp"].diff().dt.total_seconds() > GAP_S
    out = d.copy()
    out["seg"] = file_idx * 10_000 + gap.cumsum()
    t = (d["TimeStamp"] - d["TimeStamp"].iloc[0]).dt.total_seconds()
    out["block"] = file_idx * 10_000 + (t // BLOCK_S).astype(int)
    return out


def read_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["TimeStamp"])
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]          # drop the saved pandas index column
    missing = set(CH + ["TimeStamp"]) - set(d.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return d


def load_training(data_dir) -> pd.DataFrame:
    """Normal (state 0) + outlier (state 1) files, stacked with globally unique seg/block ids."""
    parts = []
    for i, name in enumerate(FILES):
        path = Path(data_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (use --data to point at the dataset folder)")
        d = segment(read_csv(path), i)
        d["source"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def quality_report(data_dir) -> pd.DataFrame:
    """Per-file data-quality facts used in the report (raw file, before dedup)."""
    rows = []
    for name in FILES:
        raw = read_csv(Path(data_dir) / name)
        seg = segment(raw)
        dt = raw["TimeStamp"].diff().dt.total_seconds()
        gaps = dt[dt > GAP_S]
        lens = seg.groupby("seg").size()
        starts = seg.groupby("seg")["TimeStamp"].min().sort_values().diff().dt.total_seconds().dropna()   # start-to-start spacing of bursts
        z = ((raw[CH] - raw[CH].mean()) / raw[CH].std()).abs()
        rows.append({
            "file": name, "rows": len(raw),
            "state_values": ",".join(map(str, sorted(raw["Equipment_state"].unique()))),
            "missing_cells": int(raw[CH + ["TimeStamp"]].isna().sum().sum()),
            "duplicated_timestamps": int(raw["TimeStamp"].duplicated().sum()),
            "start": raw["TimeStamp"].min(), "end": raw["TimeStamp"].max(),
            "wallclock_min": round((raw["TimeStamp"].max() - raw["TimeStamp"].min()).total_seconds() / 60, 1),
            "recorded_min": round(len(seg) * SAMPLE_DT / 60, 1),
            "segments": len(lens), "seg_len_min": int(lens.min()), "seg_len_median": float(lens.median()),
            "seg_len_max": int(lens.max()), "segments_full50": int((lens == 50).sum()),
            "gap_median_s": round(float(gaps.median()), 2), "gap_max_s": round(float(gaps.max()), 2),
            "seg_start_interval_median_s": round(float(starts.median()), 2),
            "seg_start_interval_q10_s": round(float(starts.quantile(.1)), 2), "seg_start_interval_q90_s": round(float(starts.quantile(.9)), 2),
            "abs_z_gt5_cells": int((z > 5).sum().sum()),
        })
    return pd.DataFrame(rows)
