"""Sliding-window feature extraction (windows never cross a time gap)."""
import numpy as np
import pandas as pd
from scipy import stats

from .data import CH, SAMPLE_DT

SHORT = {"AI0_Vibration": "vup", "AI1_Vibration": "vlow", "AI2_Current": "cur"}
MIN_LEN = 5          # segments shorter than this are dropped (too few samples for any statistic)


def _channel_feats(x: np.ndarray, p: str) -> dict:
    n = len(x)
    xc = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    zc = float(np.mean(np.signbit(xc[1:]) != np.signbit(xc[:-1]))) if n > 1 else 0.0
    return {
        f"{p}_mean": float(x.mean()), f"{p}_std": float(x.std()), f"{p}_rms": rms,
        f"{p}_ptp": float(np.ptp(x)),
        f"{p}_skew": float(np.nan_to_num(stats.skew(x), nan=0.0)),
        f"{p}_kurt": float(np.nan_to_num(stats.kurtosis(x), nan=0.0)),
        f"{p}_crest": float(np.abs(x).max() / rms) if rms > 0 else 0.0,
        f"{p}_mad": float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0,
        f"{p}_madr": float(np.mean(np.abs(np.diff(x))) / (x.std() + 1e-9)) if n > 1 else 0.0,   # smoothness, scale-free
        f"{p}_zcr": zc,
    }


def window_features(w: np.ndarray) -> dict:
    """w: (n, 3) array in CH order -> feature dict."""
    f = {}
    for j, c in enumerate(CH):
        f.update(_channel_feats(w[:, j], SHORT[c]))
    c01 = np.corrcoef(w[:, 0], w[:, 1])[0, 1] if len(w) > 2 else 0.0
    f["vib_corr"] = float(np.nan_to_num(c01))
    f["vib_ratio"] = f["vup_rms"] / (f["vlow_rms"] + 1e-9)
    f["vib_energy"] = f["vup_rms"] ** 2 + f["vlow_rms"] ** 2
    f["cur_dc_ratio"] = abs(f["cur_mean"]) / (f["cur_rms"] + 1e-9)
    return f


def make_windows(df: pd.DataFrame, win: int = 10, stride: int = 5) -> pd.DataFrame:
    """One row per window. Segments with MIN_LEN <= len < win yield a single short window."""
    rows = []
    has_label = "Equipment_state" in df
    for seg, g in df.groupby("seg", sort=True):
        n = len(g)
        if n < MIN_LEN:
            continue
        arr = g[CH].to_numpy(float)
        ts = g["TimeStamp"].to_numpy()
        starts = [0] if n <= win else sorted(set(range(0, n - win + 1, stride)) | {n - win})
        for s in starts:
            e = min(s + win, n)
            r = window_features(arr[s:e])
            r.update(seg=seg, block=int(g["block"].iloc[0]), t_start=ts[s], t_end=ts[e - 1],
                     pos_in_seg=s * SAMPLE_DT, seg_len=n, win_len=e - s)
            if has_label:
                r["y"] = int(g["Equipment_state"].iloc[0])
                r["source"] = g["source"].iloc[0] if "source" in g else ""
            rows.append(r)
    return pd.DataFrame(rows)


FEATURE_PREFIX = ("vup_", "vlow_", "cur_", "vib_")


INVARIANT_SUFFIX = ("_zcr", "_skew", "_kurt", "_crest", "_madr")
INVARIANT_EXTRA = ("vib_corr", "cur_dc_ratio")


def invariant_columns(cols) -> list:
    """Scale-free shape/dynamics features: unchanged when a normal operating mode only rescales amplitudes."""
    return [c for c in cols if c.endswith(INVARIANT_SUFFIX) or c in INVARIANT_EXTRA]


def feature_columns(w: pd.DataFrame) -> list:
    """Model inputs only (meta/bookkeeping columns such as seg, block, mode, p are never features)."""
    return [c for c in w.columns if c.startswith(FEATURE_PREFIX)]
