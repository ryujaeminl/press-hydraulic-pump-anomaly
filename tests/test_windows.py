import numpy as np
import pandas as pd

from src.data import CH, segment
from src.features import MIN_LEN, feature_columns, make_windows


def _df(runs):
    """runs: list of run lengths; a 2 s gap between runs."""
    t, ts = pd.Timestamp("2022-07-12"), []
    for n in runs:
        ts += [t + pd.Timedelta(seconds=0.1 * i) for i in range(n)]
        t = ts[-1] + pd.Timedelta(seconds=2)
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(len(ts), 3)), columns=CH)
    d["TimeStamp"], d["Equipment_state"] = ts, 0
    return d


def test_segment_splits_on_gap_and_drops_duplicate_ts():
    d = _df([30, 20])
    d = pd.concat([d, d.iloc[[5]]])
    s = segment(d)
    assert s["seg"].nunique() == 2 and len(s) == 50


def test_windows_never_cross_gap_and_short_segments_handled():
    s = segment(_df([50, 3, 7]))
    w = make_windows(s, win=10, stride=5)
    assert set(w["seg"]) == {0, 2}                    # len-3 segment dropped (< MIN_LEN)
    assert (w["win_len"] <= 10).all() and w.loc[w.seg == 2, "win_len"].iloc[0] == 7
    assert w.loc[w.seg == 0, "pos_in_seg"].max() == 4.0  # last window aligned to segment end
    assert MIN_LEN == 5 and not w[feature_columns(w)].isna().any().any()


def test_invariant_columns_drop_amplitude_features_only():
    from src.features import invariant_columns
    w = make_windows(segment(_df([50])), 10, 5)
    inv = invariant_columns(feature_columns(w))
    assert {"cur_madr", "cur_zcr", "vib_corr"} <= set(inv)
    assert not any(c.endswith(("_rms", "_std", "_mean", "_ptp", "_mad")) for c in inv)
