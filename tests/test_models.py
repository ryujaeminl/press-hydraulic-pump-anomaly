import numpy as np
import pandas as pd

from src.models import CurrentGuard
from src.analysis import _warp
from src.errors import assign_level


def _df(cur_rms, cur_crest, y):
    return pd.DataFrame({"cur_rms": cur_rms, "cur_crest": cur_crest}), np.asarray(y)


def test_guard_fits_limit_from_normal_rows_only_and_flags_above_it():
    X, y = _df(cur_rms=[100, 100, 100, 100], cur_crest=[1.0, 2.0, 1.5, 9.0], y=[0, 0, 0, 1])
    g = CurrentGuard().fit(X, y)
    assert g.limit_ == 200.0                       # max(|cur|) over normal rows only (100*2.0), ignores the y=1 outlier
    assert list(g.flag(X)) == [0.0, 0.0, 0.0, 1.0]  # only the row strictly above the normal ceiling trips


THR = {"t_red": 0.5, "t_yellow": 0.1, "unc_high": 0.3}


def test_guard_lifts_to_yellow_but_never_to_red():
    p, unc = np.array([0.01, 0.01, 0.9]), np.zeros(3)
    assert list(assign_level(p, unc, THR)) == ["green", "green", "red"]
    assert list(assign_level(p, unc, THR, guard=[0, 1, 1])) == ["green", "yellow", "red"]   # guard=1 on a red stays red
    assert list(assign_level(p, unc, THR, guard=[1, 0, 0])) == ["yellow", "green", "red"]


def test_warp_scales_apparent_frequency_of_a_sine():
    t = np.arange(200)
    a = np.column_stack([np.sin(2 * np.pi * t / 20)] * 3)
    for f in (0.5, 2.0):
        w = _warp(a, f)
        crossings = (np.diff(np.signbit(w[:, 0]).astype(int)) != 0).sum() / len(w)
        assert abs(crossings - f * 2 / 20) < 0.02       # zero-crossings per sample scale with f
