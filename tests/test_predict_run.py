import numpy as np
import pandas as pd

from predict import MAX_WAVE, segment_wave, timeline


def _windows(levels, p, seg=None):
    n = len(levels)
    return pd.DataFrame({"seg": seg if seg is not None else [0] * n, "p_anomaly": p, "level": levels,
                         "pos_in_seg": np.arange(n) * 0.5, "win_len": 10})


def test_timeline_max_pools_a_long_run_without_losing_its_one_red_window():
    n = 9000
    levels = ["green"] * n
    levels[4321] = "red"
    p = np.full(n, 0.01)
    p[4321] = 0.9
    t = timeline(_windows(levels, p), n_max=4000)
    assert t["step"] == 3 and len(t["p"]) == len(t["lv"]) == 3000        # 9000 windows -> 3 per point
    assert 2 in t["lv"] and max(t["p"]) == 0.9                            # the alarm survives the pooling
    assert t["seg_at"] == [0]


def test_segment_wave_is_cropped_around_the_peak_window_and_keeps_window_levels_inside_the_crop():
    n_samples = 2000
    sdf = pd.DataFrame({"seg": 7, "AI0_Vibration": np.arange(n_samples) * 0.001, "AI1_Vibration": 0.0, "AI2_Current": 1.0})
    levels = ["green"] * 399
    levels[300] = "red"                                                   # window 300 starts at sample 1500
    p = np.full(399, 0.01)
    p[300] = 0.99
    wave = segment_wave(sdf, _windows(levels, p, seg=[7] * 399), 7)
    assert len(wave["ai0"]) == len(wave["ai1"]) == len(wave["ai2"]) == MAX_WAVE
    assert all(0 <= a < b <= MAX_WAVE for a, b, _ in wave["wins"])
    assert [295, 305, 2] in wave["wins"]                                  # 1500..1510 minus the crop start (1205)
    assert wave["t0"] == 120.5
