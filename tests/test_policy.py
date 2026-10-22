import numpy as np
import pandas as pd

from src.analysis import outer_splits
from src.data import segment
from src.errors import assign_level, sequence_hit, unc_informative, uncertainty_capture
from src.pipeline import best_threshold
from src.features import make_windows
from src.models import ModeGuard


def _two_mode_frame(n=400, seed=0):
    """Low-load mode (peak ~70) and high-load mode (peak ~280): the same peak is normal in one mode, abnormal in the other."""
    rng = np.random.default_rng(seed)
    lo = pd.DataFrame({"vlow_rms": rng.normal(.05, .005, n), "cur_rms": rng.normal(50, 3, n), "cur_crest": rng.normal(1.4, .02, n)})
    hi = pd.DataFrame({"vlow_rms": rng.normal(.30, .02, n), "cur_rms": rng.normal(200, 8, n), "cur_crest": rng.normal(1.4, .02, n)})
    return pd.concat([lo, hi], ignore_index=True)


def test_mode_guard_uses_the_limit_of_the_nearest_mode_and_routes_unknown_states_to_new_mode():
    X = _two_mode_frame()
    g = ModeGuard(k=2).fit(X, np.zeros(len(X)))
    just_over = g.limits_[0] / 1.4 * 1.02                 # 2% above the low-load limit, still inside that mode's support
    probe = pd.DataFrame({"vlow_rms": [.05, .30, 5.0, .05], "cur_rms": [just_over, 200.0, 5000.0, 100.0], "cur_crest": [1.4] * 4})
    assert list(g.flag(probe)) == [1.0, 0.0, 0.0, 0.0]    # a peak far below the HIGH-load limit still trips the low-load mode
    assert list(g.new_mode(probe)) == [0.0, 0.0, 1.0, 1.0]  # no known mode matches the last two: approval, not alarm
    assert just_over / 1.4 * 1.4 < 0.5 * g.limits_[1]     # ... which a single global limit (= the high-load maximum) would miss
    assert g.limits_[0] < g.limits_[1] and g.limit_ == g.limits_[1]


def test_sequence_rules_r1_r2_r3_on_ordered_levels():
    lv = ["green", "red", "yellow", "red", "red"]
    assert sequence_hit(lv, "R1") and sequence_hit(lv, "R2") and sequence_hit(lv, "R3")
    assert not sequence_hit(["red", "green", "red"], "R2")                       # not consecutive
    assert sequence_hit(["yellow", "red"], "R3") and not sequence_hit(["yellow", "red"], "R2")
    assert sequence_hit(["red"], "R2")                                            # a one-window segment cannot show two in a row


def test_uncertainty_capture_only_licenses_the_trigger_when_errors_really_concentrate():
    rng = np.random.default_rng(1)
    unc = rng.random(2000)

    def frame(err_idx):
        err = np.full(2000, "OK", dtype=object)
        err[err_idx] = "FN"
        return pd.DataFrame({"unc": unc, "err": err, "rep": 0})

    top = np.argsort(unc)[-40:][:6]                       # 6 errors, all inside the top 2% (40 of 2000 windows)
    spread = rng.choice(2000, 6, replace=False)           # 6 errors anywhere
    assert unc_informative(uncertainty_capture(frame(top)))
    assert not unc_informative(uncertainty_capture(frame(spread)))


def test_outer_splits_never_share_a_time_block_and_every_test_fold_has_anomalies():
    w = pd.DataFrame({"block": np.repeat(np.arange(60), 5), "y": np.repeat([1] * 6 + [0] * 54, 5)})
    for tr, te in outer_splits(w, n_outer=3, seed=0):
        assert not set(w.block.iloc[tr]) & set(w.block.iloc[te])
        assert w.y.iloc[te].sum() > 0


def test_every_window_of_a_segment_lands_in_the_same_cv_group_even_across_a_30s_boundary():
    n = 700                                              # 70 s of continuous 10 Hz samples: crosses two 30 s block boundaries
    raw = pd.DataFrame({"TimeStamp": pd.Timestamp("2022-07-12") + pd.to_timedelta(np.arange(n) * 0.1, unit="s"),
                        "AI0_Vibration": np.random.default_rng(0).normal(size=n), "AI1_Vibration": np.arange(n) % 7 * .1,
                        "AI2_Current": np.sin(np.arange(n) / 3), "Equipment_state": 0})
    seg = segment(raw)
    assert seg.block.nunique() > 1                        # row-level blocks do split the segment ...
    w = make_windows(seg, 10, 5)
    assert w.seg.nunique() == 1 and w.block.nunique() == 1   # ... but its windows never do


def test_best_threshold_is_the_f1_optimum_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.10, 0.20, 0.30, 0.60, 0.70, 0.80])
    assert best_threshold(y, p) == 0.60                   # first score that keeps all anomalies and drops all normals


def test_a_switched_off_uncertainty_trigger_never_lifts_a_window():
    thr = {"t_red": 0.5, "t_yellow": 0.1, "unc_high": 1e9}       # alarm_policy writes 1e9 when uncertainty fails its check
    assert list(assign_level([0.01, 0.01], [0.9, 0.0], thr)) == ["green", "green"]
    assert list(assign_level([0.01, 0.01], [0.9, 0.0], {**thr, "unc_high": 0.5})) == ["yellow", "green"]
