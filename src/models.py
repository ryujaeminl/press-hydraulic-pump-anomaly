"""Candidate detectors. Each exposes fit(X, y) and margin(X): real-valued anomaly score (higher = more anomalous).

Unsupervised members (SPC, IsolationForest) are fitted on normal rows only (y == 0).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLIP = 12.0
SPC_COLS = ["vup_rms", "vlow_rms", "cur_rms"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class SPCBaseline:
    """Field-style control chart: robust z of channel RMS (median/MAD of normal), score = worst channel."""

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0, SPC_COLS]
        self.med_ = n.median()
        self.mad_ = (n - self.med_).abs().median() * 1.4826 + 1e-9
        return self

    def margin(self, X):
        return ((X[SPC_COLS] - self.med_) / self.mad_).max(axis=1).to_numpy()


class _Proba:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def margin(self, X):
        return _logit(self.est.predict_proba(X)[:, 1])


class LogReg:
    def __init__(self):
        self.est = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.3, max_iter=3000))

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def margin(self, X):
        return self.est.decision_function(X)


class IsoForest:
    """Normal-only detector: needs no anomaly labels, so it can flag unseen anomaly types."""

    def __init__(self, seed=0):
        self.est = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.est.fit(X.loc[np.asarray(y) == 0])
        return self

    def margin(self, X):
        return -self.est.score_samples(X)


class OnCols:
    """Restrict a model to a column subset (the SPC rule always reads its own RMS columns instead)."""

    def __init__(self, est, cols):
        self.est, self.cols = est, list(cols)

    def fit(self, X, y):
        self.est.fit(X[self.cols], y)
        return self

    def margin(self, X):
        return self.est.margin(X[self.cols])


class CurrentGuard:
    """Deterministic safety rule, independent of any learned model: flags a window if its peak |current|
    exceeds the equipment's observed normal ceiling. The invariant (scale-free) features cannot see this
    by construction (that is the point of them), so a current spike riding on an otherwise-smooth waveform
    can slip past M2_HGB. It only raises a window to YELLOW (errors.assign_level): the limit is the worst normal
    value seen in training, not a nameplate rating, so on a legitimately new operating mode it over-fires.
    Reads cur_rms/cur_crest directly from X regardless of `cols` (same pattern as
    SPCBaseline), so it always sees the real amplitude even when the model only sees shape features.
    """

    def __init__(self, margin: float = 1.0):
        self.margin = margin   # 1.0 = exactly the worst normal-window peak seen in training

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0]
        self.limit_ = float((n.cur_rms * n.cur_crest).max()) * self.margin
        return self

    def flag(self, X):
        return ((X.cur_rms * X.cur_crest) > self.limit_).to_numpy(dtype=float)


class ModeGuard:
    """Operating-mode-aware peak-current guard: mode classifier -> per-mode control limit.

    Normal windows are clustered on (log lower-vibration RMS, log current RMS) - the same operating-mode descriptor
    the report's mode diagnosis uses. Each mode gets its own limit = the worst peak |current| (cur_rms*cur_crest)
    seen in that mode's normal windows. A window is
      * guard-flagged  when its peak exceeds the limit of its NEAREST known mode AND it lies inside that mode's support
      * new-mode       when it lies outside the support of every known mode (distance to the nearest centroid >
                       support_mult * support_q-quantile of that mode's own training distances). That is not an alarm:
                       it is a "operator must approve / retrain" status, because no limit exists for such a state.
    Like CurrentGuard it reads raw amplitude columns regardless of the model's feature subset, and it only ever
    raises a window to YELLOW (errors.assign_level).
    """

    def __init__(self, k: int = 2, support_q: float = 0.99, support_mult: float = 1.5, seed: int = 0):
        self.k, self.support_q, self.support_mult, self.seed = k, support_q, support_mult, seed

    @staticmethod
    def _z(X):
        return np.log(np.clip(X[["vlow_rms", "cur_rms"]].to_numpy(float), 1e-9, None))

    def fit(self, X, y):
        n = X.loc[np.asarray(y) == 0]
        z = self._z(n)
        self.mu_, self.sd_ = z.mean(0), z.std(0) + 1e-9
        zs = (z - self.mu_) / self.sd_
        km = KMeans(self.k, n_init=10, random_state=self.seed).fit(zs)
        order = np.argsort(km.cluster_centers_[:, 0])                  # mode 0 = lowest vibration level
        self.centers_ = km.cluster_centers_[order]
        dist = np.linalg.norm(zs[:, None, :] - self.centers_[None], axis=2)
        near = dist.argmin(1)
        peak = (n.cur_rms * n.cur_crest).to_numpy()
        self.limits_ = np.array([peak[near == m].max() for m in range(self.k)])
        self.radius_ = np.array([self.support_mult * np.quantile(dist[near == m, m], self.support_q) for m in range(self.k)])
        self.limit_ = float(self.limits_.max())                           # summary scalar (highest mode's limit)
        return self

    def _assign(self, X):
        zs = (self._z(X) - self.mu_) / self.sd_
        dist = np.linalg.norm(zs[:, None, :] - self.centers_[None], axis=2)
        return dist.argmin(1), dist.min(1)

    def mode(self, X):
        return self._assign(X)[0]

    def new_mode(self, X):
        near, d = self._assign(X)
        return (d > self.radius_[near]).astype(float)

    def flag(self, X):
        near, d = self._assign(X)
        peak = (X.cur_rms * X.cur_crest).to_numpy()
        return ((peak > self.limits_[near]) & (d <= self.radius_[near])).astype(float)


def make_models(seed=0, cols=None) -> dict:
    def wrap(f):
        return (lambda: OnCols(f(), cols)) if cols else f

    return {
        "B0_SPC": lambda: SPCBaseline(),
        "B1_LogReg": wrap(lambda: LogReg()),
        "M1_RF": wrap(lambda: _Proba(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))),
        "M2_HGB": wrap(lambda: _Proba(HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            class_weight="balanced", random_state=seed))),
        "M3_IF": wrap(lambda: IsoForest(seed)),
    }


STACK_MEMBERS = ["B1_LogReg", "M1_RF", "M2_HGB"]   # SPC rule and the weak normal-only IF stay out of the stack
STACK = "M4_Stack"
