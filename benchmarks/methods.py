"""Adapters used only for comparisons; the estimator lives in statsmodels."""

import subprocess
import types
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from cca_zoo.linear import RidgeCCA
from sklearn.cross_decomposition import CCA, PLSCanonical
from statsmodels.multivariate.cancorr import CanCorr, select_cancorr_penalty

ROOT = Path(__file__).resolve().parents[1]
BASE = '295b96e65000793aeaf9a95fd7e97e73bace6557'


@dataclass
class Projection:
    x_mean: np.ndarray
    y_mean: np.ndarray
    x_coef: np.ndarray
    y_coef: np.ndarray
    roots: np.ndarray | None = None
    notes: list = field(default_factory=list)

    def transform(self, x, y):
        return (x - self.x_mean) @ self.x_coef, (y - self.y_mean) @ self.y_coef


def paired_corr(u, v):
    u, v = u - u.mean(0), v - v.mean(0)
    denom = np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0)
    if np.any(denom == 0):
        raise ValueError('Constant projected scores')
    return np.clip(np.sum(u * v, axis=0) / denom, -1, 1)


def folds(n, cv, seed):
    parts = np.array_split(np.random.default_rng(seed).permutation(n), cv)
    return [(np.concatenate(parts[:i] + parts[i + 1:]), part)
            for i, part in enumerate(parts)]


def preprocessing(x, y):
    mx, my = x.mean(0), y.mean(0)
    sx, sy = x.std(0, ddof=1), y.std(0, ddof=1)
    sx[sx == 0], sy[sy == 0] = 1, 1
    return (x - mx) / sx, (y - my) / sy, mx, my, sx, sy


@lru_cache(maxsize=1)
def legacy_cancorr():
    code = subprocess.check_output([
        'git', '-C', str(ROOT / 'statsmodels-checkout'), 'show',
        f'{BASE}:statsmodels/multivariate/cancorr.py'], text=True)
    module = types.ModuleType('statsmodels.multivariate._benchmark_legacy_cancorr')
    module.__package__ = 'statsmodels.multivariate'
    # Execute only the trusted module from the pinned local upstream commit.
    exec(compile(code, '<pristine statsmodels CanCorr>', 'exec'), module.__dict__)  # noqa: S102
    return module.CanCorr


def from_cancorr(model, k):
    return Projection(model.x_mean, model.y_mean, model.x_cancoef[:, :k],
                      model.y_cancoef[:, :k], model.penalized_roots[:k])


def fit_python(method, x, y, k, penalty=None):
    if method == 'ridge_cancorr':
        return from_cancorr(CanCorr(y, x, penalty=penalty, standardize=True), k)
    xs, ys, mx, my, sx, sy = preprocessing(x, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if method == 'legacy_cancorr':
            fit = legacy_cancorr()(ys, xs)
            ax, ay, roots = fit.x_cancoef[:, :k], fit.y_cancoef[:, :k], fit.cancorr[:k]
        elif method in ('sklearn_cca', 'sklearn_pls'):
            cls = CCA if method == 'sklearn_cca' else PLSCanonical
            fit = cls(n_components=k, scale=False, max_iter=5000, tol=1e-8).fit(xs, ys)
            ax, ay, roots = fit.x_rotations_, fit.y_rotations_, None
        elif method == 'cca_zoo':
            # v4 RidgeCCA uses (1-c)*S + c*I, so c=lambda/(1+lambda).
            lam = np.asarray(penalty) * [np.mean(xs.var(0, ddof=1)),
                                         np.mean(ys.var(0, ddof=1))]
            fit = RidgeCCA(n_components=k, center=False,
                           shrinkage=(lam / (1 + lam)).tolist()).fit([xs, ys])
            ax, ay = fit.weights_
            roots = None
        else:
            raise ValueError(f'Unknown method: {method}')
    return Projection(mx, my, ax / sx[:, None], ay / sy[:, None], roots,
                      sorted({str(w.message) for w in caught}))


def best_candidate(means, penalties, stronger='larger'):
    eligible = np.flatnonzero(np.isfinite(means))
    if not len(eligible):
        raise ValueError('Every penalty candidate failed in at least one fold')
    best = eligible[0]
    for i in eligible[1:]:
        tied = np.isclose(means[i], means[best], rtol=1e-12, atol=1e-12)
        strength = sum(penalties[i]) - sum(penalties[best])
        if (means[i] > means[best] and not tied) or (tied and (
                strength > 0 if stronger == 'larger' else strength < 0)):
            best = i
    return int(best)


def tune_python(method, x, y, k, grid, cv, seed):
    start = perf_counter()
    if method == 'ridge_cancorr':
        result = select_cancorr_penalty(y, x, grid, cv=cv,
                                       standardize=True, rng=seed)
        penalty, scores = result.penalty, result.cv_scores
        model = from_cancorr(result.model, k)
        failures = result.failures
        warnings_seen = []
    else:
        scores = np.full((len(grid), cv), np.nan)
        failures, warnings_seen = [], set()
        for i, penalty in enumerate(grid):
            for j, (train, valid) in enumerate(folds(len(x), cv, seed)):
                try:
                    fit = fit_python(method, x[train], y[train], 1, penalty)
                    scores[i, j] = paired_corr(*fit.transform(x[valid], y[valid]))[0]
                    warnings_seen.update(fit.notes)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    failures.append({'candidate': i, 'fold': j, 'reason': str(exc)})
        best = best_candidate(scores.mean(1), grid)
        penalty = grid[best]
        model = fit_python(method, x, y, k, penalty)
        warnings_seen = sorted(warnings_seen)
    total = perf_counter() - start
    # A separately timed final fit makes estimation cost visible beside tuning.
    start = perf_counter()
    fit_python(method, x, y, k, penalty)
    fit_seconds = perf_counter() - start
    return {'model': model, 'penalty': tuple(penalty), 'cv_scores': scores,
                'mean_cv': float(scores.mean(1)[grid.index(tuple(penalty))]),
                'selection_and_refit_seconds': total, 'fit_seconds': fit_seconds,
                'failures': failures, 'warnings': warnings_seen}
