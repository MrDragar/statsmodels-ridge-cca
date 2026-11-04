"""Run tuned baselines and proposal correctness checks without permutation tests.

MPLCONFIGDIR=/tmp/hdda-matplotlib OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python -m benchmarks.run_experiments
"""

import argparse
import hashlib
import inspect
import itertools
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.linalg import subspace_angles
from statsmodels.multivariate.cancorr import CanCorr
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.methods import (
    BASE,
    Projection,
    fit_python,
    folds,
    from_cancorr,
    legacy_cancorr,
    paired_corr,
    preprocessing,
    tune_python,
)
from benchmarks.synthetic import make_cca_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/implementation'
METHODS = ['legacy_cancorr', 'ridge_cancorr', 'sklearn_cca', 'sklearn_pls',
           'cca_zoo', 'r_ridge', 'r_classical', 'r_pma']
LABELS = {'legacy_cancorr': 'CanCorr исходный, α=0',
          'ridge_cancorr': 'CanCorr ridge + K-fold', 'sklearn_cca': 'sklearn CCA',
          'sklearn_pls': 'sklearn PLSCanonical', 'cca_zoo': 'CCA-Zoo RidgeCCA + K-fold',
          'r_ridge': 'R CCA::rcc + K-fold', 'r_classical': 'R CCA::cc',
          'r_pma': 'R PMA::CCA + K-fold'}


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


class RWorker:
    def __init__(self):
        self.stderr = (OUT / 'r_baselines.log').open('a')
        self.proc = subprocess.Popen(['Rscript', 'benchmarks/r_baselines.R'], cwd=ROOT,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, text=True, bufsize=1,
                                     env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
        self.environment = json.loads(self.proc.stdout.readline())
        if not self.environment.get('ready'):
            raise RuntimeError('R worker did not initialize')

    def run(self, request):
        self.proc.stdin.write(json.dumps(clean(request)) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('R worker exited; inspect reports/implementation/r_baselines.log')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.stderr.close()
        if self.proc.returncode:
            raise RuntimeError(f'R worker exited with {self.proc.returncode}')


def r_projection(record):
    return Projection(np.atleast_1d(record['x_mean']), np.atleast_1d(record['y_mean']),
                      np.asarray(record['x_coef']), np.asarray(record['y_coef']),
                      None if record.get('roots') is None else np.atleast_1d(record['roots']))


def relative_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-12)))


def correctness(ours, reference, x, y):
    if 'error' in reference:
        return {'passed': False, 'reference_error': reference['error']}
    ref = r_projection(reference)
    count = len(ours.roots)
    if len(ref.roots) < count:
        return {'passed': False, 'reference_error': 'R returned fewer observable components'}
    trailing = ref.roots[count:]
    # R includes covariance-null directions; CanCorr retains observed ranks.
    # Compare the retained components and check that omitted R roots are zero.
    trailing_max = float(np.max(np.abs(trailing))) if len(trailing) else 0.0
    ref = Projection(ref.x_mean, ref.y_mean, ref.x_coef[:, :count],
                     ref.y_coef[:, :count], ref.roots[:count])
    errors = {
        'components_compared': len(ours.roots),
        'reference_additional_zero_roots': len(trailing),
        'reference_trailing_roots_max_absolute': trailing_max,
        'roots_max_relative_error': relative_error(ours.roots, ref.roots),
        'pearson_max_relative_error': relative_error(paired_corr(*ours.transform(x, y)),
                                                    paired_corr(*ref.transform(x, y))),
    }
    for view in ('x', 'y'):
        angles = subspace_angles(getattr(ours, view + '_coef'), getattr(ref, view + '_coef'))
        errors[view + '_max_principal_angle_degrees'] = float(np.degrees(angles).max())
        errors[view + '_max_sin_angle'] = float(np.sin(angles).max())
        # Full column spaces alone cannot check leading canonical directions.
        # Check nested leading subspaces to detect incorrect ordering/rotation.
        leading = [subspace_angles(getattr(ours, view + '_coef')[:, :j],
                                   getattr(ref, view + '_coef')[:, :j])
                   for j in range(1, min(3, len(ours.roots)) + 1)]
        errors[view + '_leading_principal_angles_degrees'] = [
            float(np.degrees(a).max()) for a in leading]
        errors[view + '_leading_max_sin_angle'] = float(max(np.sin(a).max() for a in leading))
    errors['passed'] = trailing_max < 1e-8 and all(errors[name] < 1e-6 for name in (
        'roots_max_relative_error', 'pearson_max_relative_error',
        'x_max_sin_angle', 'y_max_sin_angle',
        'x_leading_max_sin_angle', 'y_leading_max_sin_angle'))
    return errors


def evaluate(model, x, y, data=None, test=None):
    result = {'train_correlations': paired_corr(*model.transform(x, y)).tolist()}
    if model.roots is not None:
        result['penalized_roots'] = model.roots.tolist()
    if test is not None:
        result['test_correlations'] = paired_corr(*model.transform(*test)).tolist()
    if data is not None:
        ax, ay = model.x_coef, model.y_coef
        population = np.diag(ax.T @ data.cov_xy @ ay) / (
            np.linalg.norm(ax, axis=0) * np.linalg.norm(ay, axis=0))
        result['population_correlations_of_fitted_projections'] = population.tolist()
        truth = np.zeros(len(population))
        truth[:len(data.correlations)] = data.correlations
        result['population_correlation_rmse'] = float(np.sqrt(np.mean((population - truth) ** 2)))
        if len(data.correlations):
            rank = len(data.correlations)
            for view, weights in [('x', data.x_weights), ('y', data.y_weights)]:
                result[view + '_subspace_angles_degrees'] = np.degrees(subspace_angles(
                    getattr(model, view + '_coef')[:, :rank], weights)).tolist()
    if model.notes:
        result['warnings'] = model.notes
    return result


def run_case(worker, x, y, settings, seed, k, data=None, test=None):
    grid = list(itertools.product(settings['penalty_values'], repeat=2))
    pma_grid = list(itertools.product(settings['pma_l1_bounds'], repeat=2))
    records, models = {}, {}
    for method in METHODS[:5]:
        print(f'    {method}', flush=True)
        try:
            start = perf_counter()
            if method in ('ridge_cancorr', 'cca_zoo'):
                fitted = tune_python(method, x, y, k, grid, settings['cv'], seed)
                model = fitted.pop('model')
                records[method] = fitted
            else:
                model = fit_python(method, x, y, k)
                records[method] = {'fit_seconds': perf_counter() - start}
            records[method].update(evaluate(model, x, y, data, test))
            models[method] = model
        except (ValueError, np.linalg.LinAlgError) as exc:
            records[method] = {'error': str(exc)}
    if 'ridge_cancorr' not in models:
        raise RuntimeError('The implemented ridge estimator failed')
    tmp = ROOT / 'research/cache/benchmark_inputs'
    tmp.mkdir(parents=True, exist_ok=True)
    for name, array in [('x', x), ('y', y)]:
        np.savetxt(tmp / f'{name}.csv', array, delimiter=',', fmt='%.17g')
    fold_ids = np.empty(len(x), dtype=int)
    for j, (_, valid) in enumerate(folds(len(x), settings['cv'], seed)):
        fold_ids[valid] = j + 1
    print('    R rcc/cc/PMA, independent tuning', flush=True)
    response = worker.run({'x_path': str(tmp / 'x.csv'), 'y_path': str(tmp / 'y.csv'),
                               'fold_ids': fold_ids, 'ridge_grid': grid, 'pma_grid': pma_grid,
                               'our_penalty': records['ridge_cancorr']['penalty'], 'k': k, 'seed': seed})
    for method in ('r_ridge', 'r_classical', 'r_pma'):
        record = response[method]
        if 'error' in record:
            records[method] = record
            continue
        model = r_projection(record)
        records[method] = {key: value for key, value in record.items() if key not in
                           ('x_mean', 'y_mean', 'x_coef', 'y_coef', 'roots', 'train_correlations')}
        records[method].update(evaluate(model, x, y, data, test))
        models[method] = model
    full = CanCorr(y, x, penalty=records['ridge_cancorr']['penalty'], standardize=True)
    checks = correctness(from_cancorr(full, len(full.cancorr)),
                         response['correctness_reference'], x, y)
    if 'r_ridge' in models:
        ours = records['ridge_cancorr']
        checks['independent_r_selected_same_penalty'] = bool(np.array_equal(
            ours['penalty'], records['r_ridge']['penalty']))
        checks['independent_r_cv_max_absolute_error'] = float(np.nanmax(abs(
            np.asarray(ours['cv_scores']) - np.asarray(records['r_ridge']['cv_scores'], dtype=float))))
    if 'legacy_cancorr' in models:
        xs, ys, *_ = preprocessing(x, y)
        original, new = legacy_cancorr()(ys, xs), CanCorr(ys, xs, penalty=(0, 0))
        checks['zero_penalty_exact_legacy_outputs'] = all(np.array_equal(a, b) for a, b in (
            (original.cancorr, new.cancorr), (original.x_cancoef, new.x_cancoef),
            (original.y_cancoef, new.y_cancoef)))
        if 'r_classical' in models:
            # R reference coefficients are on raw features; compare score roots
            # and subspaces in those same raw coordinates via the adapter.
            original_raw = models['legacy_cancorr']
            original_raw.roots = new.penalized_roots[:k]
            checks['classical_r'] = correctness(original_raw, response['r_classical'], x, y)
    return {'methods': records, 'correctness': checks}


def load_real():
    path = ROOT / 'data/raw/nutrimouse'
    nutri = (np.loadtxt(path / 'gene.csv', delimiter=','),
             np.loadtxt(path / 'lipid.csv', delimiter=','))
    import io
    archive = ROOT / 'data/raw/multiple_features.zip'
    manifest = json.loads((ROOT / 'data/sources.json').read_text())
    expected = next(row['sha256'] for row in manifest['artifacts']
                    if row['path'] == 'data/raw/multiple_features.zip')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('UCI source archive checksum differs')
    with ZipFile(archive) as z:
        x = np.loadtxt(io.BytesIO(z.read('mfeat-fac')))
        y = np.loadtxt(io.BytesIO(z.read('mfeat-pix')))
    _, unique = np.unique(np.column_stack([x, y]), axis=0, return_index=True)
    unique.sort()
    return {'Nutrimouse': nutri, 'UCI Multiple Features (кандидат)': (x[unique], y[unique])}


def checkpoint(report):
    save(OUT / 'proposal_results.json', report)


def recheck_correctness(report, worker):
    """Strengthen saved checks without rerunning any selected grid or baseline."""
    tmp = ROOT / 'research/cache/benchmark_inputs'

    def update(case, x, y):
        penalty = case['methods']['ridge_cancorr']['penalty']
        for name, array in [('x', x), ('y', y)]:
            np.savetxt(tmp / f'{name}.csv', array, delimiter=',', fmt='%.17g')
        ref = worker.run({'x_path': str(tmp / 'x.csv'), 'y_path': str(tmp / 'y.csv'),
                          'reference_only': True, 'our_penalty': penalty,
                          'seed': report['settings']['seed']})['correctness_reference']
        fit = CanCorr(y, x, penalty=penalty, standardize=True)
        check = correctness(from_cancorr(fit, len(fit.cancorr)), ref, x, y)
        case['correctness'].update(check)

    for row in report['synthetic']:
        data = make_cca_data(row['n'], row['p'], row['q'], row['population_correlations'], rng=row['seed'])
        update(row, data.x, data.y)
        print(f"Leading-subspace reference: {row['id']}", flush=True)
    for name, (x, y) in load_real().items():
        entry = report['real'][name]
        for row in entry['outer_folds']:
            train = np.asarray(row['train_indices'])
            update(row, x[train], y[train])
        update(entry['full_data'], x, y)
        print(f'Leading-subspace reference: {name}', flush=True)
    report['correctness_verification_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / 'benchmarks/run_experiments.py', ROOT / 'benchmarks/r_baselines.R']}
    report['protocol']['correctness'] += ' Also checked nested leading subspaces of sizes 1, 2 and 3 against actual R package fits.'
    report['protocol']['timing'] += ' R final-fit timing excludes full-data preprocessing; Python includes it. R selection totals include preprocessing.'
    checkpoint(report)


def finish_outputs(report):
    from benchmarks.export_results import write_outputs
    report = clean(report)
    write_outputs(report, OUT)


@threadpool_limits.wrap(limits=1)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='reuse completed cases with matching settings/source')
    parser.add_argument('--smoke', action='store_true', help='one small synthetic case; writes a separate smoke artifact')
    parser.add_argument('--verify-correctness-only', action='store_true',
                        help='recheck existing selected fits against R, including leading subspaces')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT / 'benchmarks/proposal.json').read_text())
    source = Path(inspect.getfile(CanCorr)).resolve()
    if not source.is_relative_to(ROOT / 'statsmodels-checkout'):
        raise RuntimeError('Install the actual statsmodels checkout in editable mode')
    if args.verify_correctness_only:
        report = json.loads((OUT / 'proposal_results.json').read_text())
        worker = RWorker()
        try:
            recheck_correctness(report, worker)
        finally:
            worker.close()
        finish_outputs(report)
        return
    sources = [source, *(ROOT / 'benchmarks' / file for file in
                        ['methods.py', 'r_baselines.R', 'synthetic.py', 'run_experiments.py',
                         'export_results.py', 'tables.py'])]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = {'scope': 'Estimation, tuned baseline comparison and correctness; no permutation tests',
              'settings': settings, 'source_sha256': hashes, 'statsmodels_base_commit': BASE,
              'python': platform.python_version(),
              'packages': {p: version(p) for p in ['numpy', 'scipy', 'statsmodels', 'scikit-learn', 'cca-zoo']},
              'numerical_libraries': [{k: item.get(k) for k in ('internal_api', 'version', 'num_threads')}
                                      for item in threadpool_info()],
              'protocol': {'synthetic': 'Full n training, exact population evaluation, 5 independent repetitions.',
                           'real': '5 outer folds, 5 inner folds; mean signed per-fold score correlations, not pooled scores. Full-data tuned fits are recorded separately.',
                           'tuning': 'First score correlation. Identical folds and additive-ridge grid across implementations. R reuses fold SVDs in exact grid scorer; final fits use actual CCA::rcc. PMA uses its own L1-bound grid and 100 iterations.',
                           'timing': 'Wall seconds, BLAS threads=1. fit_seconds excludes tuning; Python tuned fits are separately timed repeats. R timings include fold-cache preparation and package validation check. Initial namespace/import startup and data I/O excluded.',
                           'correctness': 'Ridge reference at identical penalties; relative errors <1e-6 and max sin(principal angle)<1e-6. Classical outputs must equal pristine CanCorr exactly.',
                           'choices': 'Repetitions, signal strengths, components, grids and folds are experiment settings; only dataset dimensions come from proposal.',
                           'uci': 'Candidate second dataset, full 1994 unique paired rows after removing six exact duplicate rows. Shuffled folds without digit labels. Writer/group identifiers unavailable; exchangeability not established.'},
              'synthetic': [], 'real': {}}
    if args.resume:
        prior = json.loads((OUT / 'proposal_results.json').read_text())
        if prior['settings'] != settings or prior['source_sha256'] != hashes:
            raise ValueError('Cannot resume results produced with different settings or code')
        report = prior
    worker = RWorker()
    report['r_environment'] = worker.environment
    try:
        for i, dimensions in enumerate(settings['dimensions']):
            for j, (scenario, rho) in enumerate(settings['population_correlations'].items()):
                for rep in range(settings['repetitions']):
                    key = f'{i}-{j}-{rep}'
                    if any(row['id'] == key for row in report['synthetic']):
                        continue
                    seed = settings['seed'] + 10000 * i + 100 * j + rep * 7
                    print(f'Synthetic {dimensions}, {scenario}, repetition {rep + 1}', flush=True)
                    data = make_cca_data(**dimensions, correlations=rho, rng=seed)
                    result = run_case(worker, data.x, data.y, settings, seed + 1,
                                      settings['synthetic_components'], data=data)
                    report['synthetic'].append(dict(id=key, **dimensions, scenario=scenario,
                                                     repetition=rep, seed=seed,
                                                     population_correlations=rho, **result))
                    if args.smoke:
                        save(OUT / 'smoke_results.json', report)
                        return
                    checkpoint(report)
        for name, (x, y) in load_real().items():
            entry = report['real'].setdefault(name, {'n': len(x), 'p': x.shape[1], 'q': y.shape[1],
                                                      'outer_folds': [],
                                                      'input_sha256': hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()})
            for j, (train, test) in enumerate(folds(len(x), settings['outer_cv'], settings['seed'] + 999)):
                if any(row['fold'] == j for row in entry['outer_folds']):
                    continue
                print(f'{name}, outer fold {j + 1}, train={len(train)}, test={len(test)}', flush=True)
                result = run_case(worker, x[train], y[train], settings,
                                  settings['seed'] + 2000 + j, settings['real_components'],
                                  test=(x[test], y[test]))
                entry['outer_folds'].append(dict(fold=j, train_indices=train.tolist(),
                                                 test_indices=test.tolist(), **result))
                checkpoint(report)
            if 'full_data' not in entry:
                print(f'{name}, full-data tuned fit', flush=True)
                entry['full_data'] = run_case(worker, x, y, settings,
                                               settings['seed'] + 3000, settings['real_components'])
                checkpoint(report)
    finally:
        worker.close()
    finish_outputs(report)
    print('Saved full results, CSV tables and figures.', flush=True)


if __name__ == '__main__':
    main()
