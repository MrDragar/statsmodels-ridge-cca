"""Check CSV summaries and normalization of in-memory experiment results."""

import math

import numpy as np

from benchmarks.export_results import csv_tables


def sample_report():
    rows = []
    for rep, value in enumerate([.1, .3]):
        rows.append({'id': str(rep), 'n': 20, 'p': 4, 'q': 3, 'scenario': 'one_link',
                     'population_correlations': [.8], 'repetition': rep, 'seed': 10 + rep,
                     'methods': {
                         'ridge_cancorr': {'population_correlations_of_fitted_projections': [value],
                                          'train_correlations': [.9], 'penalty': [.1, 1],
                                          'cv_scores': [[-.4, -.2], [.2, .4], [-.3, -.1], [-.2, 0]],
                                          'mean_cv': .3, 'fit_seconds': .01,
                                          'selection_and_refit_seconds': .5},
                         'legacy_cancorr': {'error': 'collinear'}},
                     'correctness': {'passed': True}})
    return {'settings': {'seed': 10, 'penalty_values': [.1, 1]}, 'synthetic': rows, 'real': {}}


def test_csv_summary_uses_sample_sd_and_preserves_failed_methods():
    tables = csv_tables(sample_report())
    rows = tables['synthetic_summary.csv']
    good = next(r for r in rows if r['method'] == 'ridge_cancorr')
    bad = next(r for r in rows if r['method'] == 'legacy_cancorr')
    assert math.isclose(good['correlation_mean'], .2)
    assert math.isclose(good['correlation_std'], math.sqrt(.02))
    assert good['n_runs'] == 2
    assert bad['correlation_mean'] is None
    assert bad['n_runs'] == 0
    assert len(tables['synthetic_runs.csv']) == 4
    raw = next(r for r in tables['synthetic_runs.csv'] if r['method'] == 'ridge_cancorr')
    assert raw['validation_component'] == 1
    assert math.isclose(raw['validation_correlation_mean'], .3)
    assert math.isclose(raw['validation_correlation_std'], math.sqrt(.02))


def test_finish_outputs_normalizes_in_memory_arrays(monkeypatch):
    from benchmarks import export_results, run_experiments
    report = sample_report()
    record = report['synthetic'][0]['methods']['ridge_cancorr']
    record['cv_scores'] = np.asarray(record['cv_scores'])
    exported = []
    monkeypatch.setattr(export_results, 'write_outputs', lambda data, path: exported.append(data))
    run_experiments.finish_outputs(report)
    assert exported[0] == sample_report()
