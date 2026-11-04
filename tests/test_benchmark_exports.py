"""Check CSV summaries and normalization of in-memory experiment results."""

import copy
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
                         'legacy_cancorr': {'error': 'collinear'},
                         'ridge_fixed': {'population_correlations_of_fitted_projections': [.15 if rep == 0 else .25],
                                         'train_correlations': [.8], 'penalty': [1, 1],
                                         'fit_seconds': .01}},
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
    assert len(tables['synthetic_runs.csv']) == 6
    raw = next(r for r in tables['synthetic_runs.csv'] if r['method'] == 'ridge_cancorr')
    assert raw['validation_component'] == 1
    assert math.isclose(raw['validation_correlation_mean'], .3)
    assert math.isclose(raw['validation_correlation_std'], math.sqrt(.02))


def test_penalty_grid_reports_each_candidate_and_marks_the_recorded_selection():
    tables = csv_tables(sample_report())
    grid, choices = tables['penalty_grid.csv'], tables['penalty_choices.csv']
    assert len(grid) == 8
    selected = [r for r in grid if r['selected']]
    assert len(selected) == len(choices) == 2
    assert all((r['penalty_x'], r['penalty_y']) == (.1, 1) for r in selected)
    assert all(math.isclose(r['validation_correlation_mean'], .3) for r in choices)
    assert all(math.isclose(r['validation_correlation_std'], math.sqrt(.02)) for r in choices)
    assert all(r['component'] == 1 for r in choices)


def test_penalty_comparison_uses_paired_differences_and_only_outer_test_folds():
    report = sample_report()
    entry = {'n': 4, 'p': 2, 'q': 2, 'outer_folds': []}
    for fold, (tuned, fixed) in enumerate([(.8, .7), (.6, .9)]):
        case = copy.deepcopy(report['synthetic'][0])
        case.update(fold=fold, train_indices=[0, 1] if fold == 0 else [2, 3],
                    test_indices=[2, 3] if fold == 0 else [0, 1])
        case['methods']['ridge_cancorr']['test_correlations'] = [tuned]
        case['methods']['ridge_fixed']['test_correlations'] = [fixed]
        entry['outer_folds'].append(case)
    entry['full_data'] = copy.deepcopy(entry['outer_folds'][0])
    for record in entry['full_data']['methods'].values():
        record.pop('test_correlations', None)
    report['real']['example'] = entry
    summary = csv_tables(report)['penalty_selection_summary.csv']
    synthetic = next(r for r in summary if r['dataset'] == 'synthetic')
    assert synthetic['n_pairs'] == 2
    assert math.isclose(synthetic['difference_mean'], 0, abs_tol=1e-15)
    assert math.isclose(synthetic['difference_std'], math.sqrt(.005))
    real = next(r for r in summary if r['dataset'] == 'example')
    assert real['evaluation'] == 'external_test'
    assert real['n_pairs'] == 2
    assert math.isclose(real['cv_correlation_mean'], .7)
    assert math.isclose(real['fixed_correlation_mean'], .8)
    assert math.isclose(real['difference_mean'], -.1)
    assert math.isclose(real['difference_std'], math.sqrt(.08))


def test_finish_outputs_normalizes_in_memory_arrays(monkeypatch):
    from benchmarks import export_results, run_experiments
    report = sample_report()
    record = report['synthetic'][0]['methods']['ridge_cancorr']
    record['cv_scores'] = np.asarray(record['cv_scores'])
    exported = []
    monkeypatch.setattr(export_results, 'write_outputs', lambda data, path: exported.append(data))
    run_experiments.finish_outputs(report)
    assert exported[0] == sample_report()
