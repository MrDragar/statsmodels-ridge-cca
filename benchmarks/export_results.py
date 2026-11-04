"""Export the experiment results to plain CSV and Matplotlib figures."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from benchmarks.penalties import penalty_tables, plot_penalties
from benchmarks.run_experiments import METHODS, OUT
from benchmarks.tables import write_tables

NAMES = {'legacy_cancorr': 'CanCorr', 'ridge_cancorr': 'CanCorr ridge (CV)',
         'sklearn_cca': 'sklearn CCA', 'sklearn_pls': 'sklearn PLSCanonical',
         'cca_zoo': 'CCA-Zoo ridge (CV)', 'r_ridge': 'R rcc (CV)',
         'r_classical': 'R cc', 'r_pma': 'R PMA (CV)', 'ridge_fixed': 'CanCorr ridge (fixed)'}


def at(values, index):
    return values[index] if values is not None and index < len(values) else None


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def sample_sd(values):
    values = [v for v in values if v is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def maximum(values):
    values = [v for v in values if v is not None]
    return float(max(values)) if values else None


def case_rows(case, context, components, synthetic, settings):
    rows, times = [], []
    for method, record in case['methods'].items():
        error = record.get('error', '')
        shared = {**context, 'method': method, 'status': 'not_applicable' if error else 'ok',
                  'error': error}
        times.append({**shared, 'fit_seconds': record.get('fit_seconds'),
                      'selection_and_refit_seconds': record.get('selection_and_refit_seconds')})
        validation = []
        if 'cv_scores' in record:
            values = settings['pma_l1_bounds' if method == 'r_pma' else 'penalty_values']
            selected = list(product(values, repeat=2)).index(tuple(record['penalty']))
            validation = record['cv_scores'][selected]
        for j in range(components):
            row = {**shared, 'component': j + 1,
                   'train_correlation': at(record.get('train_correlations'), j),
                   'penalized_root': at(record.get('penalized_roots'), j),
                   'penalty_x': at(record.get('penalty'), 0),
                   'penalty_y': at(record.get('penalty'), 1),
                   'validation_component': 1 if len(validation) else None,
                   'validation_correlation_mean': average(validation),
                   'validation_correlation_std': sample_sd(validation)}
            if synthetic:
                row.update(true_correlation=at(case['population_correlations'], j) or 0.,
                           population_correlation=at(record.get('population_correlations_of_fitted_projections'), j),
                           max_angle_x_deg=maximum(record.get('x_subspace_angles_degrees', [])),
                           max_angle_y_deg=maximum(record.get('y_subspace_angles_degrees', [])))
            else:
                row['test_correlation'] = at(record.get('test_correlations'), j)
            rows.append(row)
    return rows, times


def summarize(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in groups.items():
        values = [r[value] for r in group if r[value] is not None]
        record = dict(zip(keys, key))
        record.update(n_runs=len(values), n_total=len(group),
                      correlation_mean=average(values), correlation_std=sample_sd(values),
                      status='ok' if len(values) == len(group) else 'partial' if values else 'not_applicable')
        if 'true_correlation' in group[0]:
            record['true_correlation'] = group[0]['true_correlation']
            for view in ('x', 'y'):
                angles = [r['max_angle_' + view + '_deg'] for r in group]
                record['max_angle_' + view + '_mean_deg'] = average(angles)
                record['max_angle_' + view + '_std_deg'] = sample_sd(angles)
        result.append(record)
    return result


def csv_tables(report):
    synthetic, real, times, checks = [], [], [], defaultdict(list)
    for case in report['synthetic']:
        context = {k: case[k] for k in ('n', 'p', 'q', 'scenario', 'repetition', 'seed')}
        context = {'dataset': 'synthetic', **context}
        rows, elapsed = case_rows(case, context, report['settings'].get('synthetic_components', 1),
                                 True, report['settings'])
        synthetic.extend(rows)
        times.extend(elapsed)
        checks[('synthetic', case['n'], case['p'], case['q'])].append(case['correctness'])
    for name, entry in report['real'].items():
        dataset = 'nutrimouse' if name == 'Nutrimouse' else 'uci_multiple_features' if name.startswith('UCI ') else name
        context = {'dataset': dataset, **{k: entry[k] for k in ('n', 'p', 'q')}}
        cases = [('outer_fold', row['fold'] + 1, row) for row in entry['outer_folds']]
        if 'full_data' in entry:
            cases.append(('full_data', None, entry['full_data']))
        for phase, fold, case in cases:
            current = {**context, 'phase': phase, 'fold': fold,
                       'n_train': len(case['train_indices']) if phase == 'outer_fold' else entry['n'],
                       'n_test': len(case['test_indices']) if phase == 'outer_fold' else None}
            rows, elapsed = case_rows(case, current, report['settings'].get('real_components', 3),
                                     False, report['settings'])
            real.extend(rows)
            times.extend(elapsed)
            checks[(dataset, entry['n'], entry['p'], entry['q'])].append(case['correctness'])
    syn_summary = summarize(synthetic, ['n', 'p', 'q', 'scenario', 'method', 'component'], 'population_correlation')
    real_summary = summarize([r for r in real if r['phase'] == 'outer_fold'],
                             ['dataset', 'n', 'p', 'q', 'method', 'component'], 'test_correlation')
    for row in real_summary:
        row['n_folds'] = row.pop('n_runs')
        row['n_total_folds'] = row.pop('n_total')
        full = next((r for r in real if r['dataset'] == row['dataset'] and r['method'] == row['method']
                     and r['phase'] == 'full_data' and r['component'] == row['component']), {})
        row.update(full_data_penalty_x=full.get('penalty_x'), full_data_penalty_y=full.get('penalty_y'))
    correctness = []
    for (dataset, n, p, q), group in checks.items():
        row = {'dataset': dataset, 'n': n, 'p': p, 'q': q,
               'n_checks': len(group), 'n_passed': sum(c['passed'] for c in group)}
        for field in ('roots_max_relative_error', 'pearson_max_relative_error'):
            row[field] = maximum([c.get(field) for c in group])
        for view in ('x', 'y'):
            row[view + '_leading_max_angle_deg'] = maximum([
                maximum(c.get(view + '_leading_principal_angles_degrees', [])) for c in group])
        row['n_legacy_checks'] = sum('zero_penalty_exact_legacy_outputs' in c for c in group)
        row['n_legacy_exact'] = sum(c.get('zero_penalty_exact_legacy_outputs', False) for c in group)
        correctness.append(row)
    return {'synthetic_summary.csv': syn_summary, 'synthetic_runs.csv': synthetic,
            'real_summary.csv': real_summary, 'real_runs.csv': real,
            'correctness.csv': correctness, 'runtime.csv': times, **penalty_tables(report)}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plots(tables, settings, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ['#1f77b4', '#d95f02', '#2ca02c']
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
             'axes.spines.right': False, 'figure.facecolor': 'white'}

    def points(ax, rows, components):
        for i, method in enumerate(METHODS):
            for j in range(components):
                row = next((r for r in rows if r['method'] == method and r['component'] == j + 1), None)
                if row is None or row['correlation_mean'] is None:
                    if j == 0:
                        ax.text(.02, i, 'NA', va='center', color='0.5', transform=ax.get_yaxis_transform())
                    continue
                offset = (j - (components - 1) / 2) * .17
                ax.errorbar(row['correlation_mean'], i + offset, xerr=row['correlation_std'],
                            fmt='o', markersize=3.5, capsize=2, linewidth=1, color=colors[j])
        ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
        ax.set_ylim(len(METHODS) - .5, -.5)
        ax.grid(axis='x', color='0.9', linewidth=.6)
        ax.set_axisbelow(True)

    def save(fig, name):
        fig.savefig(directory / (name + '.png'), dpi=180, bbox_inches='tight',
                    metadata={'Software': 'Matplotlib'})
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout='constrained')
        for col, dims in enumerate(settings['dimensions']):
            for row, scenario in enumerate(('one_link', 'two_links')):
                ax = axes[row, col]
                selected = [r for r in tables['synthetic_summary.csv'] if r['scenario'] == scenario
                            and all(r[k] == v for k, v in dims.items())]
                components = 1 if scenario == 'one_link' else 2
                points(ax, selected, components)
                for j, rho in enumerate(settings['population_correlations'][scenario]):
                    ax.axvline(rho, color=colors[j], linestyle='--', linewidth=.8)
                ax.set_xlim(-.05, .9)
                ax.set_xlabel('Population correlation of fitted projections')
                ax.set_title(f"n={dims['n']}, p={dims['p']}, q={dims['q']}\n"
                             + ('One link' if row == 0 else 'Two links'))
        fig.suptitle(f"Synthetic data: mean ± sample SD over {settings['repetitions']} datasets. Dashed lines: planted correlations", fontsize=11)
        fig.legend([Line2D([], [], color=c, marker='o', linestyle='') for c in colors[:2]],
                   ['Component 1', 'Component 2'], loc='outside lower center', ncol=2, frameon=False)
        save(fig, 'synthetic_correlations')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained')
        valid = [r for r in tables['real_summary.csv'] if r['correlation_mean'] is not None]
        lower = min(r['correlation_mean'] - (r['correlation_std'] or 0) for r in valid)
        upper = max(r['correlation_mean'] + (r['correlation_std'] or 0) for r in valid)
        for ax, dataset, title in zip(axes, ('nutrimouse', 'uci_multiple_features'),
                                       ('Nutrimouse', 'UCI Multiple Features')):
            selected = [r for r in tables['real_summary.csv'] if r['dataset'] == dataset]
            points(ax, selected, 3)
            ax.set_xlim(min(-.05, lower - .04), max(1.02, upper + .04))
            ax.set_title(title)
            ax.set_xlabel('Outer-fold correlation')
        fig.suptitle(f"Real data: mean ± sample SD over {settings['outer_cv']} outer folds", fontsize=11)
        fig.legend([Line2D([], [], color=c, marker='o', linestyle='') for c in colors],
                   ['Component 1', 'Component 2', 'Component 3'], loc='outside lower center', ncol=3, frameon=False)
        save(fig, 'real_correlations')


def write_outputs(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = csv_tables(report)
    for name, rows in tables.items():
        write_csv(directory / name, rows)
    write_tables(report, directory / 'benchmark_tables.md')
    plots(tables, report['settings'], directory)
    plot_penalties(report, tables, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUT / 'proposal_results.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    write_outputs(report, args.output_dir or args.input.parent)


if __name__ == '__main__':
    main()
