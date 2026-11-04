"""Inspect validation grids and evaluate CV against fixed ridge penalties."""

from collections import defaultdict
from itertools import product

import numpy as np


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return (float(values.mean()) if len(values) else None,
            float(values.std(ddof=1)) if len(values) > 1 else None)


def experiment_cases(report):
    for case in report['synthetic']:
        yield {'dataset': 'synthetic', **{k: case[k] for k in
               ('n', 'p', 'q', 'scenario', 'repetition', 'seed')}}, case, 'population'
    for name, entry in report['real'].items():
        dataset = 'nutrimouse' if name == 'Nutrimouse' else 'uci_multiple_features' if name.startswith('UCI ') else name
        base = {'dataset': dataset, **{k: entry[k] for k in ('n', 'p', 'q')}}
        for case in entry['outer_folds']:
            yield {**base, 'phase': 'outer_fold', 'fold': case['fold'] + 1}, case, 'external_test'
        if 'full_data' in entry:
            yield {**base, 'phase': 'full_data', 'fold': None}, entry['full_data'], 'not_available'


def penalty_tables(report):
    grids, choices, paired = [], [], defaultdict(list)
    for context, case, evaluation in experiment_cases(report):
        field = 'population_correlations_of_fitted_projections' if evaluation == 'population' else 'test_correlations'
        for method, record in case['methods'].items():
            if 'cv_scores' not in record:
                continue
            values = report['settings']['pma_l1_bounds' if method == 'r_pma' else 'penalty_values']
            grid = list(product(values, repeat=2))
            selected = grid.index(tuple(record['penalty']))
            means = []
            for i, (penalty, scores) in enumerate(zip(grid, record['cv_scores'], strict=True)):
                valid = [s for s in scores if s is not None and np.isfinite(s)]
                mean, std = mean_std(valid) if len(valid) == len(scores) else (None, None)
                means.append(mean)
                grids.append({**context, 'method': method, 'component': 1,
                              'candidate': i + 1, 'penalty_x': penalty[0], 'penalty_y': penalty[1],
                              'selected': i == selected, 'n_valid_folds': len(valid),
                              'validation_correlation_mean': mean, 'validation_correlation_std': std,
                              **{f'validation_fold_{j + 1}': score for j, score in enumerate(scores)}})
            chosen = grids[-len(grid) + selected]
            alternatives = [v for i, v in enumerate(means) if i != selected and v is not None]
            choices.append({**context, 'method': method, 'component': 1,
                            'penalty_x': record['penalty'][0], 'penalty_y': record['penalty'][1],
                            'selected_at_boundary': any(p in (min(values), max(values)) for p in record['penalty']),
                            'validation_correlation_mean': chosen['validation_correlation_mean'],
                            'validation_correlation_std': chosen['validation_correlation_std'],
                            'validation_gap_to_next_best': means[selected] - max(alternatives) if alternatives and means[selected] is not None else None,
                            'evaluation': evaluation,
                            'evaluation_correlation': record[field][0] if field in record and evaluation != 'not_available' else None})
        tuned, fixed = (case['methods'].get(name, {}) for name in ('ridge_cancorr', 'ridge_fixed'))
        if evaluation != 'not_available' and field in tuned and field in fixed:
            key = (context['dataset'], context['n'], context['p'], context['q'], context.get('scenario'), evaluation)
            paired[key].append((tuned[field][0], fixed[field][0], tuple(fixed['penalty'])))
    summary = []
    for key, pairs in paired.items():
        if len({pair[2] for pair in pairs}) != 1:
            raise ValueError('Fixed penalties must agree within a comparison group')
        tuned, fixed = np.asarray([pair[:2] for pair in pairs]).T
        difference = tuned - fixed
        row = dict(zip(('dataset', 'n', 'p', 'q', 'scenario', 'evaluation'), key))
        row.update(component=1, n_pairs=len(pairs),
                   fixed_penalty_x=pairs[0][2][0], fixed_penalty_y=pairs[0][2][1])
        for prefix, values in [('cv_correlation', tuned), ('fixed_correlation', fixed), ('difference', difference)]:
            row[prefix + '_mean'], row[prefix + '_std'] = mean_std(values)
        row.update(n_cv_better=int(np.sum(difference > 1e-12)),
                   n_fixed_better=int(np.sum(difference < -1e-12)))
        summary.append(row)
    return {'penalty_grid.csv': grids, 'penalty_choices.csv': choices,
            'penalty_selection_summary.csv': summary}


def plot_penalties(report, tables, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = report['settings']['penalty_values']
    ticks = [f'{v:g}' for v in values]
    grid = [r for r in tables['penalty_grid.csv'] if r['method'] == 'ridge_cancorr']
    choices = [r for r in tables['penalty_choices.csv'] if r['method'] == 'ridge_cancorr']
    style = {'font.family': 'DejaVu Sans', 'font.size': 8, 'figure.facecolor': 'white'}

    def axes_labels(ax):
        ax.set_xticks(range(len(values)), ticks, rotation=45)
        ax.set_yticks(range(len(values)), ticks)
        ax.set_xlabel('alpha_y')
        ax.set_ylabel('alpha_x')

    def save(fig, name):
        fig.savefig(directory / name, dpi=180, bbox_inches='tight', metadata={'Software': 'Matplotlib'})
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(2, report['settings']['outer_cv'], figsize=(16, 6.8),
                                 squeeze=False, layout='constrained')
        for row, (dataset, label) in enumerate([('nutrimouse', 'Nutrimouse'), ('uci_multiple_features', 'UCI')]):
            selected = [r for r in grid if r['dataset'] == dataset and r.get('phase') == 'outer_fold']
            finite = [r['validation_correlation_mean'] for r in selected if r['validation_correlation_mean'] is not None]
            if not finite:
                for ax in axes[row]:
                    ax.set_visible(False)
                continue
            for col, ax in enumerate(axes[row]):
                records = [r for r in selected if r['fold'] == col + 1]
                matrix = np.full((len(values), len(values)), np.nan)
                for record in records:
                    matrix[values.index(record['penalty_x']), values.index(record['penalty_y'])] = record['validation_correlation_mean']
                im = ax.imshow(matrix, origin='lower', vmin=min(finite), vmax=max(finite), cmap='viridis', aspect='auto')
                choice = next(r for r in choices if r['dataset'] == dataset and r.get('phase') == 'outer_fold' and r['fold'] == col + 1)
                ax.scatter(values.index(choice['penalty_y']), values.index(choice['penalty_x']),
                           marker='s', s=170, facecolors='none', edgecolors='red', linewidths=2)
                digits = 6 if choice['validation_correlation_std'] < .001 else 3
                ax.set_title(f"{label}, outer fold {col + 1}\n"
                             f"Selected CV r = {choice['validation_correlation_mean']:.{digits}f} ± {choice['validation_correlation_std']:.{digits}f}")
                axes_labels(ax)
            fig.colorbar(im, ax=list(axes[row]), label='Inner-CV mean correlation', shrink=.8)
        fig.suptitle('CanCorr ridge: validation grids. Red outline: selected pair; SD across inner folds.', fontsize=11)
        save(fig, 'penalty_cv_real.png')

        dimensions = report['settings']['dimensions']
        scenarios = list(report['settings']['population_correlations'])
        fig, axes = plt.subplots(len(dimensions), len(scenarios), figsize=(11, 10),
                                 squeeze=False, layout='constrained')
        for row, dims in enumerate(dimensions):
            for col, scenario in enumerate(scenarios):
                ax = axes[row, col]
                counts = np.zeros((len(values), len(values)), dtype=int)
                for choice in choices:
                    if choice['dataset'] == 'synthetic' and choice['scenario'] == scenario and all(choice[k] == v for k, v in dims.items()):
                        counts[values.index(choice['penalty_x']), values.index(choice['penalty_y'])] += 1
                im = ax.imshow(counts, origin='lower', vmin=0, vmax=report['settings']['repetitions'], cmap='Blues')
                for i in range(len(values)):
                    for j in range(len(values)):
                        ax.text(j, i, str(counts[i, j]), ha='center', va='center',
                                color='white' if counts[i, j] > report['settings']['repetitions'] / 2 else 'black')
                ax.set_title(f"{scenario.replace('_', ' ')}; n={dims['n']}, p={dims['p']}, q={dims['q']}")
                axes_labels(ax)
        fig.colorbar(im, ax=list(axes.flat), label='Number of selections', shrink=.6,
                     ticks=range(report['settings']['repetitions'] + 1))
        fig.suptitle('CanCorr ridge: selected penalties across independent synthetic datasets', fontsize=11)
        save(fig, 'penalty_frequencies_synthetic.png')
