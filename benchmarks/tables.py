"""Render observed benchmark results; no model fitting or statistical inference."""

from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.run_experiments import LABELS, METHODS


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    mean = a.mean(0)
    sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(mean)
    if mean.ndim == 0:
        return f'{mean:.3f} ± {sd:.3f}'
    return '; '.join(f'{m:.3f} ± {s:.3f}' for m, s in zip(mean, sd))


def penalty_counts(rows):
    pairs = Counter(tuple(row['penalty']) for row in rows if row.get('penalty'))
    return '; '.join('(' + ', '.join(f'{v:g}' for v in pair) + f') ×{count}'
                     for pair, count in pairs.most_common()) or '—'


def write_tables(report, path):
    lines = [
        '# Бейзлайны и изменённый CanCorr', '',
        'Синтетика: обучение на полном n из proposal. В каждой ячейке — среднее ± SD',
        'по пяти независимым наборам. Качество — точная популяционная корреляция',
        'найденных проекций; test для её вычисления не нужен. Значения через «;» —',
        'первая и вторая компоненты. Все подборы используют пять внутренних folds,',
        'train-only стандартизацию и корреляцию первой компоненты как критерий.', '',
        'Компоненты сохраняют порядок своего метода. Их не переставляют и не',
        'меняют знаки по внешним результатам; Pearson correlations могут не убывать.', '',
        'Время — секунды, один поток BLAS; показаны фактические реализации.',
        'В синтетических таблицах время — mean ± SD по всем 15 запускам данного размера.',
        'В R для сетки ridge повторно используются SVD обучающих folds; итоговая',
        'модель обучается настоящим CCA::rcc. PMA решает другую, sparse CCA задачу;',
        'его L1 bounds подбираются отдельно. CCA-Zoo 4.0.1 называет rCCA `RidgeCCA`.',
        'Для ridge пары означают (αx, αy), больше — сильнее регуляризация.',
        'Для PMA пары означают L1 bounds: больше — слабее ограничение sparsity.',
        'Fit R исключает подготовку means/scales, Fit Python её включает;',
        'полное время подбора включает preprocessing в обоих случаях. Время R ridge',
        'дополнительно включает проверку cached scorer одним package fit на fold.',
        'Permutation tests и p-values отсутствуют.', '',
    ]
    for dims in report['settings']['dimensions']:
        rows = [row for row in report['synthetic'] if all(row[k] == v for k, v in dims.items())]
        lines += [f"## Синтетика: n={dims['n']}, p={dims['p']}, q={dims['q']}", '',
                  '| Метод | ρ=(0,0) | ρ=(.8,0) | ρ=(.8,.5) | Fit, с | Подбор + refit, с |',
                  '|---|---|---|---|---:|---:|']
        for method in METHODS:
            cells = []
            for scenario in report['settings']['population_correlations']:
                group = [row['methods'][method] for row in rows if row['scenario'] == scenario]
                values = [row['population_correlations_of_fitted_projections'] for row in group if 'error' not in row]
                cells.append(mean_sd(values) if values else 'Неприменим / ошибка')
                if values and len(values) < len(group):
                    cells[-1] += f' ({len(values)}/{len(group)})'
            successful = [row['methods'][method] for row in rows if 'error' not in row['methods'][method]]
            times = [row['fit_seconds'] for row in successful]
            total = [row['selection_and_refit_seconds'] for row in successful if 'selection_and_refit_seconds' in row]
            lines.append('| ' + ' | '.join([LABELS[method], *cells,
                                             mean_sd(times) if times else '—',
                                             mean_sd(total) if total else '—']) + ' |')
        lines += ['', 'Углы к истинному подпространству (максимальный угол, градусы; меньше лучше):', '',
                  '| Метод | Одна связь: X / Y | Две связи: X / Y |', '|---|---|---|']
        for method in METHODS:
            cells = []
            for scenario in ('one_link', 'two_links'):
                group = [row['methods'][method] for row in rows if row['scenario'] == scenario
                         and 'error' not in row['methods'][method]]
                if not group:
                    cells.append('—')
                    continue
                cells.append(' / '.join(mean_sd([max(row[view + '_subspace_angles_degrees'])
                                                  for row in group]) for view in ('x', 'y')))
            lines.append('| ' + ' | '.join([LABELS[method], *cells]) + ' |')
        lines += ['', 'Выбранные пары параметров: пара × число повторений.', '',
                  '| Метод | Нет связей | Одна связь | Две связи |', '|---|---|---|---|']
        for method in ('ridge_cancorr', 'cca_zoo', 'r_ridge', 'r_pma'):
            cells = [penalty_counts([row['methods'][method] for row in rows if row['scenario'] == scenario])
                     for scenario in report['settings']['population_correlations']]
            lines.append('| ' + ' | '.join([LABELS[method], *cells]) + ' |')
        lines.append('')
    for name, entry in report['real'].items():
        lines += [f"## {name}: n={entry['n']}, p={entry['p']}, q={entry['q']}", '',
                  'Внешняя 5-fold оценка: среднее ± SD корреляций по пяти внешним folds.',
                  'Штрафы для каждой внешней модели подбираются заново только на её train.',
                  'SD между folds не является доверительным интервалом.',
                  'Пара штрафов и время справа относятся к отдельному итоговому fit на всех данных.', '',
                  '| Метод | Внешние корреляции: 1; 2; 3 | Итоговые штрафы | Fit, с | Подбор + refit, с |',
                  '|---|---|---|---:|---:|']
        for method in METHODS:
            rows = [row['methods'][method] for row in entry['outer_folds']]
            values = [row['test_correlations'] for row in rows if 'error' not in row]
            full = entry['full_data']['methods'][method]
            penalty = full.get('penalty')
            pair = '(' + ', '.join(f'{v:g}' for v in penalty) + ')' if penalty else '—'
            lines.append('| ' + ' | '.join([
                LABELS[method], mean_sd(values) if values else 'Неприменим / ошибка', pair,
                f"{full['fit_seconds']:.4f}" if 'fit_seconds' in full else '—',
                f"{full['selection_and_refit_seconds']:.3f}" if 'selection_and_refit_seconds' in full else '—']) + ' |')
        lines += ['', 'Параметры, выбранные внутри каждого внешнего train:', '',
                  '| Метод | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |', '|---|---|---|---|---|---|']
        for method in ('ridge_cancorr', 'cca_zoo', 'r_ridge', 'r_pma'):
            cells = [penalty_counts([row['methods'][method]]).removesuffix(' ×1')
                     for row in entry['outer_folds']]
            lines.append('| ' + ' | '.join([LABELS[method], *cells]) + ' |')
        lines.append('')
    lines += ['## Correctness из proposal', '',
              'Ridge сравнивается с CCA::rcc на тех же данных и тех же выбранных штрафах',
              'по всем наблюдаемым компонентам: min(rank(X),rank(Y)). Дополнительные',
              'нулевые корни R проверяются отдельно (absolute value <10⁻⁸). Дополнительно для ridge',
              'проверены вложенные ведущие подпространства размерностей 1, 2 и 3;',
              'в столбце углов показан наибольший угол именно этих подпространств.',
              'Для classical R cc',
              'сравниваются первые две/три компоненты. Порог relative error — 10⁻⁶;',
              'дополнительный численный порог для согласования подпространств:',
              'max sin(principal angle) < 10⁻⁶. Это точность относительно R,',
              'она не означает восстановление planted truth.', '',
              '| Данные | Проверок | Max relative error roots | Max relative error correlations | Max угол X / Y, ° | Пройдено |',
              '|---|---:|---:|---:|---|---:|']
    groups = []
    for dims in report['settings']['dimensions']:
        checks = [row['correctness'] for row in report['synthetic'] if all(row[k] == v for k, v in dims.items())]
        groups.append((f"{dims['n']}, {dims['p']}, {dims['q']}", checks))
    for name, entry in report['real'].items():
        groups.append((name, [row['correctness'] for row in entry['outer_folds']] + [entry['full_data']['correctness']]))
    all_checks = []
    for name, checks in groups:
        all_checks.extend(checks)
        good = [c for c in checks if 'reference_error' not in c]
        lines.append('| ' + ' | '.join([
            name, str(len(checks)),
            f"{max(c['roots_max_relative_error'] for c in good):.2e}" if good else '—',
            f"{max(c['pearson_max_relative_error'] for c in good):.2e}" if good else '—',
            ' / '.join(f"{max(max(c.get(v + '_leading_principal_angles_degrees', [c[v + '_max_principal_angle_degrees']])) for c in good):.2e}" for v in ('x', 'y')) if good else '—',
            f"{sum(c['passed'] for c in checks)}/{len(checks)}"]) + ' |')
    legacy = [c['zero_penalty_exact_legacy_outputs'] for c in all_checks if 'zero_penalty_exact_legacy_outputs' in c]
    cc = [c['classical_r'] for c in all_checks if 'classical_r' in c]
    same = [c['independent_r_selected_same_penalty'] for c in all_checks if 'independent_r_selected_same_penalty' in c]
    lines += ['', f'Нулевые штрафы: точное совпадение с исходным CanCorr — {sum(legacy)}/{len(legacy)}.',
              f'Classical reference R cc: {sum(c["passed"] for c in cc)}/{len(cc)} проверок пройдено.',
              f'Независимый R-подбор выбрал ту же пару, что CanCorr: {sum(same)}/{len(same)}.',
              'Совпадение пары не является условием Correctness: при близких CV scores возможны различия.', '',
              'Параметры каждой внешней модели, все CV scores, предупреждения, ошибки и',
              'principal angles сохранены в [proposal_results.json](proposal_results.json).',
              'UCI остаётся кандидатом второго набора; использованы все уникальные пары,',
              'без искусственного сокращения train до 100 объектов. Writer/group IDs',
              'отсутствуют, поэтому ограничение независимости остаётся.', '']
    errors = {}
    records = [row['methods'] for row in report['synthetic']]
    for entry in report['real'].values():
        records.extend(row['methods'] for row in entry['outer_folds'])
        records.append(entry['full_data']['methods'])
    for record in records:
        for method, row in record.items():
            if 'error' in row:
                errors[(method, row['error'])] = errors.get((method, row['error']), 0) + 1
    if errors:
        lines += ['Ошибки и неприменимые методы:', '']
        lines += [f'- {LABELS[m]}: {count} случаев — {error}' for (m, error), count in errors.items()]
    warnings = {}
    for record in records:
        for method, row in record.items():
            for message in sorted(set(row.get('warnings', []))):
                warnings[(method, message)] = warnings.get((method, message), 0) + 1
    if warnings:
        lines += ['', 'Предупреждения фактических запусков:', '']
        lines += [f'- {LABELS[m]}: {count} случаев — {message}' for (m, message), count in warnings.items()]
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    import json
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'reports/implementation/proposal_results.json').read_text())
    write_tables(report, root / 'reports/implementation/benchmark_tables.md')
