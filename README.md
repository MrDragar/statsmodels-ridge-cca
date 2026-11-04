# Ridge CCA для statsmodels

Требования: [proposal](proposal/proposal_submitted.pdf),
[комментарий преподавателя](proposal/feedback.md). Полный оставшийся объём —
в [плане](notes/project_plan_ru.md), формулы и нормировки — в
[методической записке](notes/method_note_stage1_ru.md).

Изменён **сам CanCorr** в `statsmodels-checkout`, ветка `codex/ridge-cancorr`,
исходная ревизия `295b96e65000793aeaf9a95fd7e97e73bace6557`.
`.venv` использует этот checkout через editable install.
Переносимые изменения — [statsmodels.patch](statsmodels.patch).
`research/ridge.py` служит адаптером прежних численных сверок к CanCorr.

В Git хранятся код Python/R, тесты, patch statsmodels, конфигурация экспериментов,
зависимости, proposal, рабочие заметки и [манифест источников](data/sources.json).
`statsmodels-checkout/` создаётся при установке. Результаты в `reports/`, сырые
данные в `data/raw/`, кэши и локальные окружения в Git не включаются.
Небольшие CSV с входами и эталонными ответами R входят в patch как fixtures
для тестов. Условия распространения исходного кода statsmodels сохранены в
[LICENSES/statsmodels.txt](LICENSES/statsmodels.txt).

```python
from statsmodels.multivariate.api import CanCorr, select_cancorr_penalty

fit = CanCorr(y_train, x_train, penalty=(0.1, 1), standardize=True)
selection = select_cancorr_penalty(
    y_train, x_train,
    [(a, b) for a in (0.01, 0.1, 1) for b in (0.01, 0.1, 1)],
    cv=5, standardize=True, rng=123,
)
x_scores, y_scores = selection.model.transform(x_test, y_test)
```

`penalized_roots` — значения штрафной задачи; `cancorr` — Pearson correlations
train scores. При ridge они различаются. Нулевые штрафы с обычными defaults
сохраняют прежний расчёт CanCorr. Коэффициенты применяются к центрированным
исходным признакам; стандартизация включена в них. Means/scales и масштабы
штрафов обучаются внутри каждого train fold. Корреляции validation/test
сохраняют знак и порядок компонент из train.

## Сравнительные эксперименты

Результаты создаются локально после запуска экспериментов:
`reports/implementation/benchmark_tables.md` — таблицы сравнения,
`reports/implementation/proposal_results.json` — подробные результаты и параметры.

Численная реализация прошла Correctness относительно R во всех 57 случаях;
максимальная relative error корреляций — около 6.3e-12. Это не означает
хорошего восстановления истинных связей во всех конфигурациях. При одной
истинной корреляции .8 получено .7844 ± .0023 для (n,p,q)=(500,20,20),
.0152 ± .0240 для (50,100,100) и .0065 ± .0068 для (200,500,200).
Здесь и далее mean ± sample std. R ridge даёт те же результаты.
Преимущество ridge на этих широких синтетических данных не установлено.
На реальных данных средние внешние корреляции нашего ridge: Nutrimouse —
.947 ± .042, .607 ± .242, .660 ± .229; UCI — .99885 ± .00016,
.99771 ± .00021, .99683 ± .00033.

После завершения `benchmarks.run_experiments` автоматически сохраняются обычные CSV:

- `synthetic_summary.csv` — среднее и SD по независимым синтетическим наборам;
- `synthetic_runs.csv` — значения каждого синтетического запуска и выбранные штрафы;
- `real_summary.csv` — среднее и SD по внешним folds, штрафы итоговой модели;
- `real_runs.csv` — каждый внешний fold и отдельный fit на всех данных;
- `correctness.csv` — ошибки относительно R и углы между подпространствами;
- `runtime.csv` — время выполнения отдельно от численных результатов.
- `penalty_choices.csv` — выбранная пара каждого подбора, validation mean/std,
  разрыв до следующего кандидата, выбор на границе сетки и внешняя корреляция;
- `penalty_grid.csv` — все кандидаты и их validation correlations по folds,
  mean/std и отметка выбранной пары для CanCorr, CCA-Zoo, R ridge и PMA;
- `penalty_selection_summary.csv` — первая корреляция CanCorr с подбором и
  с фиксированной парой, а также парная разность CV − fixed, всё с mean/std.

Все файлы находятся в `reports/implementation/`. В CSV среднее и SD — отдельные
числовые столбцы `correlation_mean` и `correlation_std`. Пропуски остаются пустыми;
неприменимость метода указана в `status` и `error`. Число наблюдаемых результатов —
`n_runs` для синтетики и `n_folds` для реальных данных.

Запись `0.015 ± 0.024` означает mean ± sample SD по пяти наборам:
`mean = sum(r_i)/5`, `SD = sqrt(sum((r_i-mean)^2)/(5-1))`.
Это разброс запусков, не standard error и не доверительный интервал. Значения
после `;` в Markdown относятся к следующей компоненте. На реальных данных SD
считается по пяти внешним folds; их обучающие части пересекаются. SD и std
обозначают одно и то же стандартное отклонение.

Столбцы подробных CSV:

- `component` — номер пары канонических переменных: `u_j = X a_j`,
  `v_j = Y b_j`. Это взвешенные комбинации признаков двух представлений.
  Номера начинаются с 1; корреляция относится к паре с этим номером.
- `train_correlation` — Pearson correlation этой пары на обучающей выборке.
- `validation_correlation_mean` и `validation_correlation_std` — mean/std
  по внутренним validation folds для выбранной пары штрафов. Подбор использует
  только первую компоненту (`validation_component=1`), поэтому эти значения
  повторяются в строках остальных компонент модели. Для методов без подбора
  поля пустые. Это оценка для выбора параметров, а не независимый test.
- `test_correlation` в `real_runs.csv` — корреляция на внешнем test fold,
  который не участвовал в подборе штрафов. В строках `full_data` поле пустое:
  итоговая модель обучена на всех данных, собственного test у неё нет.
- `population_correlation` в `synthetic_runs.csv` — точная корреляция найденных
  проекций в заданном генератором распределении. Она вычисляется по известной
  covariance, а не на train. Отдельная синтетическая test-выборка не создаётся.
- `max_angle_x_mean_deg` и `max_angle_x_std_deg` в сводке — mean/std максимального
  principal angle между найденным и истинным сигнальными подпространствами X.
  Для Y смысл аналогичен. В каждом повторе сначала берётся максимальный угол,
  затем результаты пяти повторов агрегируются. 0° означает совпадение;
  близость к 90° — слабое восстановление. При отсутствии истинных связей
  поле пустое. Это метрика всего сигнального подпространства, поэтому она
  повторяется в строках компонент. Углы относительно R для Correctness
  находятся отдельно в `correctness.csv`.

Два графика корреляций автоматически сохраняются в PNG:
`synthetic_correlations` и `real_correlations`. Точки — средние корреляции,
отрезки — ±1 SD. В синтетике пунктир отмечает заданную популяционную корреляцию.
Отрезки SD могут выходить за [-1,1], поскольку они не являются границами
допустимых корреляций или доверительными интервалами.

Ещё два PNG показывают подбор штрафов нашего CanCorr:
`penalty_cv_real.png` — validation grid внутри каждого внешнего train для двух
реальных наборов, выбранная пара отмечена красным; `penalty_frequencies_synthetic.png`
— число выборов каждой пары по пяти синтетическим повторениям. Цвета validation
grid показывают средние, заголовки — mean/std выбранной пары по внутренним folds;
mean/std всех кандидатов находятся в CSV. Цветовая шкала отдельная для каждого
реального набора. Full-data fits есть в CSV и не участвуют во внешней оценке.

Сравниваются исходный CanCorr из закреплённой ревизии, изменённый ridge CanCorr
с K-fold, sklearn CCA и PLSCanonical, CCA-Zoo RidgeCCA, R CCA::rcc/cc и PMA::CCA.
PMA — отдельный sparse CCA baseline; numerical reference для ridge — CCA::rcc.
CCA-Zoo 4.0.1 называет прежний rCCA `RidgeCCA` и используется только как baseline.
Эти пакеты не добавляют runtime-зависимостей расширению statsmodels.

Синтетика обучается на полном n из proposal: `(500,20,20)`, `(50,100,100)`,
`(200,500,200)`. Для каждого размера — отсутствие связей, одна связь ρ=.8,
две связи ρ=(.8,.5), по пять независимых повторений. Генератор задаёт точную
популяционную covariance; качество найденных проекций и principal angles
вычисляются относительно этой известной истины. Отдельный маленький test
не отнимает объекты у обучения. Прежние одиночные запуски с 80/20 заменены.

Реальные данные: Nutrimouse и UCI Multiple Features как кандидат второго набора.
Используется внешняя 5-fold оценка; внутри каждого внешнего train — новый
5-fold подбор штрафов. Внешний test не участвует ни в подборе, ни в preprocessing.
Отдельно строится итоговый fit с подбором на всех данных. UCI использует все
1994 уникальные пары, без прежнего искусственного train=100.
[Происхождение и ограничения реальных данных](data/README.md).

Все ridge-реализации получают одну сетку 25 пар α из `{.001,.01,.1,1,10}`,
одинаковые folds и согласованные шкалы регуляризации. R самостоятельно выбирает
пару, используя exact grid scorer с повторным использованием fold SVD;
итоговая модель обучается настоящим `CCA::rcc`. Scorer дополнительно проверяется
на validation fold против package fit. PMA отдельно подбирает 25 пар L1 bounds
из `{.25,.4,.6,.8,1}` и использует 100 итераций. Критерий — средняя знаковая
корреляция первой компоненты. Сетка, силы связей, seeds, число компонент и
повторений — настройки эксперимента в [proposal.json](benchmarks/proposal.json);
из proposal взяты размеры и требования к сравнению.

Для проверки пользы подбора добавлен контроль `ridge_fixed`: тот же CanCorr
с заранее фиксированными `fixed_penalty=(1,1)`, на тех же обучающих данных.
Фиксированная пара — настройка контрольного эксперимента. Внешний test не
используется для её выбора или изменения сетки. Разность качества сначала
считается для каждого совпадающего повторения/fold и затем агрегируется.
Это сравнение с одной фиксированной парой, а не с неизвестным оптимальным штрафом.

Первая test-корреляция на реальных данных (mean ± sample std по внешним folds):

| Данные | K-fold подбор | Фиксированные (1,1) | Парная разность CV − fixed |
|---|---|---|---|
| Nutrimouse | .9468 ± .0416 | .9074 ± .0262 | .0395 ± .0632 |
| UCI | .99885 ± .00016 | .99591 ± .00038 | .00294 ± .00044 |

На Nutrimouse подбор лучше фиксированной пары в четырёх из пяти внешних folds,
на UCI — в пяти из пяти. На широкой синтетике результат неоднородный: например,
при одной связи и (50,100,100) CV даёт .0152 ± .0240 против .0269 ± .0270
у фиксированной пары. Общего преимущества CV на всех конфигурациях нет.

Выбор максимума validation score проверяет выполнение критерия, совпадение с R —
реализацию, а внешнее качество — пользу подбора. Малый разрыв между кандидатами
показывает слабое различие их средних validation scores; различающиеся пары
могут давать близкое качество. Выбор на границе сетки означает, что максимум
за пределами проверенных значений не исследован. Стабильность пары и отсутствие
пограничных выборов сами по себе не доказывают оптимальность.

## Correctness

На тех же матрицах и штрафах проверяются relative error roots и Pearson
correlations относительно R, а также principal angles. Сравниваются все
наблюдаемые компоненты и отдельно вложенные ведущие подпространства размерностей
1, 2, 3. Дополнительные нулевые корни R учитываются отдельно. Для обычного
режима проверяется точное совпадение с исходным CanCorr при нулевых штрафах.

Совпадение с R проверяет реализацию. Восстановление planted truth проверяет
качество оценивания; это другая метрика. Значения mean ± SD по пяти повторениям
или внешним folds не являются p-values или доверительными интервалами.
Permutation tests исключены из текущих экспериментов по запросу владельца.
Классическая F-аппроксимация при ridge не применяется.

## Воспроизведение

В установленном окружении (из корня проекта):

```bash
MPLCONFIGDIR=/tmp/hdda-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.run_experiments
MPLCONFIGDIR=/tmp/hdda-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check benchmarks research tests
```

`--resume` продолжает совместимый checkpoint. `--verify-correctness-only`
перепроверяет сохранённые выбранные модели против R, включая ведущие подпространства,
без повторного подбора штрафов. Результаты записываются после каждого случая.
Полный test suite statsmodels и полный сайт документации пока не проверены.

Для экспорта из уже сохранённого JSON без повторения экспериментов:

```bash
MPLCONFIGDIR=/tmp/hdda-matplotlib .venv/bin/python -m benchmarks.export_results
```

Генератор и folds используют явные seeds и локальные NumPy RNG. Для PMA
задаётся R seed; RNGkind установлен явно. Скрипт ограничивает численные
библиотеки одним потоком. Это воспроизводимость в закреплённом окружении;
побитовое совпадение между разными BLAS/версиями библиотек не предполагается.
Детерминированность проверена двумя полными расчётами: численные результаты,
выбранные штрафы и разбиения совпали точно. Время в `runtime.csv` измеряется
заново и между экспериментами меняется.

Для нового окружения нужны Python 3.13, R и компиляторы C/C++/Fortran:

```bash
git clone https://github.com/statsmodels/statsmodels.git statsmodels-checkout
git -C statsmodels-checkout checkout -b codex/ridge-cancorr 295b96e65000793aeaf9a95fd7e97e73bace6557
git -C statsmodels-checkout apply ../statsmodels.patch
uv --cache-dir /tmp/hdda-uv-cache venv --python 3.13 .venv
uv --cache-dir /tmp/hdda-uv-cache pip sync --python .venv/bin/python requirements.lock
CCACHE_DIR=/tmp/hdda-ccache uv --cache-dir /tmp/hdda-uv-cache pip install --python .venv/bin/python --no-build-isolation --no-deps -e ./statsmodels-checkout --config-settings=compile-args=-j2
Rscript -e 'dir.create(".r-library", showWarnings=FALSE); install.packages(c("CCA", "PMA", "jsonlite"), lib=normalizePath(".r-library"), repos="https://cloud.r-project.org")'
.venv/bin/python -m research.fetch_stage1
MPLCONFIGDIR=/tmp/hdda-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.run_stage1
```

Использованные версии R/CCA/PMA и Python-пакетов записаны в результатах;
Python-зависимости закреплены в `requirements.lock`. R-сверка численных fixtures
и Nutrimouse отдельно воспроизводится через `research.run_stage1`.
