# Численная задача ridge CCA и открытые вопросы

Это техническая записка о формулах. Scope и статус определяются
[планом](project_plan_ru.md) и [README](../README.md).

## Задача

После центрирования train X и Y, при необходимости стандартизации по train:

```text
Sxx = XᵀX/(n−1), Syy = YᵀY/(n−1), Sxy = XᵀY/(n−1)
μx = tr(Sxx)/p, μy = tr(Syy)/q
λx = αx μx, λy = αy μy
Gx = Sxx + λx I, Gy = Syy + λy I
max aᵀ Sxy b subject to aᵀGx a = bᵀGy b = 1
```

Для нескольких компонент AᵀGx A=I, BᵀGy B=I. Замена переменных даёт SVD
матрицы Gx⁻¹ᐟ² Sxy Gy⁻¹ᐟ². Это covariance-whitening ridge: фильтр ridge-регрессии
s/(s²+λ), приведённый в PDF, сам по себе не задаёт этот алгоритм.

## Thin-SVD реализация

Пусть X=Ux Dx Vxᵀ, Y=Uy Dy Vyᵀ на численно ненулевых направлениях:

```text
hx_i = sqrt(dx_i² + (n−1)λx), hy_j = sqrt(dy_j² + (n−1)λy)
Fx = diag(dx_i/hx_i), Fy = diag(dy_j/hy_j)
Fx Uxᵀ Uy Fy = P diag(d) Qᵀ
A = sqrt(n−1) Vx diag(1/hx_i) P
B = sqrt(n−1) Vy diag(1/hy_j) Q
```

Подстановка даёт AᵀGx A=I, BᵀGy B=I, AᵀSxy B=diag(d).
При λ=0 и full column ranks roots совпадают с CanCorr. Его коэффициенты
меньше в sqrt(n−1) раз из-за нормировки scores на единичную сумму квадратов.
В изменённом CanCorr коэффициенты сохраняют эту прежнюю нормировку:
приведённые выше A,B делятся на sqrt(n−1); стандартизация включается в
коэффициенты для применения к исходным признакам. При обоих нулевых штрафах
и standardize=False используется прежний код без изменения вычислений.
В ridge-пути нулевой штраф требует полного column rank соответствующей view.
Положительные штрафы допускают wide/collinear данные; сохраняются направления
с singular value выше max(tolerance, s_max·max(n,p)·machine_epsilon).

## Reference и смысл результатов

R CCA::rcc использует sample covariance с additive diagonal penalty:
на одинаково обработанных данных lambda1=αx μx, lambda2=αy μy.
Для classical case подходит также CCA::cc. PMA::CCA решает sparse-задачу
и не является numerical oracle для ridge. Python-реализация сверена с R CCA 1.2.2.

Penalized roots d — значения штрафной задачи. Pearson(Xa,Yb) — отдельные
train correlations; при ridge они не обязаны совпадать с d. Correlations
новых scores могут быть отрицательными. Transform использует train means/scales;
знак и порядок компонент фиксируются по train. Ridge scores ортогональны
в метрике G, но не обязательно декоррелированы в исходной covariance метрике.

## Significance testing: permutation test для каждой канонической корреляции

Classical Hk: ρk=…=ρm=0, то есть population CCA rank ≤ k−1.
Глобальная независимость X и Y — другая гипотеза; вне Gaussian модели она
не эквивалентна одной лишь нулевой cross-covariance.

В статье Winkler et al. для обычного CCA permutation test строится с учётом
ранее объяснённой вариации, дополнением базисов при p≠q и коррекцией
множественных проверок. Это изучаемый способ выполнить требование proposal.
Прямая подстановка ridge-оценивателя не обоснована: ridge зависит от метрики
признаков, а scores не имеют всех classical orthogonality свойств. Если штрафы
подбираются по данным, подбор входит в повторяемую permutation-процедуру.
Нужны сценарии, где предыдущие связи существуют, а проверяемая и последующие
отсутствуют; проверки полного отсутствия связей недостаточно.
В текущем коде p-values нет. Существующая Wilks' Λ F-аппроксимация не
переносится автоматически на значения штрафной задачи.

Исходники references и SHA-256:
[manifest](../data/sources.json). Алгоритмы других пакетов изучаются
как references, не как новые поставляемые пользователю продукты.
