
# f(x) = xの2乗 x xの2乗
# f(x)が最小値になるxを求める

# `range` を使用する場合、[-1, 1] とすると、-1, 0, 1 のように整数の範囲内で値が選択され、
# [-1., 1.] とすると -1.0, -0.999...., のように実数値の範囲内で選択されます。
# 今回はパラメータを連続値の `range` を用いて定義します。

import ax

parameters = [
    {'name': 'x1', 'type': 'range', 'bounds': [-10., 10.]},
    {'name': 'x2', 'type': 'range', 'bounds': [-10., 10.]},
]


def evaluation_function(parameters):
    x1 = parameters.get('x1')
    x2 = parameters.get('x2')
    f = x1**2 + x2**2
    return f


results = ax.optimize(parameters, evaluation_function, minimize=True, random_seed=0)

# 結果の確認
best_parameters, values, experiment, model = results

# 最適化後に得られたパラメータ
print(best_parameters)

# 最適化後に得られたパラメータでの関数の値
print(values)
