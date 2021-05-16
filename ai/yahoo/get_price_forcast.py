import numpy as np
import pandas as pd
from sklearn import tree


def train_data(arr):
    print(type(arr))
    train_x = []
    train_t = []
    # 30 日間のデータを学習、1日ずつ後ろにずらしていく
    for i in np.arange(-30, -15):
        print(i)
        s = i + 14  # 14 日間の変化を素性する
        feature = arr.iloc[i:s]
        print(type(feature))
        print(type(feature.values))
        print(feature.values)
        print(feature)
        # -1 で最後の値を取り出す
        if feature.values[-1] < arr.iloc[s]:  # その翌日、株価は上がったか？
            train_t.append(1)  # YES なら 1 を
        else:
            train_t.append(0)  # NO　なら 0 を
        train_x.append(feature.values)
    # 上げ下げの結果と教師データのセットを返す
    return np.array(train_x), np.array(train_t)


# ファイルを読込み
with open('price', 'r', encoding="utf-8") as f:
    file_data = [float(data.strip()) for data in f.readlines()]

print(file_data)

# 騰落率を求める
# df = pd.DataFrame(file_data)
series = pd.Series(file_data)
print(series.values)
returns = series.pct_change()
# 累積積を求める
ret_index = (1 + returns).cumprod()
# 最初の値を 1.0 にする
ret_index[0] = 1
# リターンインデックスを教師データを取り出す
train_X, train_T = train_data(ret_index)
# 決定木のインスタンスを生成
clf = tree.DecisionTreeClassifier()
# 学習させる
clf.fit(train_X, train_T)

test_y = []
# 過去 30 日間のデータでテストをする
for i1 in np.arange(-30, -15):
    s1 = i1 + 14
    # リターンインデックスのt全く同じ期間をテストとして分類させてみる
    test_X = ret_index.iloc[i1:s1].values
    print('type:' + str(type(test_X)))
    print(test_X)
    test_X = test_X.reshape([1, -1])
    print(test_X)

    # 結果を格納して返す
    result = clf.predict(test_X)
    test_y.append(result[0])

print(train_T)  # 期待すべき答え
print(np.array(test_y))  # 分類器が出した予測
