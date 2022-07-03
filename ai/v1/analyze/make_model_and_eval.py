# 機械学習モデルの構築・評価 p272
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
import os


# y_ : 目的変数 従属変数 正解
# x_ : 説明変数 独立変数
# train : 学習データ
# test  : テストデータ
def make_model_and_eval(l_model, l_x_train, l_x_test, l_y_train, l_y_test):
    # モデル構築
    l_model.fit(l_x_train, l_y_train)
    # 構築したモデルで予測
    y_pred_train = l_model.predict(l_x_train)
    y_pred_test = l_model.predict(l_x_test)

    acc_train = accuracy_score(l_y_train, y_pred_train)
    acc_test = accuracy_score(l_y_test, y_pred_test)
    # 多クラス分類の場合、average='micro'を付ける。
    f1_train = f1_score(l_y_train, y_pred_train, average='micro')
    f1_test = f1_score(l_y_test, y_pred_test, average='micro')
    recall_train = recall_score(l_y_train, y_pred_train, average='micro')
    recall_test = recall_score(l_y_test, y_pred_test, average='micro')
    precision_train = precision_score(l_y_train, y_pred_train, average='micro')
    precision_test = precision_score(l_y_test, y_pred_test, average='micro')
    # TODO 未利用
    result_train = confusion_matrix(l_y_train, y_pred_train).ravel()
    result_test = confusion_matrix(l_y_test, y_pred_test).ravel()
    score_train = pd.DataFrame({'DataCategory': ['train'],
                                'acc': [acc_train],
                                'f1': [f1_train],
                                'recall': [recall_train],
                                'precision': [precision_train]})
    score_test = pd.DataFrame({'DataCategory': ['test'],
                               'acc': [acc_test],
                               'f1': [f1_test],
                               'recall': [recall_test],
                               'precision': [precision_test]
                               })
    l_score = pd.concat([score_train, score_test], ignore_index=True)
    l_importance = pd.DataFrame({'cols': l_x_train.columns, 'importance': l_model.feature_importances_})
    l_importance = l_importance.sort_values('importance', ascending=False)
    l_cols = pd.DataFrame({'x_cols': l_x_train.columns})
    return l_score, l_importance, l_model, l_cols


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    date_start = 20220509
    date_end = 20220619
    path = 'data/ml_base_data'
    output_ml_result_dir = 'model/'
    ml_base_data = pd.read_csv(path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]

    train_data, test_data = train_test_split(ml_base_data, test_size=0.3, random_state=0)
    print(f'Train:{len(train_data)}件/ Test:{len(test_data)}')
    print(f'price_status Train  3:{len(train_data.loc[train_data["price_result"]==3])}件')
    print(f'price_status Train  2:{len(train_data.loc[train_data["price_result"]==2])}件')
    print(f'price_status Train  1:{len(train_data.loc[train_data["price_result"]==1])}件')
    print(f'price_status Train  0:{len(train_data.loc[train_data["price_result"]==0])}件')
    print(f'price_status Test  3:{len(test_data.loc[test_data["price_result"]==3])}件')
    print(f'price_status Test  2:{len(test_data.loc[test_data["price_result"]==2])}件')
    print(f'price_status Test  1:{len(test_data.loc[test_data["price_result"]==1])}件')
    print(f'price_status Test  0:{len(test_data.loc[test_data["price_result"]==0])}件')
    print(f'ans Train positive:{len(train_data.loc[train_data["positive"]>=train_data["negative"]])}件')
    print(f'ans Train negative:{len(train_data.loc[train_data["positive"]<train_data["negative"]])}件')
    print(f'ans Test  positive:{len(test_data.loc[test_data["positive"]>=test_data["negative"]])}件')
    print(f'ans Test  negative:{len(test_data.loc[test_data["positive"]<test_data["negative"]])}件')

    x_cols = list(train_data.columns)
    # 目的変数と説明変数を分離する
    x_cols.remove('price_result')
    # 説明変数名から目的変数名を削除
    x_cols.remove('id')
    x_cols.remove('s_date')
    x_cols.remove('closed_flag')

    # 目的変数名
    y_targets = ['price_result']

    target_output_dir_name = f'results_{date_start}'
    target_output_dir = os.path.join(output_ml_result_dir, target_output_dir_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(target_output_dir)

    score_all = []
    importance_all = []

    for y_target in y_targets:
        y_train = train_data[y_target]
        x_train = train_data[x_cols]
        y_test = test_data[y_target]
        x_test = test_data[x_cols]

        models = {'tree': DecisionTreeClassifier(random_state=0),
                  'RandomForest': RandomForestClassifier(random_state=0),
                  'GradientBoosting': GradientBoostingClassifier(random_state=0)}

        for model_name, model in models.items():
            print(model_name)
            score, importance, model, cols = make_model_and_eval(model, x_train, x_test, y_train, y_test)
            score['model_name'] = model_name
            importance['model_name'] = model_name
            score['model_target'] = y_target
            print(score)
            importance['model_target'] = y_target
            print(importance)

            model_name = f'model_{y_target}_{model_name}.pickle'
            model_path = os.path.join(target_output_dir, model_name)
            with open(model_path, mode='wb') as f:
                pickle.dump(model, f, protocol=2)
            score_all.append(score)
            importance_all.append(importance)

    # listの要素を列にする
    # axis=0とすると縦方向(列)に連結される。これがデフォルトなので省略しても同じ。
    p_score_all = pd.concat(score_all, ignore_index=True)
    p_importance_all = pd.concat(importance_all, ignore_index=True)
    # ディクショナリー key : valueがlist
    p_cols = pd.DataFrame({'x_cols': x_cols})

    score_name = 'score.csv'
    importance_name = 'importance.csv'
    cols_name = 'x_cols.csv'
    score_path = os.path.join(target_output_dir, score_name)
    importance_path = os.path.join(target_output_dir, importance_name)
    cols_path = os.path.join(target_output_dir, cols_name)
    p_score_all.to_csv(score_path, index=False)
    p_importance_all.to_csv(importance_path, index=False)
    p_cols.to_csv(cols_path, index=False)
