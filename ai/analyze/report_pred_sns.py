import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    date_start = '20220417'
    date_end = '20220502'
    date_path = f'data3/ml_base_data'
    model_path = "model/results_20220316"

    # model
    model_g_name = 'model_price_result_GradientBoosting.pickle'
    model_r_name = 'model_price_result_RandomForest.pickle'
    model_t_name = 'model_price_result_tree.pickle'
    model_g_path = os.path.join(model_path, model_g_name)
    model_r_path = os.path.join(model_path, model_r_name)
    model_t_path = os.path.join(model_path, model_t_name)

    with open(model_g_path, mode='rb') as f:
        model_g = pickle.load(f)
    with open(model_r_path, mode='rb') as f:
        model_r = pickle.load(f)
    with open(model_t_path, mode='rb') as f:
        model_t = pickle.load(f)

    # 説明変数列
    x_cols_name = 'x_cols.csv'
    x_cols = pd.read_csv(os.path.join(model_path, x_cols_name))
    # dataFrame型からobject型?に変換
    x_cols = x_cols['x_cols']

    # 新規データ
    df_date = pd.read_csv(date_path, header=0)
    df_date['s_date'] = df_date['s_date'].astype(str)
    # 休場は除外
    df_date = df_date.loc[df_date['closed_flag'] == 0]

    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")

    pred = pd.DataFrame()
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    for index, date in enumerate(date_ary.values):
        print(f'{index}={date}')
        # 予測に使用するデータ
        x = df_date.loc[df_date['s_date'] == date, x_cols].copy()

        if len(x) != 0:
            # 0:株価が下がる or 1:株価が上がる
            pred_g = model_g.predict(x)
            pred_r = model_r.predict(x)
            pred_t = model_t.predict(x)
            # 1: 株価が上がる確率 ([:, 0]で、0:株価が下がる確率)
            pred_proba_g = model_g.predict_proba(x)[:, 1]
            pred_proba_r = model_r.predict_proba(x)[:, 1]
            pred_proba_t = model_t.predict_proba(x)[:, 1]

            # 予測結果格納
            pred = pred.append(
                pd.DataFrame({'s_date': date,
                              'grad': pred_g, 'rand': pred_r, 'tree': pred_t,
                              'proba_grad': pred_proba_g, 'proba_rand': pred_proba_r, 'proba_tree': pred_proba_t}))

    # 予測結果データと検証データを結合
    report_valid = pd.merge(pred, df_date, on='s_date', how='left')
    # 日付ごとにグルーピング s_date毎にprice_resultは変化するので平均を使えば元の値と同じになる
    report_valid = report_valid.groupby('s_date').mean()[['grad', 'rand', 'tree', 'proba_grad', 'proba_rand', 'proba_tree', 'price_result']]
    # 予測結果検証
    view_data = report_valid.copy()
    print(view_data)
    # 正解の判定　正解が1
    view_data.loc[(view_data['proba_grad'] >= 0.5) & (view_data['price_result'] == 1), 'correct_grad'] = 1
    view_data.loc[(view_data['proba_grad'] < 0.5) & (view_data['price_result'] == 0), 'correct_grad'] = 1
    view_data.loc[(view_data['proba_rand'] >= 0.5) & (view_data['price_result'] == 1), 'correct_rand'] = 1
    view_data.loc[(view_data['proba_rand'] < 0.5) & (view_data['price_result'] == 0), 'correct_rand'] = 1
    view_data.loc[(view_data['proba_tree'] >= 0.5) & (view_data['price_result'] == 1), 'correct_tree'] = 1
    view_data.loc[(view_data['proba_tree'] < 0.5) & (view_data['price_result'] == 0), 'correct_tree'] = 1
    view_data.loc[:, 'count'] = 1
    view_data.fillna(0, inplace=True)
    view_data = view_data.groupby('s_date').sum()[['correct_grad', 'correct_rand', 'correct_tree', 'count']]
    view_data.loc[:, 'acc_grad'] = view_data['correct_grad'] / view_data['count']
    view_data.loc[:, 'acc_rand'] = view_data['correct_rand'] / view_data['count']
    view_data.loc[:, 'acc_tree'] = view_data['correct_tree'] / view_data['count']
    view_data = view_data[['acc_grad', 'acc_rand', 'acc_tree']]

    print(view_data)

    plt.figure()
    # .Tは転置
    # annot : 数値表示
    sns.heatmap(view_data.T, cmap='Blues', annot=True, yticklabels=3)
    plt.savefig('data4/seaborn_heatmap_list.png')
    plt.close('all')
