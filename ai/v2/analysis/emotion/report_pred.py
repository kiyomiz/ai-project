import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report

from datetime import datetime
from dateutil.relativedelta import relativedelta
from ai.ai.v2.analysis.common.date_utils import get_last_date


if __name__ == '__main__':
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.expand_frame_repr', False)
    mouth_period = 1
    # date_start = '20220801'
    date_start = '20220901'
    # date_start = '20221001'

    data_dir = 'data-1'
    data_path = f'{data_dir}/ml_base_data'
    model_path = "model-1/20220509-3"

    # 日付
    tdatetime = datetime.strptime(str(date_start), '%Y%m%d')
    tdatetime = tdatetime + relativedelta(months=mouth_period - 1)
    date_end = get_last_date(tdatetime).strftime('%Y%m%d')

    # model
    model_g_name = 'model_label_GradientBoosting.pickle'
    model_r_name = 'model_label_RandomForest.pickle'
    model_t_name = 'model_label_tree.pickle'
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
    df_data = pd.read_csv(data_path, header=0)
    df_data['s_date'] = df_data['s_date'].astype(str)

    # 正解ラベルを付与
    # 変化率(%)
    #  1:  上昇
    #  0:  下降
    df_data.loc[df_data['delta_ratio'] > 0, 'label'] = 1
    df_data.loc[df_data['delta_ratio'] < 0, 'label'] = 0
    # 変動なしは、正解・不正解なしなので除外
    df_data = df_data[(df_data['label'] == 0) | (df_data['label'] == 1)]

    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")

    view_data = pd.DataFrame()
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    for index, date in enumerate(date_ary.values):
        print(f'{index}={date}')
        # 予測に使用するデータ
        x = df_data.loc[df_data['s_date'] == date, x_cols].copy()
        # 正解データ
        price_result = df_data.loc[df_data['s_date'] == date, 'label'].max()

        if len(x) != 0:
            # 予測
            # 0:株価が下がる or 1:株価が上がる
            # pred_g = model_g.predict(x)
            # pred_r = model_r.predict(x)
            # pred_t = model_t.predict(x)

            # 1: 株価が上がる確率 ([:, 0]で、0:株価が下がる確率)
            pred_proba_g = model_g.predict_proba(x)[:, 1]
            pred_proba_r = model_r.predict_proba(x)[:, 1]
            pred_proba_t = model_t.predict_proba(x)[:, 1]

            # 予測結果格納
            pred = pd.DataFrame({'s_date': date,
                                 'proba_grad': pred_proba_g,
                                 'proba_rand': pred_proba_r,
                                 'proba_tree': pred_proba_t}
                                )

            # 予測結果データと検証データを結合
            report_valid = pd.merge(pred, df_data, on='s_date', how='inner')

            # 日付でグルーピング s_date毎にlabelは変化するので平均を使えば元の値と同じになる
            view_data = view_data.append(report_valid.groupby('s_date').mean()[['proba_grad', 'proba_rand', 'proba_tree', 'label']])

    pred_grad = []
    pred_rand = []
    pred_tree = []
    Y = []
    # += でリストに要素を追加
    pred_grad += [int(round(l)) for l in view_data['proba_grad']]
    pred_rand += [int(round(l)) for l in view_data['proba_rand']]
    pred_tree += [int(round(l)) for l in view_data['proba_tree']]
    pred = {'grad': pred_grad, 'rand': pred_rand, 'tree': pred_tree}
    Y += [int(l) for l in view_data['label']]

    for k, v in pred.items():
        print(f'*** {k} ***')
        # 適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)，マクロ平均，マイクロ平均
        print(classification_report(Y, v))
