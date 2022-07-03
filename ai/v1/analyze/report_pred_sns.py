import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def evaluation(df):
    df.columns = ['s_date', 'kind', 'price_result', 'cnt']
    view_data = df.loc[df.groupby(['s_date'])['cnt'].idxmax()].reset_index(drop=True)
    view_data.loc[(view_data['kind'] == view_data['price_result']), 'correct'] = 1
    return view_data['correct']


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    date_start = '20220620'
    date_end = '20220630'
    date_path = 'data/ml_base_data'
    model_path = "model/results_20220509"

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
        # 正解データ
        price_result = df_date.loc[df_date['s_date'] == date, 'price_result'].max()

        if len(x) != 0:
            # 予測
            # 前日比(%)
            #  3:  x > 1
            #  2:  0 <= x < 1
            #  1: -1 <= x < 0
            #  0: -1 > x
            # のどれかになる 1次元のリスト
            pred_g = model_g.predict(x)
            pred_r = model_r.predict(x)
            pred_t = model_t.predict(x)

            # 1ツイートについて、4クラスのそれぞれの出現確率 2次元のリスト
            #pred_proba_g = model_g.predict_proba(x)
            #pred_proba_r = model_r.predict_proba(x)
            #pred_proba_t = model_t.predict_proba(x)
            #print(pred_proba_g)
            #print(pred_proba_r)
            #print(pred_proba_t)

            # 予測結果格納
            pred = pred.append(pd.DataFrame({'s_date': date, 'grad': pred_g, 'rand': pred_r, 'tree': pred_t,
                                             'price_result': price_result}))

    # print(pred)

    # 予測結果検証
    # 日付ごとにグルーピング s_date毎にprice_resultは変化するので、値の分類の数を数える
    # 's_date', 'grad'は、indexとして格納。<class 'pandas.core.series.Series'>
    # report_valid_grad = pred.groupby(['s_date', 'grad', 'price_result']).size().reset_index()
    df_eval = pd.DataFrame(
        {'acc_grad': evaluation(pred.groupby(['s_date', 'grad', 'price_result']).size().reset_index()),
         'acc_rand': evaluation(pred.groupby(['s_date', 'rand', 'price_result']).size().reset_index()),
         'acc_tree': evaluation(pred.groupby(['s_date', 'tree', 'price_result']).size().reset_index())
         })

    print(df_eval)

    plt.figure()
    # .Tは転置
    # annot : 数値表示
    sns.heatmap(df_eval.T, cmap='Blues', annot=True, yticklabels=3)
    plt.savefig('data/seaborn_heatmap_list.png')
    plt.close('all')
