import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    date = '20220417'
    date_path = f'data2/ml_data_{date}'
    model_path = "model/results_20220316"

    # 新規データ
    df_date = pd.read_csv(date_path, header=0)
    # 説明変数列
    x_cols_name = 'x_cols.csv'
    x_cols = pd.read_csv(os.path.join(model_path, x_cols_name))
    # dataFrame型からobject型?に変換
    x_cols = x_cols['x_cols']
    x = df_date[x_cols].copy()

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

    # 0 or 1
    pred_g = model_g.predict(x)
    pred_r = model_r.predict(x)
    pred_t = model_t.predict(x)
    # 確率
    pred_proba_g = model_g.predict_proba(x)[:, 1]
    pred_proba_r = model_r.predict_proba(x)[:, 1]
    pred_proba_t = model_t.predict_proba(x)[:, 1]

    pred = pd.DataFrame({'pred_g': pred_g, 'pred_r': pred_r, 'pred_t': pred_t,
                        'pred_proba_g': pred_proba_g, 'pred_proba_r': pred_proba_r, 'pred_proba_t': pred_proba_t})
    # pred_viz = pred[['pred_proba_g']]

    plt.figure()
    # .Tは転置
    sns.heatmap(pred.T)
    plt.savefig('data4/seaborn_heatmap_list.png')
    plt.close('all')
