import collections
import os
from enum import Enum

import pandas as pd
import torch

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split

from ai.ai.v2.analysis.nlp.tfidf.make_model_and_eval import Net
from ai.ai.v2.analysis.nlp.tfidf.make_ml_vocab import get_tfidf

from datetime import datetime
from dateutil.relativedelta import relativedelta
from ai.ai.v2.analysis.common.date_utils import get_last_date

import warnings

warnings.filterwarnings('ignore')


def collect_one_day_top(p):
    # 1.上昇確率:値が大きい10件の平均
    # 0.下降確率:値が小さい10件の平均
    p.sort()
    p_1 = p[-10:]
    p_0 = p[:10]
    p_a = (sum(p_1) + sum(p_0)) / (len(p_1) + len(p_0))
    return round(p_a.item())


def collect_one_day_all(p):
    # pはlist型で、値がTensor型
    p_r = sum(p)/len(p)
    # item()で、Tensor型から値を取得
    return round(p_r.item())


# 列挙型の定義
class Collect(Enum):
    ALL = 1
    TOP = 2


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    mouth_period = 1
    # date_start = 20220801
    date_start = 20220901
    # date_start = 20221001

    data_dir = 'data-2'
    data_path = f'../{data_dir}/ml_base_data'
    vocab_dir = 'vocab'
    vocab_path = f'{vocab_dir}/20220509-20220731'
    model_path = 'model-2/'
    model_name = '20220509-3-1-False'
    collect_type = Collect.TOP

    # 集計方法
    if collect_type == Collect.ALL:
        collect_one_day = collect_one_day_all
    elif collect_type == Collect.TOP:
        collect_one_day = collect_one_day_top

    # 日付
    tdatetime = datetime.strptime(str(date_start), '%Y%m%d')
    tdatetime = tdatetime + relativedelta(months=mouth_period - 1)
    date_end = int(get_last_date(tdatetime).strftime('%Y%m%d'))

    # データの読込み
    ml_base_data = pd.read_csv(data_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]

    # 正解ラベルを付与
    # 変化率(%)
    #  1:  上昇
    #  0:  下降
    ml_base_data.loc[ml_base_data['delta_ratio'] > 0, 'label'] = 1
    ml_base_data.loc[ml_base_data['delta_ratio'] < 0, 'label'] = 0

    # 不要項目削除
    del ml_base_data['id']
    # del ml_base_data['s_date']
    del ml_base_data['favorite_count']
    del ml_base_data['retweet_count']

    # 変動なしは、正解・不正解なしなので除外
    ml_base_data = ml_base_data[(ml_base_data['label'] == 0) | (ml_base_data['label'] == 1)]

    print(f'sum:{len(ml_base_data)}件')
    print(f'price_status 1:{len(ml_base_data.loc[ml_base_data["label"]==1])}件')
    print(f'price_status 0:{len(ml_base_data.loc[ml_base_data["label"]==0])}件')

    # モデルパス
    model_path = os.path.join(model_path, model_name)
    # モデルの読込み モデル全体
    model = torch.load(model_path)
    # 評価モードにする
    model = model.eval()

    # 結果
    pred = []
    Y = []

    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=f'{date_start}', end=f'{date_end}', freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    for index, date in enumerate(date_ary.values):
        print(f'{index}={date}')
        ml_base_data_d = ml_base_data.loc[ml_base_data["s_date"] == float(date)]

        # データをtf_idf, labelに変換
        tf_idf, labels, dictionary = get_tfidf(vocab_path, ml_base_data_d)

        if tf_idf is None:
            continue

        # PyTorchで学習に使用できる形式へ変換
        x = torch.tensor(tf_idf, dtype=torch.float32)
        t = torch.tensor(labels, dtype=torch.int64)
        # print(type(x), x.dtype, type(t), t.dtype)

        # 入力値と目標値をまとめて、ひとつのオブジェクトdatasetに変換
        dataset = TensorDataset(x, t)

        # ランダムに分割を行うため、シードを固定して再現性を確保
        torch.manual_seed(0)

        # バッチサイズ
        batch_size = len(ml_base_data_d)

        if batch_size > 0:
            # Data Loadkerを用意
            test_loader = torch.utils.data.DataLoader(dataset, batch_size)

            for i, (x, y) in enumerate(test_loader):
                # print(i)
                with torch.no_grad():
                    output = model(x)
                pred += [collect_one_day([l[1] for l in output])]
                Y += [collect_one_day([l for l in y])]

    # 適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)，マクロ平均，マイクロ平均
    print(classification_report(Y, pred))
