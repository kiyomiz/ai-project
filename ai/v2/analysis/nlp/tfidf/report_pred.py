import os
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

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    mouth_period = 1
    date_start = 20220801
    # date_start = 20220901
    # date_start = 20221001

    data_dir = 'data-1'
    data_path = f'../{data_dir}/r_ml_base_data'
    vocab_dir = 'vocab'
    vocab_path = f'{vocab_dir}/20220509-20220630'
    model_path = 'model-1/'
    model_name = '20220509-2'

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
    del ml_base_data['s_date']
    del ml_base_data['favorite_count']
    del ml_base_data['retweet_count']

    # 変動なしは、正解・不正解なしなので除外
    ml_base_data = ml_base_data[(ml_base_data['label'] == 0) | (ml_base_data['label'] == 1)]

    print(f'sum:{len(ml_base_data)}件')
    print(f'price_status 1:{len(ml_base_data.loc[ml_base_data["label"]==1])}件')
    print(f'price_status 0:{len(ml_base_data.loc[ml_base_data["label"]==0])}件')

    # データをtf_idf, labelに変換
    tf_idf, labels, dictionary = get_tfidf(vocab_path, ml_base_data)

    # PyTorchで学習に使用できる形式へ変換
    x = torch.tensor(tf_idf, dtype=torch.float32)
    t = torch.tensor(labels, dtype=torch.int64)
    # print(type(x), x.dtype, type(t), t.dtype)

    # 入力値と目標値をまとめて、ひとつのオブジェクトdatasetに変換
    dataset = TensorDataset(x, t)

    # ランダムに分割を行うため、シードを固定して再現性を確保
    torch.manual_seed(0)

    # バッチサイズ
    batch_size = 200

    # Data Loadkerを用意
    test_loader = torch.utils.data.DataLoader(dataset, batch_size)

    # model
    model_path = os.path.join(model_path, model_name)

    # モデルの読込み
    # パラメータのみ 学習時のサイズと異なるとエラーになる。
    # n_input = len(dictionary)
    # n_hidden = round(len(dictionary) / 2)
    # n_outputは、labelの種類の数を指定
    # n_output = 2
    # model = Net(n_input, n_hidden, n_output)
    # model.load_state_dict(torch.load(model_path))
    # モデル全体
    model = torch.load(model_path)

    # 評価モードにする
    model = model.eval()

    pred = []
    Y = []
    for i, (x, y) in enumerate(test_loader):
        # print(i)
        with torch.no_grad():
            output = model(x)
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    # 適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)，マクロ平均，マイクロ平均
    print(classification_report(Y, pred))
