import os
import pandas as pd
import torch
from ai.ai.v2.analysis.nlp.lstm.make_model_and_eval import Net, Collate
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

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
    data_path = f'../{data_dir}/ml_base_data'
    vocab_dir = 'vocab'
    vocab_path = f'{vocab_dir}/20220509-20220731'
    model_path = 'model-1/'
    model_name = '20220509-3-1-False'

    # 日付
    tdatetime = datetime.strptime(str(date_start), '%Y%m%d')
    tdatetime = tdatetime + relativedelta(months=mouth_period - 1)
    date_end = int(get_last_date(tdatetime).strftime('%Y%m%d'))

    # 辞書の読込み
    collate = Collate(vocab_path, None)

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
    del ml_base_data['delta_ratio']

    # 変動なしは、正解・不正解なしなので除外
    ml_base_data = ml_base_data[(ml_base_data['label'] == 0) | (ml_base_data['label'] == 1)]

    # 分かち書きを実行
    ml_base_data['text'] = ml_base_data['text'].apply(Collate.tokenize)
    # labelはfloatから文字列に変換(vocabが文字列のみのため)
    ml_base_data['label'] = ml_base_data['label'].map('{:.0f}'.format)

    print(f'sum:{len(ml_base_data)}件')
    print(f'price_status 1:{len(ml_base_data.loc[ml_base_data["label"]=="1"])}件')
    print(f'price_status 0:{len(ml_base_data.loc[ml_base_data["label"]=="0"])}件')

    # バッチサイズ
    batch_size = 200
    # loader
    test_loader = DataLoader(ml_base_data.values, batch_size, collate_fn=collate.collate_batch)

    # model
    model_path = os.path.join(model_path, model_name)

    # モデルの読込み
    # パラメータのみ 学習時のサイズと異なるとエラーになる。
    # n_input = 200000
    # n_embed = 100
    # n_hidden = 100
    # n_layers = 3
    # n_output = 2
    # model = Net(n_input, n_embed, n_hidden, n_layers, n_output)
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
        # argmax関数で最大値のインデックスを取得
        # 予測値において最も値が大きなクラス(最も確率が高い)の番号を取得
        pred += [l.argmax() for l in output]
        Y += [l for l in y]

    # 適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)，マクロ平均，マイクロ平均
    print(classification_report(Y, pred))
