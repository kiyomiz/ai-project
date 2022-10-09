import os
import pandas as pd
import torch
from ai.ai.v1.analiyze_pytorch.make_model_and_eval import Net, Collate
from torch.utils.data import DataLoader
import MeCab
from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings('ignore')

mecab = MeCab.Tagger('-Owakati')

def tokenize(x):
    return mecab.parse(x).split(' ')[:-1]


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    # date_start = 20220601
    # date_end = 20220630
    # date_start = 20220701
    # date_end = 20220731
    date_start = 20220801
    date_end = 20220831
    # date_start = 20220901
    # date_end = 20220923

    data_dir = 'data2'
    data_path = f'{data_dir}/ml_base_data'
    vocab_dir = 'vocab2-5'
    vocab_path = f'{vocab_dir}/20220509-20220923'
    model_path = 'model-1-epoch4/'
    model_name = '20220509-3'

    # 辞書の読込み
    collate = Collate(vocab_path, None)

    # データの読込み
    ml_base_data = pd.read_csv(data_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]
    del ml_base_data['id']
    del ml_base_data['s_date']

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
    # パラメータのみ 学習時のサイズと異なるとエラーになる。
    # n_input = 200000
    # n_embed = 100
    # n_hidden = 100
    # n_layers = 3
    # n_output = 4

    model_path = os.path.join(model_path, model_name)

    # モデルの読込み
    # パラメータのみ
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
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    # 適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)，マクロ平均，マイクロ平均
    print(classification_report(Y, pred))
