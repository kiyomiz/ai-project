import os
import pandas as pd
import torch
from ai.ai.v1.analiyze_pytorch.make_model_and_eval import Net, Collate
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
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
    date_start = 20220509
    date_end = 20220520
    date_path = 'data/ml_base_data'
    model_path = 'model/'

    ml_base_data = pd.read_csv(date_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]
    del ml_base_data['id']
    del ml_base_data['s_date']

    collate = Collate(ml_base_data)

    # バッチサイズ
    batch_size = 200
    # loader
    test_loader = DataLoader(ml_base_data.values, batch_size, collate_fn=collate.collate_batch)

    # model
    model_name = f'{date_start}'
    model_path = os.path.join(model_path, model_name)

    # モデルの読込み
    model = torch.load(model_path)
    # 評価モードにする
    model = model.eval()

    pred = []
    Y = []
    for i, (x, y) in enumerate(test_loader):
        print(i)
        with torch.no_grad():
            output = model(x)
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    # 適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)，マクロ平均，マイクロ平均
    print(classification_report(Y, pred))
