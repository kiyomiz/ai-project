from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from ai.ai.v1.analiyze_pytorch.make_ml_vocab import Collate

from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta

import warnings

warnings.filterwarnings('ignore')


class Net(pl.LightningModule):

    def __init__(self, n_input, n_embed, n_hidden, n_layers, n_output):
        super().__init__()
        self.embed = nn.Embedding(n_input, n_embed, padding_idx=1)
        # 双方向LSTM : bidirectional=True
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, bidirectional=True)
        # 前方向と後ろ方向の最後の隠れ層ベクトルを結合したものを受け取るので、n_hiddenは2倍にしている
        self.fc = nn.Linear(n_hidden * 2 , n_output)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.embed(x)
        # (h, c)はタブルのそれぞれの要素を分けて取得
        x, (h, c) = self.lstm(x)
        # 双方向かつlayersが1の場合、
        # h[0]がforward（前から後ろへ）
        # h[1]がbackward（後ろから前へ）
        h_forward = h[::2, :, :]
        h_backward = h[1::2, :, :]
        bih = torch.cat([h_forward[-1], h_backward[-1]], dim=1)
        x = self.fc(bih)
        return x

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y, t), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(y, t), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc(y, t), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def get_last_date(dt):
    # calendar.monthrange()関数で、月の初日の曜日（月曜が0、日曜が6）と、月の日数のタプルが取得できる。
    # replaceは、dayを置き換えて、datetimeクラスのインスタンスを作成
    return dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    mouth_period = 2
    date_start = 20220509
    # date_start = 20220601
    # date_start = 20220701
    # date_start = 20220801

    tdatetime = datetime.strptime(str(date_start), '%Y%m%d')
    tdatetime = tdatetime + relativedelta(months=mouth_period - 1)
    date_end = int(get_last_date(tdatetime).strftime('%Y%m%d'))

    data_dir = 'data2-6'
    data_path = f'{data_dir}/ml_base_data'
    vocab_dir = 'vocab2'
    vocab_path = f'{vocab_dir}/20220509-20220923'
    model_dir = 'model'
    output_ml_result_dir = f'{model_dir}/{date_start}-{mouth_period}'

    # 辞書の読込み
    collate = Collate(vocab_path, None)

    # データの読込み
    ml_base_data = pd.read_csv(data_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]
    del ml_base_data['id']
    del ml_base_data['s_date']

    # データ件数を揃える
    ml_base_data_1 = ml_base_data.loc[ml_base_data["label"] == 1]
    ml_base_data_0 = ml_base_data.loc[ml_base_data["label"] == 0]

    if len(ml_base_data_1) > len(ml_base_data_0):
        ml_base_data_1 = ml_base_data_1.sample(
            frac=(1 - (len(ml_base_data_1) - len(ml_base_data_0)) / len(ml_base_data_1)),
            random_state=0)
        ml_base_data = ml_base_data_1.append(ml_base_data_0, ignore_index=True)
    else:
        ml_base_data_0 = ml_base_data_0.sample(
            frac=(1 - (len(ml_base_data_0) - len(ml_base_data_1)) / len(ml_base_data_0)),
            random_state=0)
        ml_base_data = ml_base_data_0.append(ml_base_data_1, ignore_index=True)

    # 分かち書きを実行
    ml_base_data['text'] = ml_base_data['text'].apply(Collate.tokenize)
    # labelはfloatから文字列に変換(vocabが文字列のみのため)
    ml_base_data['label'] = ml_base_data['label'].map('{:.0f}'.format)

    train_data, val_data = train_test_split(ml_base_data, test_size=0.4, random_state=0)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=0)
    print(f'Train:{len(train_data)}件 Val:{len(val_data)}件 Test:{len(test_data)}件')
    print(f'price_status Train  1:{len(train_data.loc[train_data["label"]=="1"])}件')
    print(f'price_status Train  0:{len(train_data.loc[train_data["label"]=="0"])}件')
    print(f'price_status Val    1:{len(val_data.loc[val_data["label"]=="1"])}件')
    print(f'price_status Val    0:{len(val_data.loc[val_data["label"]=="0"])}件')
    print(f'price_status Test   1:{len(test_data.loc[test_data["label"]=="1"])}件')
    print(f'price_status Test   0:{len(test_data.loc[test_data["label"]=="0"])}件')

    # バッチサイズ
    batch_size = 200

    # Data Loadkerを用意
    train_loader = DataLoader(train_data.values, batch_size, shuffle=True, collate_fn=collate.collate_batch)
    val_loader = DataLoader(val_data.values, batch_size, collate_fn=collate.collate_batch)
    test_loader = DataLoader(test_data.values, batch_size, collate_fn=collate.collate_batch)

    # 詳細設定
    n_input = len(collate.text_vocab)
    # n_input = 300000
    n_embed = 100
    n_hidden = 100
    n_layers = 3
    # n_outputは、labelの種類の数を指定
    n_output = 4
    # 学習回数
    epoch = 5

    # 学習の実行
    pl.seed_everything(0)
    net = Net(n_input, n_embed, n_hidden, n_layers, n_output)
    trainer = pl.Trainer(max_epochs=epoch)
    trainer.fit(net, train_loader, val_loader)

    # テストデータに対する検証
    results = trainer.test(dataloaders=test_loader)
    print(results)

    # モデルの保存
    # パラメータのみ
    # torch.save(net.state_dict(), output_ml_result_dir)
    # モデル全体
    torch.save(net, output_ml_result_dir)
