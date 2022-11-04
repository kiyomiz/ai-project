from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split

from ai.ai.v2.analysis.nlp.tfidf.make_ml_vocab import get_tfidf

import torchmetrics

from datetime import datetime
from dateutil.relativedelta import relativedelta
from ai.ai.v2.analysis.common.date_utils import get_last_date

import warnings

warnings.filterwarnings('ignore')


class Net(pl.LightningModule):

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()

        self.bn = nn.BatchNorm1d(n_input)
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        h = self.bn(x)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.softmax(h, dim=1)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        # 予測
        y = self(x)
        # 損失
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y, t), on_step=True, on_epoch=True)
        return loss

    # 検証データに対する処理
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(y, t), on_step=False, on_epoch=True)
        return loss

    # テストデータに対する処理
    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc(y, t), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    mouth_period = 3
    delta_ratio = 1  # 変化率の0からの差
    favorite_retweet_flag = True
    date_start = 20220509

    tdatetime = datetime.strptime(str(date_start), '%Y%m%d')
    tdatetime = tdatetime + relativedelta(months=mouth_period - 1)
    date_end = int(get_last_date(tdatetime).strftime('%Y%m%d'))

    data_dir = 'data-1'
    data_path = f'../{data_dir}/ml_base_data'
    vocab_dir = 'vocab'
    vocab_path = f'{vocab_dir}/20220509-20220630'
    model_dir = 'model-1'
    output_ml_result_dir = f'{model_dir}/{date_start}-{mouth_period}'

    # データの読込み
    ml_base_data = pd.read_csv(data_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]

    # 正解ラベルを付与
    # 変化率(%)
    #  1:  上昇
    #  0:  下降
    ml_base_data.loc[ml_base_data['delta_ratio'] >= delta_ratio, 'label'] = 1
    ml_base_data.loc[ml_base_data['delta_ratio'] <= -1 * delta_ratio, 'label'] = 0

    # お気に入りが0、または、リツイートが0は除外
    if favorite_retweet_flag:
        ml_base_data = ml_base_data[(ml_base_data['favorite_count'] != 0) | (ml_base_data['retweet_count'] != 0)]

    # 不要項目削除
    del ml_base_data['id']
    del ml_base_data['s_date']
    del ml_base_data['favorite_count']
    del ml_base_data['retweet_count']

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

    # データ数の参考情報を表示
    train_data, val_data = train_test_split(ml_base_data, test_size=0.4, random_state=0)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=0)
    print(f'Train:{len(train_data)}件 Val:{len(val_data)}件 Test:{len(test_data)}件')
    print(f'price_status Train  1:{len(train_data.loc[train_data["label"]==1])}件')
    print(f'price_status Train  0:{len(train_data.loc[train_data["label"]==0])}件')
    print(f'price_status Val    1:{len(val_data.loc[val_data["label"]==1])}件')
    print(f'price_status Val    0:{len(val_data.loc[val_data["label"]==0])}件')
    print(f'price_status Test   1:{len(test_data.loc[test_data["label"]==1])}件')
    print(f'price_status Test   0:{len(test_data.loc[test_data["label"]==0])}件')

    # データをtf_idf, labelに変換
    tf_idf, labels, dictionary = get_tfidf(vocab_path, ml_base_data)

    # PyTorchで学習に使用できる形式へ変換
    x = torch.tensor(tf_idf, dtype=torch.float32)
    t = torch.tensor(labels, dtype=torch.int64)
    # print(type(x), x.dtype, type(t), t.dtype)

    # 入力値と目標値をまとめて、ひとつのオブジェクトdatasetに変換
    dataset = TensorDataset(x, t)

    # train : val : text = 60% : 20% : 20%
    n_train = int(len(dataset) * 0.6)
    n_val = int(len(dataset) * 0.2)
    n_test = int(len(dataset)) - n_train - n_val

    # ランダムに分割を行うため、シードを固定して再現性を確保
    torch.manual_seed(0)

    # データセットの分割
    train, val, test = random_split(dataset, [n_train, n_val, n_test])

    # バッチサイズ
    batch_size = 200

    # Data Loadkerを用意
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size)

    # 詳細設定
    n_input = len(dictionary)
    n_hidden = round(len(dictionary) / 2)
    # n_outputは、labelの種類の数を指定
    n_output = 2
    # 学習回数
    epoch = 5

    # 学習の実行
    pl.seed_everything(0)
    net = Net(n_input, n_hidden, n_output)
    # 学習ループの設定
    trainer = pl.Trainer(max_epochs=epoch)
    # 学習の実行
    trainer.fit(net, train_loader, val_loader)

    # テストデータで検証
    results = trainer.test(dataloaders=test_loader)
    print(results)

    # モデルの保存
    # パラメータのみ
    # torch.save(net.state_dict(), output_ml_result_dir)
    # モデル全体
    torch.save(net, output_ml_result_dir)
