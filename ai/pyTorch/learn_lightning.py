import torch
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset, random_split

import warnings

warnings.filterwarnings('ignore')

# Irisデータセットの読み込み
x, t = load_iris(return_X_y=True)

# Pytorchで学習に使用できる形式へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

# 入力値と目標値をまとめて、1つのオブジェクトdatasetに変換
dataset = TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = random_split(dataset, [n_train, n_val, n_test])

# バッチサイズの定義
batch_size = 32

# Data Loaderを用意
# shuffleはデフォルトでFalseのため、訓練データのみTrueに指定
# drop_lastはデフォルトでFalse、バッチサイズで割り切れない場合、Trueだと切り捨てる。Falseだと少ないまま使う
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(4)  # バッチ正規化
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    # 順伝播
    def forward(self, x):
        h = self.bn(x)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        return h

    # 学習ループのミニバッチ抽出後の予測と損失算出の処理
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        # ログ出力
        # on_stepをTrueにすると各イテレーションごとの値が記録される (default : True)
        # on_epochをTrueにすると各エポックごとの値が記録される (default : False)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y, t), on_step=False, on_epoch=True)
        return loss

    # 検証データに対する処理
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc(y, t), on_step=False, on_epoch=True)
        return loss

    # テストデータに対する処理
    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', self.test_acc(y, t), on_step=False, on_epoch=True)
        return loss

    # 最適化手法の設定
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


# GPUを含めた乱数のシードを固定
pl.seed_everything(0)
net = Net()

# 学習を行う Trainerの設定
# 引数名                デフォルトの値   説明
# max_epochs           1000           学習時の最大エポック数
# min_epochs           1              学習時の最小エポック数
# gpus                 None           使用するGPUの数
# distributed_backend  None           分散学習の方法
trainer = pl.Trainer(max_epochs=5)

# 学習を実行
trainer.fit(net, train_loader, val_loader)

# テストデータで検証
results = trainer.test(dataloaders=test_loader)
