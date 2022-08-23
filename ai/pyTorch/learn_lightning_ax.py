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
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class Net(pl.LightningModule):

    def __init__(self, n_mid=4, lr=0.01):
        super().__init__()

        self.fc1 = nn.Linear(4, n_mid)
        self.fc2 = nn.Linear(n_mid, 3)
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    # 順伝播
    def forward(self, x):
        h = self.fc1(x)
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

    # 最適化手法の設定
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


# GPUを含めた乱数のシードを固定
pl.seed_everything(0)
net = Net()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(net, train_loader, val_loader)

# ハイパーパラメータの最適化を行う関数には、検証データに対する最終的な loss か accuracy のどちらかを用います。
# 今回は accuracy を採用し、この指標を最大化するように最適化を行います。

# 最後のエポックに対する検証データの accuracy
print(trainer.callback_metrics['val_acc'])

# Axによるハイパーパラメータの調整
import ax

parameters = [
    {'name': 'n_mid', 'type': 'range', 'bounds': [1, 100]},  # 1 ~ 100 の整数
    {'name': 'lr', 'type': 'range', 'bounds': [0.001, 0.1]}  # 0.001 から 0.1 までの実数
]

def evaluation_function (parameters):

    # パラメータの取得
    n_mid = parameters.get('n_mid')
    lr = parameters.get('lr')

    # ネットワークの定義と学習
    torch.manual_seed(0)
    net = Net(n_mid, lr=lr)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(net, train_loader, val_loader)

    # 検証データに対する accuracy を目的関数の値として返す
    val_acc = float(trainer.callback_metrics['val_acc'])

    # 確認のためにテストデータのaccuracyも算出
    trainer.test(dataloaders=test_loader)
    test_acc = trainer.callback_metrics['test_acc']

    # 各試行の選択されたハイパーパラメータの値と結果を表示
    print('n_mid:', n_mid)
    print('lr:', lr)
    print('val_acc', val_acc)
    print('test_acc', test_acc)

    return val_acc


# 最適化の実行
# デフォルトがminimize=False であり、今回は最大化であるため、デフォルトでOK
# minimize(ミニマイズ)=Trueの場合は、最小化
results = ax.optimize(parameters, evaluation_function, random_seed=0)

# 結果を取得
best_parameters, values, experiment, best_model = results
# 得られた最適パラメータ
print(best_parameters)
# 最適なパラメータにおける目的関数の値
print(values)
