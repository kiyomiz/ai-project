import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split
from torch.utils.data import DataLoader

import warnings

# 警告の非表示
warnings.filterwarnings('ignore')


# iris(アイアリス) ： アヤメ
from sklearn.datasets import load_iris

# Irisデータセットの読み込み
x, t = load_iris(return_X_y=True)

# PyTorchでは torch.Tensor形式が標準
# 分類の問題は、int64
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

# PyTorchでは、学習時に使用するデータxとtを1つのオプジェクトdatasetにまとめます。
from torch.utils.data import TensorDataset

# 入力値と目的値をdatasetにまとめる
dataset = TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val: test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = int(len(dataset) * 0.2)

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)
# データセットの分割
train, val, test = random_split(dataset, [n_train, n_val, n_test])

# バッチサイズの決定
# 1バッチのデータ数
batch_size = 10

# shuffleはデフォルトでfalseのため、学習データのみTrueに指定
train_loader = DataLoader(train, batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size)
test_loader = DataLoader(test, batch_size)

# 今回は、入力変数が4、分類するクラス数が3
# ハイパーパラメータ（中間層のノード数や層の数）
# fc1 : input : 4 -> output : 4
# fc2 : input : 4 -> output : 3

# 順伝播の計算の流れ
# 線形変換(fc1) -> 非線形変換(ReLU) -> 線形変換(fc2) -> 非線形変換(Softmax)
class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    # 順伝播 (オーバーライド)
    # 損失関数にSoftmax関数が含まれるので、順伝播では記述しない
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# エポックの数
max_epoch = 1

# 乱数のシードを固定して再現性を確保
torch.manual_seed(0)

# インスタンス化
net = Net()

# デバイスでGPUが使用可能な場合は、GPUを使用し、使用不可の場合はCPUを使用する。
# 指定デバイスで学習するには、モデルとデータをデバイスに転送する必要がある。
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 指定したデバイスへのモデルの転送
net = net.to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()
# 最適化手法（パラメータの更新方法)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(max_epoch):

    for batch in train_loader:

        # バッチサイズ分のサンプルを抽出
        x, t = batch
        # 学習に使用するデバイスへデータの転送
        x = x.to(device)
        t = t.to(device)
        # パラメータの勾配を初期化
        # loss.backward()では求めた勾配情報が各パラメータのgradに代入されるのではなく、現状の勾配情報に加算される
        # これは、累積勾配を用いた方が良い場合のための仕様
        # 今回は累積勾配は利用しないので、バッチごとに勾配の初期化を行う。
        optimizer.zero_grad()

        # 予測値の算出
        y = net(x)
        # 目標値と予測値から損失関数の値を算出
        loss = criterion(y, t)
        # 損失関数の値を表示して確認
        # .item(): tensor.Tensor => float
        print('loss:', loss.item())

        # 正解率の算出
        y_label = torch.argmax(y, dim=1)
        acc = (y_label == t).sum() * 1.0 / len(t)
        print('accuracy:', acc)

        # 各パラメータの勾配を算出
        loss.backward()
        # 勾配の情報を用いたパラメータの更新
        optimizer.step()


# 学習後の正解率を検証
# 正解率の計算
def calc_acc(data_loader):
    # 学習を行わないため、勾配情報が必要なし設定
    with torch.no_grad():
        # 各バッチの結果格納用
        accs = []
        for batch in data_loader:
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = net(x)

            y_label = torch.argmax(y, dim=1)
            acc = (y_label == t).sum() * 1.0 /len(t)
            accs.append(acc)

    # 全体の平均を算出
    avg_acc = torch.tensor(accs).mean()

    return avg_acc

# 検証データで確認
val_acc = calc_acc(val_loader)
print(val_acc)

# テストデータで確認
test_acc = calc_acc(test_loader)
print(test_acc)
