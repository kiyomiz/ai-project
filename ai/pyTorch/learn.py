# Step1 : データセットを準備
# Step2 : ネットワークを定義
# Step3 : 損失関数を選択
# Step4 : 最適化手法の選択
# Step5 : ネットワークを学習

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 警告の非表示
warnings.filterwarnings('ignore')


# iris(アイアリス) ： アヤメ
from sklearn.datasets import load_iris

# Irisデータセットの読み込み
x, t = load_iris(return_X_y=True)
# 確認 行数,列数
print(x.shape)
print(t.shape)
# typeの確認
print(type(x))
print(type(t))

# PyTorchでは torch.Tensor形式が標準
# 分類の問題は、int64
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

# PyTorchでは、学習時に使用するデータxとtを1つのオプジェクトdatasetにまとめます。
from torch.utils.data import TensorDataset

# 入力値と目的値をdatasetにまとめる
dataset = TensorDataset(x, t)
print(type(dataset))
# (入力変数、教師データ)のようにタブルで格納されている
print(dataset[0])
print(type(dataset[0]))
# 1サンプル目の入力値
print(dataset[0][0])
# 1サンプル目の目標値
print(dataset[0][1])
# サンプル数はlenで取得可能
print(len(dataset))

# 各データセットのサンプル数を決定
# train : val: test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = int(len(dataset) * 0.2)

from torch.utils.data import random_split

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)
# データセットの分割
train, val, test = random_split(dataset, [n_train, n_val, n_test])
# サンプル数の確認
print(len(train), len(val), len(test))

# バッチサイズの決定
# 1バッチのデータ数
batch_size = 10

# ミニバッチ学習を行う場合には、各データセットからバッチサイズ分のサンプルを取得する必要がある
# 学習時にはランダムにシャップルをして抽出する必要がある。
# PyTorchにはランダムシャッフルして抽出するなので機能を実現するために torch.utils.data.DataLoaderがある
from torch.utils.data import DataLoader

# shuffleはデフォルトでfalseのため、学習データのみTrueに指定
train_loader = DataLoader(train, batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size)
test_loader = DataLoader(test, batch_size)

#### ネットワークを学習
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

# 乱数のシードを固定して再現性を確保
torch.manual_seed(0)
# インスタンス化
net = Net()
# ネットワークの確認
print(net)

# 損失関数
# PyTorchでは計算を高速化させるために、Softmax関数と対数変換を同時に行う関数であるF.log_softmaxが用意されている
# Softmax -> Logの計算をそれぞれ行うと速度として遅いこと、そして計算後の値が安定しないといった理由で、それぞれ別々に計算するのではなくまとめて計算する方が優れている。
# PyTorchに用意されているF.cross_entropyでは、内部の計算にF.log_softmaxが使われている
# 今回は多クラスの分類なので、損失関数としてクロスエントロピーを使う
criterion = nn.CrossEntropyLoss()
# 最適化手法（パラメータの更新方法)
# 今回は確率的勾配降下法(SGD)を選択
# 引数としてネットワークのパラメータを渡す必要があり、パラメータの取得にはnet.parameters()を用いる
# もう一つの引数は、学習係数(lr : learning rate)
# 学習係数：一回の処理で進む学習の度合い。
# 学習係数が大きいほど、処理の回数が少なく済むが、学習係数が大きすぎると目的値を通り越してしまう。
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

def dispParam(net):
    print('全結合層fc1の重み')
    print(net.fc1.weight)
    print('全結合層fc1のバイアス')
    print(net.fc1.bias)
    print('全結合層fc2の重み')
    print(net.fc2.weight)
    print('全結合層fc2のバイアス')
    print(net.fc2.bias)

def dispGrad(net):
    print('全結合層fc1の重みに関する勾配')
    print(net.fc1.weight.grad)
    print('全結合層fc1のバイアスに関する勾配')
    print(net.fc1.bias.grad)
    print('全結合層fc2の重みに関する勾配')
    print(net.fc2.weight.grad)
    print('全結合層fc2のバイアスに関する勾配')
    print(net.fc2.bias.grad)


# バッチサイズ分のサンプルの抽出
batch = next(iter(train_loader))
# 入力値と目標値に分割
x, t = batch
# 予測値の算出
# y = net.forward(x)
# callメソッドを用いたforwardの計算(推奨)
y = net(x)
# 損失関数の計算
# criterionのcallメソッドを利用
# y : 予測値、t : 目標値
loss = criterion(y, t)

print('### 逆伝播前の勾配情報')
dispGrad(net)
# 勾配の算出 逆伝播
loss.backward()
print('### 逆伝播後の勾配情報')
dispGrad(net)

print('### 更新前 パラメータの表示')
dispParam(net)
# 勾配の情報を用いたパラメータの更新
optimizer.step()
print('### 更新後パラメータの表示')
dispParam(net)

# dim=1で行ごとの最大値に対する要素番号を取得(dim=0は列ごと)
y_label = torch.argmax(y, dim=1)
print(y_label)
# 目標値
print(t)
# 値が一致しているか確認
print(y_label == t)
# 値がTrueとなる個数の総和
print((y_label == t).sum())
# 1.0を乗じて、int型をfloat型へ変換
print((y_label == t).sum() * 1.0)
# 正解率
print((y_label == t).sum() * 1.0 / len(t))


