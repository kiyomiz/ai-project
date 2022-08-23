import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 警告の非表示
warnings.filterwarnings('ignore')

# 乱数のシードを固定
torch.manual_seed(1)

# 入力値の定義
x = torch.tensor([1, 2, 3], dtype=torch.float32)
# 全結合層
fc1 = nn.Linear(3, 2)
fc2 = nn.Linear(2, 1)
# 線形変換
u1 = fc1(x)
# 非線形変換
h1 = F.relu(u1)
# 線形変換
y = fc2(h1)
print(y)

# 目標値(今回は1とする)
# 回帰：float32
# 分類：int64
t = torch.tensor([1], dtype=torch.float32)
# 平均二乗誤差
loss = F.mse_loss(t, y)
print(loss)

# 補足
# PyTorchではネットワーク内での計算を行う際に、torch.nn (nn)を用いる場合と、torch.nn.functinal (F)を用いる場合の2通りがある
# nn : パラメータを持つ、weightやbias
# F  : パラメータを持たない、ReLU関数や平均二乗誤差

