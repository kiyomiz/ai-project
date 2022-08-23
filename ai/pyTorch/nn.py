import torch

print(torch.__version__)

import warnings

# 警告の非表示
warnings.filterwarnings('ignore')

import torch.nn as nn

# 乱数のシードを固定して再現性を確保
# 固定しないとパラメータ（重みとバイアス）の値がランダムに初期化される
torch.manual_seed(1)

# inが3、outが2の全結合層
fc = nn.Linear(3, 2)
print(fc.weight)
print(fc.bias)

# リストからPyTorchのTensorへ変換
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x)
# データ型の表示
print(x.dtype)
# 線形変換 u = wx + b (w : 重み、b : バイアス)
# 線形変換は、nn.Linearのcallメソッドとして定義されている
u = fc(x)
print(u)

# 非線形変換
import torch.nn.functional as F

# ReLu関数
h = F.relu(u)
print(u)

