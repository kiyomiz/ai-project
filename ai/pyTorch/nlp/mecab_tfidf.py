from glob import glob
import warnings

warnings.filterwarnings('ignore')


# 前処理関数
def preprocessing(filepath):
    with open(filepath, encoding='utf-8') as f:
        # 1行目と2行目は不要なので、3行目から読み込む
        text = f.readlines()[2:]
        # 形態素解析を行う上で、文書全体を 1 つの文字列として扱う必要があるのですが、
        # 現在は複数の文章に区切られてリスト型に格納されています。
        # そのため、すべてのテキストを結合して 1 つの文字列に成形します。
        # '間に挿入する文字列'.join([連結したい文字列のリスト])
        text = ''.join(text)
        # エスケープシーケンスを置換
        text = text.replace('\u3000', '')
        text = text.replace('\n', '')
    return text


# テキストデータの読み込み
dirs = glob('data/text/*')

texts, labels = [], []

for (label, dir) in enumerate(dirs):
    # 各ディレクトリ内のファイルパスを全取得
    filepaths = glob(f'{dir}/*.txt')

    for filepath in filepaths:
        # テキストデータ取得
        text = preprocessing(filepath)
        texts.append(text)

        # 正解ラベル作成
        labels.append(label)


import MeCab

mecab = MeCab.Tagger("-Ochasen")


# 名詞抽出用関数
def get_nouns(text):
    nouns = []
    res = mecab.parse(text)
    words = res.split('\n')[:-2]
    for word in words:
        part = word.split('\t')
        if '名詞' in part[3]:
            nouns.append(part[0])
    return nouns


# 抽出した名詞を格納
word_collect = []
for text in texts:
    nouns = get_nouns(text)
    word_collect.append(nouns)

# gensimを用いてBoW用の辞書を作成
from gensim import corpora, matutils

# 辞書を作成
dictionary = corpora.Dictionary(word_collect)
print(len(dictionary))

# 出現回数でフィルタリング
# 出現回数が少ない単語をフィルタリングすることで特徴のある単語のみに絞る
# より重要な単語で特長抽出できたことと併せて、全体の計算量を削減する効果もあります。
dictionary.filter_extremes(no_below=2)
print(len(dictionary))

# 後から使えるように保存
# dictionary.save('livedoordic.txt')

# 辞書を用いて文章をBoWに変換する
# 辞書内の全単語数を取得
n_words = len(dictionary)
# BoW による特徴ベクトルの作成
x = []
for nouns in word_collect:
    bow_id = dictionary.doc2bow(nouns)
    bow = matutils.corpus2dense([bow_id], n_words).T[0]
    x.append(bow)


import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

# インストンス化
tfidf = TfidfTransformer()

# precision は少数以下の桁数の指定
np.set_printoptions(precision=4)

# TF-IDFによる特徴ベクトルの生成
tf_idf = tfidf.fit_transform(x).toarray()

# 変換前: BoW
print(type(x))
print(x[:2])
# 変換後: TF-IDF
print(type(tf_idf))
print(tf_idf[:2])

# 学習の準備
# PyTorch で取り扱えるデータセットの形式への変換
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# PyTorchで学習に使用できる形式へ変換
x = torch.tensor(tf_idf, dtype=torch.float32)
t = torch.tensor(labels, dtype=torch.int64)
print(type(x), x.dtype, type(t), t.dtype)

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
batch_size = 5

# Data Loadkerを用意
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

# ネットワークの定義と学習
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchmetrics

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(540)
        self.fc1 = nn.Linear(540, 270)
        self.fc2 = nn.Linear(270, 4)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        h = self.bn(x)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
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

### 学習の実行
pl.seed_everything(0)
net = Net()
# 学習ループの設定
trainer = pl.Trainer(max_epochs=30)
# 学習の実行
trainer.fit(net, train_loader, val_loader)

# テストデータで検証
results = trainer.test(dataloaders=test_loader)
print(results)
