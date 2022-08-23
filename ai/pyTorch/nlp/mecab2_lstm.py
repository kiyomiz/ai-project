import MeCab
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import warnings

warnings.filterwarnings('ignore')

path = 'data/novel/train.csv'
mecab = MeCab.Tagger('-Owakati')


def tokenize(x):
    return mecab.parse(x).split(' ')[:-1]


df = pd.read_csv(path, header=None, names=['text', 'label'])

# 分かち書きを実行
# df['text']は、Seriesになる
df['text'] = df['text'].apply(tokenize)
# labelは文字列に変換
df['label'] = df['label'].astype(str)

# 辞書作成 <unk>や<pad>も辞書に含める
# 文字列のみ
# textの辞書(text_vocab)
text_vocab = build_vocab_from_iterator(df['text'], specials=['<unk>', '<pad>'])
text_vocab.set_default_index(text_vocab['<unk>'])
# print(text_vocab.get_stoi())
# labelの辞書(label_vocab)
label_vocab = build_vocab_from_iterator(df['label'])
# print(label_vocab.get_stoi())


# transform生成
# テキストは、辞書による変換(数値化)とパディング、Tensor型への変換を行います。パディングは、ミニバッチごとに系列長を統一するため不足部分がパディングされます。
# ラベルは、辞書による変換(数値化)とTensor型への変換を行います。
text_transform = T.Sequential(
    T.VocabTransform(text_vocab),
    T.ToTensor(padding_value=text_vocab['<pad>'])
)
label_transform = T.Sequential(
    T.VocabTransform(label_vocab),
    T.ToTensor()
)

# ミニバッチ時のデータ変換関数
# リスト内包表記
# x = [リストの要素を計算する式 for 計算で使用する変数 in 反復可能オブジェクト]
def collate_batch(batch):
    texts = text_transform([text for (text, label) in batch]).T
    labels = label_transform([label for (text, label) in batch])
    return texts, labels


class Net(pl.LightningModule):

    def __init__(self, n_input, n_embed, n_hidden, n_layers, n_output):
        super(Net, self).__init__()
        self.embed = nn.Embedding(n_input, n_embed, padding_idx=1)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers)
        self.fc = nn.Linear(n_hidden, n_output)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.embed(x)
        x, (h, c) = self.lstm(x)
        x = self.fc(h[-1])
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


# df.valuesをtrainとvalidとtestに分ける必要がある
# train : val : text = 60% : 20% : 20%
df_train = df.sample(frac=0.6, random_state=0)
# print(pd.merge(df, df_train, how='left', left_index=True, right_index=True).head(1))
df_temp = pd.merge(df, df_train, how='left', left_index=True, right_index=True)
df_temp = df_temp[df_temp["text_y"].isna()][['text_x', 'label_x']]
df_temp.rename(columns={'text_x': 'text', 'label_x': 'label'}, inplace=True)
df_val = df_temp.sample(frac=0.5, random_state=0)
df_temp2 = pd.merge(df_temp, df_val, how='left', left_index=True, right_index=True)
df_temp2 = df_temp2[df_temp2["text_y"].isna()][['text_x', 'label_x']]
df_temp2.rename(columns={'text_x': 'text', 'label_x': 'label'}, inplace=True)
df_test = df_temp2

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)


# バッチサイズ
batch_size = 10

# Data Loadkerを用意
train_loader = DataLoader(df_train.values, batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(df_val.values, batch_size, collate_fn=collate_batch)
test_loader = DataLoader(df_test.values, batch_size, collate_fn=collate_batch)

# 詳細設定
n_input = len(text_vocab)
n_embed = 100
n_hidden = 100
n_layers = 3
n_output = 10

# 学習の実行
pl.seed_everything(0)
net = Net(n_input, n_embed, n_hidden, n_layers, n_output)
trainer = pl.Trainer(max_epochs=3)
trainer.fit(net, train_loader, val_loader)

# テストデータに対する検証
results = trainer.test(dataloaders=test_loader)
print(results)
