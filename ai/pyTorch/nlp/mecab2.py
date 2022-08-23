import MeCab
import warnings
warnings.filterwarnings('ignore')


mecab = MeCab.Tagger('-Owakati')

res = mecab.parse('こんにちは。私の名前はキカガクです。')
print(res)

# split() でリスト型に変換
res = res.split(' ')
print(res)
# エスケープシーケンスを取り除く
res = res[:-1]
print(res)


def tokenize(t):
    print(t)
    return mecab.parse(t).split(' ')[:-1]


import pandas as pd

df = pd.DataFrame({'text':    ['こんにちは。私の名前はキカガクです。',  'こんばんは。私の名前はキカガクです。'],
                    'label': ['0', '0']})
# print(df.values)
# print(type(df.values))

# 分かち書きを実行
# df['text']は、Seriesになる
df['text'] = df['text'].apply(tokenize)
print(df.head)

from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T


# 辞書作成 <unk>や<pad>も辞書に含める
# textの辞書(text_vocab)
text_vocab = build_vocab_from_iterator(df['text'], specials=['<unk>', '<pad>'])
text_vocab.set_default_index(text_vocab['<unk>'])
print(text_vocab.get_stoi())
# labelの辞書(label_vocab)
label_vocab = build_vocab_from_iterator(df['label'])
print(label_vocab.get_stoi())

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
    texts = text_transform([text for (text, label) in batch])
    labels = label_transform([label for (text, label) in batch])
    return texts, labels


import numpy as np
from torch.utils.data import DataLoader

# DataLoaderの定義
batch_size = 1

# 1. 単語が辞書内のIDに置き換わっている（インデックス化)
# 2. 各文章の末尾に1が挿入されてパディングできている
# 3. 各ミニバッチのサイズは、[num_seq, batch]となる（文章に含む単語の数：num_seq、バッチサイズ：batch)
data_loader = DataLoader(df.values, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
print(f'data_loader : {data_loader}')

for i, (texts, labels) in enumerate(data_loader):
    print(i)
    for text, label in zip(texts, labels):
        print(f'text : {text}')
        print(f'label : {label}')

batch = next(iter(data_loader))

# batchはタブル
print(f'テキスト：{batch[0]}')
print(f'ラベル：{batch[1]}')

n_input = len(text_vocab)
n_embed = 5

# Embedding層を定義
# ネットワークの定義と学習
import torch.nn as nn

# padding_idx=1でマスキング処理をする
embed_enc = nn. Embedding(n_input, n_embed, padding_idx=1)
x_embeded = embed_enc(batch[0])

print(x_embeded)
print(batch[0].shape)
print(x_embeded.shape)

# LSTM層の定義
n_hidden = 200
n_layers = 3

lstm_enc = nn.LSTM(n_embed, n_hidden, n_layers)
# hが出力
x_lstm, (h, c) = lstm_enc(x_embeded)

# サイズの変化を確認
print(x_embeded.shape)
print(x_lstm.shape, h.shape, c.shape)
