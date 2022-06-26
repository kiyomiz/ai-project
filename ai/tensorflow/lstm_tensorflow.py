import pandas as pd
from janome.tokenizer import Tokenizer
# 正規表現ライブラリ
import re

# リスト
utterance_txt = ['こんにちは', '分からない', '昼ごはんは何を食べましたか', 'ええ', 'いえ、どうもしませんよ', '行楽シーズンになります']
system_txt = ['こんー', 'そっか', 'ごはんはあったかいです', 'どうした', '行楽シーズンに行きます', '行楽シーズンに入りますか？']
label = ['0', '0', '0', '1', '1', '0']
# タプル
# label = '0', '', '', '', '', ''

print(utterance_txt)
print(system_txt)
print(label)

df = pd.DataFrame({
    'utterance_txt': utterance_txt,
    'system_txt': system_txt,
    'label': label
    })

print(df)
print(df['label'].value_counts())


def parse(utterance_txt):
    '''
    分かち書きを行って形態素に分解する
    '''
    # Tokenizerクラスのオブジェクトを生成
    t = Tokenizer()
    # 形態素を一時保存するリスト
    separation_tmp = []
    # 形態素に分解
    for row in utterance_txt:
        # リストから発話テキストの部分を抽出して形態素解析を実行
        tokens = t.tokenize(row)
        # 形態素の見出しの部分を取得してseparation_tmpに追加
        separation_tmp.append(
            [token.surface for token in tokens if(
                not re.match('記号', token.part_of_speech)  # 記号を除外
                and not re.match('助詞', token.part_of_speech)  # 助詞を除外
                and not re.match('助動詞', token.part_of_speech) # 助動詞は除外
                )
            ]
        )

    # 空の要素があれば取り除く
    while separation_tmp.count('') > 0:
        separation_tmp.remove('')

    print(separation_tmp)

    return separation_tmp


# 人間の発話を形態素に分解する
df['utterance_token'] = parse(df['utterance_txt'])
# システムの応答を形態素に分解する
df['system_token'] = parse(df['system_txt'])
# .apply(lambda x : len(x))とapply(len)は同じ意味
df['u_token_len'] = df['utterance_token'].apply(len)
df['s_token_len'] = df['system_token'].apply(len)

print(df)

# カウント処理のためのライブラリ
from collections import Counter
# イテレーションのためのライブラリ
import itertools


# {単語：出現回数}の辞書を作成
def makedictionary(data):
    return Counter(itertools.chain(* data))


def update_word_dictionary(worddic):
    word_list = []
    word_dic = {}
    # most_common()で出現回数順に要素を取得しword_listに追加
    for w in worddic.most_common():
        word_list.append(w[0])

    # 頻度順に並べた単語をキーに、1から始まる連番を値に設定
    for i, word in enumerate(word_list, start=1):
        word_dic.update({word: i})

    return word_dic


# 単語を出現頻度の数値に置き換える関数
def bagOfWords(word_dic, token):
    return [[word_dic[word] for word in sp] for sp in token]


utter_word_frequency = makedictionary(df['utterance_token'])
print(utter_word_frequency)
system_word_frequency = makedictionary(df['system_token'])
print(system_word_frequency)

utter_word_dic = update_word_dictionary(utter_word_frequency)
print(utter_word_dic)
system_word_dic = update_word_dictionary(system_word_frequency)
print(system_word_dic)

utter_dic_size = len(utter_word_dic)
system_dic_size = len(system_word_dic)
print(utter_dic_size)
print(system_dic_size)

train_utter = bagOfWords(utter_word_dic, df['utterance_token'])
train_system = bagOfWords(system_word_dic, df['system_token'])
print(train_utter)
print(train_system)

UTTER_MAX_SIZE = len(sorted(train_utter, key=len, reverse=True)[0])
SYSTEM_MAX_SIZE = len(sorted(train_system, key=len, reverse=True)[0])
print(UTTER_MAX_SIZE)
print(SYSTEM_MAX_SIZE)



from tensorflow.keras.preprocessing import sequence


def padding_sequences(data, max_len):
    ''' 最長のサイズになるまでゼロを埋め込む
    :param data: 操作対象の配列
    :param max_len: 配列のサイズ
    :return:
    '''
    return sequence.pad_sequences(
        data, max_len, padding='post', value=0.0)


train_U = padding_sequences(train_utter, UTTER_MAX_SIZE)
print(train_U.shape)
print(train_U)

train_S = padding_sequences(train_system, SYSTEM_MAX_SIZE)
print(train_S.shape)
print(train_S)

# 多入力の複合型モデル
# 人間の発話
# システムの応答
# 人間の発話の単語数
# システムの応答の単語数
# の4系統の層を配置し、それぞれEmbedding層に入力して埋め込み処理を行う。
# 人間の発話とシステムの応答について、3層構造の128ユニットのGRUに入力します。
# 以上の4系統のEmbedding層からの出力になりますが、これを全結合層で結合して1系統にまとめます
# 以降、512ユニット、256ユニット、128ユニットの全結合型の層を経て、2ユニットの出力層から出力するようにします。
# 会話が「破綻している」「破綻していない」の二値分類ですが、ユニットを2個配置してマルチクラス分類のかたちにしました。
# 単語      -> input -> Embedding        ->  -┓
# 発話・応答 -> input -> Embedding -> GRU -> 全結合 -> Dropout -> output

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, GRU, Embedding, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

## ---- 入力 ----
# ユニット（ニューロン）
# 1単語=1ユニット、1文で1学習
# 人間の発話：ユニット数は単語配列(一文を形態素解析し単語に分解した配列)の最長サイズと同じ
utterance = Input(shape=(UTTER_MAX_SIZE,), name='utterance')
# システムの応答：ユニット数は単語配列の最長サイズと同じ
system = Input(shape=(SYSTEM_MAX_SIZE,), name='system')
# 人間の発話の単語数：ユニット数は１
u_token_len = Input(shape=[1], name='u_token_len')
# システムの応答の単語数：ユニット数は１
s_token_len = Input(shape=[1], name='s_token_len')

## ---- Embedding層 ----
# 人間の発話：入力は単語の総数＋１００、出力の次元数は１２８
emb_utterance = Embedding(
    input_dim=utter_dic_size+100,  # 発話の単語数＋１００
    output_dim=128                 # 出力の次元数はRecurrent層のユニット数
)(utterance)
# システムの応答：入力は単語の総数＋１００、出力の次元数は１２８
emb_system = Embedding(
    input_dim=system_dic_size+100,  # 応答の単語数
    output_dim=128                  # 出力の次元数はRecurrent層のユニット数
)(system)
# 人間の発話の単語数のEmbedding
emb_u_len = Embedding(
    input_dim=UTTER_MAX_SIZE+1,  # 入力の次元数は発話の形態素数の最大値＋１
    output_dim=5                 # 出力は５
)(u_token_len)
# システムの発話の単語数のEmbedding
emb_s_len = Embedding(
    input_dim=SYSTEM_MAX_SIZE+1,  # 入力の次元数は発話の形態素数の最大値＋１
    output_dim=5                 # 出力は５
)(s_token_len)

## ---- Recurrent層 ----
# GRUは、LSTMを高速化して改良したもの
# 人間の発話：GRUユニット×１２８×３段
rnn_layer1_1 = GRU(128, return_sequences=True)(emb_utterance)
rnn_layer1_2 = GRU(128, return_sequences=True)(rnn_layer1_1)
rnn_layer1_3 = GRU(128, return_sequences=False)(rnn_layer1_2)
# システムの発話：GRUユニット×１２８×３段
rnn_layer2_1 = GRU(128, return_sequences=True)(emb_utterance)
rnn_layer2_2 = GRU(128, return_sequences=True)(rnn_layer2_1)
rnn_layer2_3 = GRU(128, return_sequences=False)(rnn_layer2_2)

## ---- 全結合層 ----
main_1 = concatenate([
    Flatten()(emb_u_len),  # 人間の発話の単語数のEmbedding
    Flatten()(emb_s_len),  # システムの応答の単語数のEmbedding
    rnn_layer1_3,          # 人間の発話のGRUユニット
    rnn_layer2_3           # システムの発話のGRUユニット
])

## ---- ５１２、２５６、１２８ユニットの層を追加 ----
main_1 = Dropout(0.2)(
    Dense(512, kernel_initializer='normal', activation='relu')(main_1)
)
main_1 = Dropout(0.2)(
    Dense(256, kernel_initializer='normal', activation='relu')(main_1)
)
main_1 = Dropout(0.2)(
    Dense(128, kernel_initializer='normal', activation='relu')(main_1)
)

## ---- 出力層（２ユニット） ----
output = Dense(units=2,             # 出力層のニューロン数＝２
               activation='softmax' # 活性化はソフトマックス関数
               )(main_1)

# Modelオブジェクトの生成
model = Model(
    # 入力層はマルチ入力モデルなのでリストにする
    inputs=[utterance, system,
            u_token_len, s_token_len
            ],
    # 出力層
    outputs=output
)

# Sequentialオブジェクトをコンパイル
model.compile(
    loss='categorical_crossentropy', # 誤差関数はクロスエントロピー
    optimizer=Adam(), # Adamオプティマイザー
    metrics=['accuracy']  # 学習評価として正解率を指定
)

print(model.summary())

import numpy as np
from tensorflow.keras.utils import to_categorical

trainX = {
    # 人間の発話
    'utterance': train_U,
    # システムの応答
    'system': train_S,
    # 人間の発話の形態素の数(int)
    'u_token_len': np.array(df[['u_token_len']]),
    # システムの応答の形態素の数(int)
    's_token_len': np.array(df[['s_token_len']])
}

# 正解ラベルをOne-Hot表現にする
trainY = to_categorical(df['label'], 2)

# 学習を実行する
# エポック：学習回数
# バッチサイズ：データの分割サイズ
# 1,000件のデータセットを200件ずつのサブセットに分ける場合、バッチサイズは200
import math
from tensorflow.keras.callbacks import LearningRateScheduler

batch_size = 32  # ミニバッチのサイズ
lr_min = 0.0001  # 最小学習率
lr_max = 0.001   # 最大学習率

# 学習率のスケジューリング
# decay:ディケイ:減衰
def step_decay(epoch):
    initial_lrate = 0.001 # 学習率の初期値
    drop = 0.5  # 減数率は50%
    epochs_drop = 10.0   # 10エポック毎に減衰する
    lrate = initial_lrate * math.pow(
        drop,
        math.floor(epoch/epochs_drop)
    )
    return lrate


# 学習率のコールバック
lrate = LearningRateScheduler(step_decay)

# エポック数
epoch = 100

# 学習を開始
history = model.fit(trainX, trainY,         # 訓練データ、正解ラベル
                    batch_size=batch_size,  # ミニバッチのサイズ
                    epochs=epoch,           # 学習回数
                    verbose=1,              # 学習の進捗状況を出力する
                    validation_split=0.2,   # データの20%を検証データにする
                    shuffle=True,           # 検証データ抽出後にシャッフル
                    callbacks=[lrate]
                    )

