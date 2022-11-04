import os
import MeCab
import pandas as pd

from gensim import corpora, matutils

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from datetime import datetime
from dateutil.relativedelta import relativedelta
from ai.ai.v2.analysis.common.date_utils import get_last_date


def get_tfidf(vocab_path, ml_base_data):
    labels = [int(l) for l in ml_base_data['label']]

    # 名詞を抽出
    word_collect = get_word_collect(ml_base_data['text'])

    # 辞書の読込み
    dictionary = corpora.Dictionary.load_from_text(vocab_path)
    # dictionary = corpora.Dictionary.load(vocab_path)

    # 辞書を用いて文章をBoWに変換する
    # 辞書内の全単語数を取得
    n_words = len(dictionary)
    # BoW による特徴ベクトルの作成
    x = []
    for nouns in word_collect:
        bow_id = dictionary.doc2bow(nouns)
        bow = matutils.corpus2dense([bow_id], n_words).T[0]
        x.append(bow)

    # インストンス化
    tfidf = TfidfTransformer()

    # precision は少数以下の桁数の指定
    np.set_printoptions(precision=4)

    # TF-IDFによる特徴ベクトルの生成
    tf_idf = tfidf.fit_transform(x).toarray()

    # 変換前: BoW
    # print(type(x))
    # print(x[:2])
    # 変換後: TF-IDF
    # print(type(tf_idf))
    # print(tf_idf[:2])

    return tf_idf, labels, dictionary

def get_word_collect(texts):

    # 抽出した名詞を格納
    word_collect = []
    for text in texts:
        nouns = get_nouns(text)
        word_collect.append(nouns)

    return word_collect


# 名詞抽出用関数
def get_nouns(text):
    mecab = MeCab.Tagger("-Ochasen")

    nouns = []
    res = mecab.parse(text)
    words = res.split('\n')[:-2]
    for word in words:
        part = word.split('\t')
        if '名詞' in part[3]:
            nouns.append(part[0])
    return nouns


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    mouth_period = 3
    delta_ratio = 1  # 変化率の0からの差
    favorite_retweet_flag = True
    date_start = 20220509

    tdatetime = datetime.strptime(str(date_start), '%Y%m%d')
    tdatetime = tdatetime + relativedelta(months=mouth_period - 1)
    date_end = int(get_last_date(tdatetime).strftime('%Y%m%d'))

    vocab_dir = 'vocab'
    vocab_path = f'{vocab_dir}/{date_start}-{date_end}'
    data_dir = 'data-1'
    data_path = f'../{data_dir}/ml_base_data'

    ml_base_data = pd.read_csv(data_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]

    # 正解ラベルを付与
    # 変化率(%)
    #  1:  上昇
    #  0:  下降
    ml_base_data.loc[ml_base_data['delta_ratio'] >= delta_ratio, 'label'] = 1
    ml_base_data.loc[ml_base_data['delta_ratio'] <= -1 * delta_ratio, 'label'] = 0

    # お気に入りが0、または、リツイートが0は除外
    if favorite_retweet_flag:
        p_data_twitter = ml_base_data[(ml_base_data['favorite_count'] != 0) | (ml_base_data['retweet_count'] != 0)]

    # 不要項目削除
    del ml_base_data['id']
    del ml_base_data['s_date']
    del ml_base_data['favorite_count']
    del ml_base_data['retweet_count']

    # 変動なしは除外
    ml_base_data = ml_base_data.loc[(ml_base_data['label'] == 0) | (ml_base_data['label'] == 1)]

    # 名詞を抽出
    word_collect = get_word_collect(ml_base_data['text'])

    # gensimを用いてBoW用の辞書を作成
    # 辞書を作成
    dictionary = corpora.Dictionary(word_collect)
    print(len(dictionary))

    # 出現回数でフィルタリング
    # 出現回数が少ない単語をフィルタリングすることで特徴のある単語のみに絞る
    # より重要な単語で特長抽出できたことと併せて、全体の計算量を削減する効果もあります。
    dictionary.filter_extremes(no_below=2)
    print(len(dictionary))

    # 単語->id
    print(dictionary.token2id)

    # 保存
    dictionary.save_as_text(vocab_path)
    # dictionary.save(vocab_path)
