import os
import pandas as pd
import MeCab
# import glob
import oseti
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np


def analyze(s):
    try:
        # print(type(s))
        # sは、pandas.core.series.Series
        analyzer = oseti.Analyzer()
        ans = analyzer.analyze(s['full_text'])
        ans_sum = sum(ans)
        return 1 if ans_sum > 0 else -1 if ans_sum < 0 else 0
    except:
        print("analyze error")


# 引数1 : 説明変数(twitterの情報)の日付
# 引数2 : 目的変数(日経平均株価の増減)と説明変数(twitterの情報)の日付差
def data_processing(s_date, delta):
    path = f'data/{s_date}'
    if not os.path.isfile(path):
        return None

    path_prie = f'data/price'
    if not os.path.isfile(path_prie):
        return None

    # twitter
    # twitterのデータをpandasに読み込む(ヘッダーなしCSV)
    p_data_twitter = pd.read_csv(path, header=None, names=['id', 'screen_name', 'jst_time', 'full_text', 'favorite_count', 'retweet_count'])

    # 感情分析 axis=1 : analyzeを各行に適用、指定なしは、列に適用
    p_data_twitter.loc[:, 'sentiment'] = p_data_twitter[['full_text']].apply(analyze, axis=1)

    # 集計
    # 日付
    # ツイート数 補足：dtype: int64
    tweet_num = p_data_twitter['id'].count()
    # 感情分析の集計
    sentiment = p_data_twitter['sentiment'].sum()
    # お気に入り件数
    favorite_count = p_data_twitter['favorite_count'].sum()
    # リツイート件数
    retweet_count = p_data_twitter['retweet_count'].sum()
    # 指定日（過去）
    date_time = dt.strptime(s_date, '%Y%m%d')
    s_delta_date = (date_time + relativedelta(days=delta)).strftime('%Y%m%d')
    # twitterデータ作成
    p_data_twitter = pd.DataFrame({'s_date': [s_date],
                                   'tweet_num': [tweet_num],
                                   'sentiment': [sentiment],
                                   'favorite_count': [favorite_count],
                                   'retweet_count': [retweet_count],
                                   's_delta_date': [s_delta_date]})

    # price
    # priceデータを読み込む(ヘッダーありCSV)
    p_data_price = pd.read_csv(path_prie, header=0)
    p_data_price['s_date'] = p_data_price['date'].astype(str)

    # twitterデータにpriceデータを追加
    df_temp = pd.merge(p_data_twitter, p_data_price, on='s_date', how='inner')[['close', 'closed_flag']]
    p_data_twitter.loc[:, 'close_price'] = df_temp['close']
    p_data_twitter.loc[:, 'closed_flag'] = df_temp['closed_flag']

    p_data_twitter.loc[:, 'delta_close_price'] = pd.merge(p_data_twitter,
                                                          p_data_price,
                                                          left_on='s_delta_date',
                                                          right_on='s_date',
                                                          how='inner')['close']

    if len(p_data_twitter.index) != 0:
        # 多クラス分類
        # 前日比
        #  2: x > 1.5 %
        #  1: 0 % <= x <= 1.5 %
        # -1: -1.5 % <= x < 0 %
        # -2: -1.5 % > x
        p_data_twitter.loc[:, 'delta_ratio'] = \
            p_data_twitter[['close_price', 'delta_close_price']].apply(
                lambda x: (x['delta_close_price'] - x['close_price']) / x['close_price'] * 100, axis=1)
        # 目的変数設定
        p_data_twitter.loc[p_data_twitter['delta_ratio'] > 1.5, 'price_result'] = 2
        p_data_twitter.loc[(p_data_twitter['delta_ratio'] >= 0) &
                           (p_data_twitter['delta_ratio'] <= 1.5), 'price_result'] = 1
        p_data_twitter.loc[(p_data_twitter['delta_ratio'] >= -1.5) &
                           (p_data_twitter['delta_ratio'] < 0), 'price_result'] = -1
        p_data_twitter.loc[p_data_twitter['delta_ratio'] < -1.5, 'price_result'] = -2
        # 前日比 削除
        del p_data_twitter['delta_ratio']

    # 削除項目
    del p_data_twitter['delta_close_price']
    del p_data_twitter['s_delta_date']

    # 欠損データの除去
    p_data_twitter.dropna(inplace=True)

    return p_data_twitter


if __name__ == '__main__':
    date_start = '20220317'
    date_end = '20220416'
    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    for index, date in enumerate(date_ary.values):
        print(f'{index}={date}')
        # date = '20220317'
        path = f'data2/ml_data_{date}'
        df = data_processing(date, -1)
        # データをファイルに出力
        df.to_csv(path, index=False)
