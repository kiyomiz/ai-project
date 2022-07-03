import os
import pandas as pd
import MeCab
# import glob
import oseti
import numpy as np

from dateutil.relativedelta import relativedelta


def analyze(s):
    try:
        analyzer = oseti.Analyzer()
        ans = analyzer.analyze(s['full_text'])
        # print(ans)
        ans_positive = len([i for i in ans if i > 0])
        ans_negative = len([i for i in ans if i < 0])
        # print(pd.Series([ans_positive + ans_negative, ans_positive, ans_negative]))
        return pd.Series([ans_positive + ans_negative, ans_positive, ans_negative])
    except:
        # print("analyze error")
        return pd.Series([np.nan, np.nan, np.nan])


# 引数1 : 説明変数(twitterの情報)の日付
# 引数2 : 目的変数(日経平均株価の増減)と説明変数(twitterの情報)の日付差
def data_processing(s_date, delta):
    path = f'../data/twitter/{s_date}'
    if not os.path.isfile(path):
        return None

    path_prie = f'../data/yahoo/price_1306.T'
    if not os.path.isfile(path_prie):
        return None

    # twitterのデータをpandasに読み込む(ヘッダーなしCSV)
    p_data_twitter = pd.read_csv(path, header=None, names=['id', 'screen_name', 'jst_time', 'full_text', 'favorite_count', 'retweet_count'])
    print(f'データ数：{len(p_data_twitter)}')

    # twitter
    # 感情分析
    p_data_twitter[['sentiment', 'positive', 'negative']] = p_data_twitter[['full_text']].apply(analyze, axis=1)
    # 日付追加
    p_data_twitter.loc[:, 'date_time'] = pd.to_datetime(p_data_twitter['jst_time'])
    p_data_twitter.loc[:, 's_date'] = p_data_twitter['date_time'].dt.strftime('%Y%m%d')
    p_data_twitter.loc[:, 'delta_date'] = p_data_twitter['date_time'].map(lambda x: x + relativedelta(days=delta))
    p_data_twitter.loc[:, 's_delta_date'] = p_data_twitter['delta_date'].dt.strftime('%Y%m%d')

    # price
    # priceデータを読み込む(ヘッダーありCSV)
    p_data_price = pd.read_csv(path_prie, header=0)
    # 日付追加
    p_data_price['s_date'] = p_data_price['date'].astype(str)

    # priceデータとtwitterデータを結合
    p_data_twitter.loc[:, 'close_price'] = pd.merge(p_data_twitter,
                                                    p_data_price,
                                                    on='s_date',
                                                    how='inner')['close']
    p_data_twitter.loc[:, 'closed_flag'] = pd.merge(p_data_twitter,
                                                    p_data_price,
                                                    on='s_date',
                                                    how='inner')['closed_flag']

    p_data_twitter.loc[:, 'delta_close_price'] = pd.merge(p_data_twitter,
                                                          p_data_price,
                                                          left_on='s_delta_date',
                                                          right_on='s_date',
                                                          how='inner')['close']

    if len(p_data_twitter.index) != 0:
        # 前日比(%)
        #  3:  x > 1
        #  2:  0 <= x < 1
        #  1: -1 <= x < 0
        #  0: -1 > x
        p_data_twitter.loc[:, 'delta_ratio'] = p_data_twitter[['close_price', 'delta_close_price']].apply(
                lambda x: (x['delta_close_price'] - x['close_price']) / x['close_price'] * 100, axis=1)
        # 目的変数設定
        p_data_twitter.loc[p_data_twitter['delta_ratio'] > 1, 'price_result'] = 3
        p_data_twitter.loc[(p_data_twitter['delta_ratio'] >= 0) &
                           (p_data_twitter['delta_ratio'] < 1), 'price_result'] = 2
        p_data_twitter.loc[(p_data_twitter['delta_ratio'] >= -1) &
                           (p_data_twitter['delta_ratio'] < 0), 'price_result'] = 1
        p_data_twitter.loc[p_data_twitter['delta_ratio'] < -1, 'price_result'] = 0
        # 前日比 削除
        del p_data_twitter['delta_ratio']

    # id,favorite_count,retweet_count,sentiment_analysis,close_price,price_result
    del p_data_twitter['screen_name']
    del p_data_twitter['jst_time']
    del p_data_twitter['full_text']
    del p_data_twitter['date_time']
    del p_data_twitter['close_price']
    del p_data_twitter['delta_date']
    del p_data_twitter['s_delta_date']
    del p_data_twitter['delta_close_price']

    # 欠損データの除去
    p_data_twitter.dropna(inplace=True)

    return p_data_twitter


if __name__ == '__main__':
    date_start = '20220509'
    date_end = '20220630'
    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    for index, date in enumerate(date_ary.values):
        print(f'{index}={date}')
        path = f'data/ml_data_{date}'
        df = data_processing(date, -1)
        # データをファイルに出力
        df.to_csv(path, index=False)

    # date = '20220509'
    # path = f'data/ml_data_{date}'
    # df = data_processing(date, -1)
    # # データをファイルに出力
    # df.to_csv(path, index=False)
