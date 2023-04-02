import os
import pandas as pd
import MeCab
# import glob
import oseti
import numpy as np

from dateutil.relativedelta import relativedelta


# 引数1 : 説明変数(twitterの情報)の日付
def check_data(s_date):
    path = f'data/twitter/{s_date}'
    if not os.path.isfile(path):
        return 0

    # twitterのデータをpandasに読み込む(ヘッダーなしCSV)
    p_data_twitter = pd.read_csv(path, header=None, names=['id', 'screen_name', 'jst_time', 'full_text', 'favorite_count', 'retweet_count'])
    result = len(p_data_twitter)
    print(f'{s_date}   {result}')
    return result


if __name__ == '__main__':
    date_start = '20220509'
    date_end = '20221031'

    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    print('日付  データ件数')
    total = 0
    for index, date in enumerate(date_ary.values):
        # print(f'{index}={date}')
        total += check_data(date)

    print(f'合計：{total}')
