import os
import pandas as pd
import MeCab
# import glob
import oseti

from dateutil.relativedelta import relativedelta


# 引数1 : 説明変数(twitterの情報)の日付
# 引数2 : 目的変数(日経平均株価の増減)と説明変数(twitterの情報)の日付差
def data_processing(s_date, delta):
    path = f'data/{s_date}'
    if not os.path.isfile(path):
        return None

    # path_prie = f'data/price_{s_date[0, 6]}'
    path_prie = f'data/price'
    if not os.path.isfile(path_prie):
        return None

    # twitterのデータをpandasに読み込む(ヘッダーなしCSV)
    p_data = pd.read_csv(path, header=None, names=['id', 'screen_name', 'jst_time', 'full_text', 'favorite_count', 'retweet_count'])
    # print(f'{s_date} {len(p_data)}')

    # priceデータを読み込む(ヘッダーありCSV)
    p_data_price = pd.read_csv(path_prie, header=0)
    # print(p_data_price.head(3))

    # twitter
    # 感情分析
    p_data.loc[:, 'ans'] = p_data[['full_text']].apply(analyze, axis=1)
    # full_textカラムを削除
    del p_data['full_text']
    # print(p_data.head(10))
    # s = p_data.loc[1, 'full_text']
    # analyzer = oseti.Analyzer()
    # analyzer.analyze(s)

    # 取得日と指定日分の過去日を作成
    p_data.loc[:, 'date_time'] = pd.to_datetime(p_data['jst_time'])
    p_data.loc[:, 's_date_time'] = p_data['date_time'].dt.strftime('%Y%m%d')
    p_data.loc[:, 'one_date_time'] = p_data['date_time'].map(lambda x: x + relativedelta(days=delta))
    p_data.loc[:, 's_one_date_time'] = p_data['one_date_time'].dt.strftime('%Y%m%d')

    # price
    p_data_price.loc[:, 'date_time'] = pd.to_datetime(p_data_price['datetime'])
    p_data_price.loc[:, 's_date_time'] = p_data_price['date_time'].dt.strftime('%Y%m%d')

    # print(p_data.head(3))
    # print(p_data_price.head(3))

    # 指定日分の過去日のpriceが上がった場合、1、下がった場合、-1、変わらなかった場合、0をpriceに付加
    # p_data.loc[:, 'date_time_price'] = pd.merge(p_data,
    #                                             p_data_price,
    #                                             on='s_date_time',
    #                                             how='inner')
    # Wrong number of items passed 19, placement implies 1
    # 右辺から左辺に代入する時に行数が足りない
    # これだと、1つのカラムに、複数カラム代入しようとしている。

    # df4 = pd.merge(p_data, p_data_price, on='s_date_time', how='inner')
    # print(df4.count())
    # print(p_data.count())

    p_data.loc[:, 'close_price'] = pd.merge(p_data,
                                            p_data_price,
                                            on='s_date_time',
                                            how='inner')['close']
    # print(p_data.head(3))

    p_data.loc[:, 'one_close_price'] = pd.merge(p_data,
                                                p_data_price,
                                                left_on='s_one_date_time',
                                                right_on='s_date_time',
                                                how='inner')['close']
    p_data.loc[p_data['close_price'] - p_data['one_close_price'] > 0, 'price_status'] = 1
    p_data.loc[p_data['close_price'] - p_data['one_close_price'] == 0, 'price_status'] = 0
    p_data.loc[p_data['close_price'] - p_data['one_close_price'] < 0, 'price_status'] = -1

    # データをファイルに出力
    p_data.to_csv(f'data2/{s_date}', index=False)


def analyze(s):
    try:
        analyzer = oseti.Analyzer()
        ans = analyzer.analyze(s['full_text'])
        ans_sum = sum(ans)
        return 1 if ans_sum > 0 else -1 if ans_sum < 0 else 0
    except:
        # print("*************")
        # print(s['full_text'])
        # print("*************")
        print("analyze error")


if __name__ == '__main__':
    data_processing('20220316', -1)

    # files = os.listdir("./data")
    # files_file = [f for f in files if os.path.isfile(os.path.join("./data", f))]
    # print(files_file)
    #
    # for f_name in files_file:
    #     data_processing(f_name, 1)

    # files = glob.glob("./data/*")
    # for file in files:
    #     print(file)

    # # pandasの結合
    # df1 = pd.DataFrame({
    #     "data1": range(6),
    #     "key": ['A', 'B', 'B', 'A', 'C', 'A']
    # })
    # df2 = pd.DataFrame({
    #     "data2": range(3),
    #     "key": ['A', 'B', 'D'],
    #     "data3": range(3)
    # })
    #
    # df3 = pd.merge(df1, df2, on='key')
    # print(df3)

