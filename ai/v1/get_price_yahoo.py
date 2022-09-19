import sys
import os
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import datetime
import pandas as pd


def company_stock(period_type, period, company_code):
    my_share = share.Share(company_code)

    try:
        # share.PERIOD_TYPE_DAY, 10 だと過去10日間のデータを取得
        # share.FREQUENCY_TYPE_MINUTE, 5 だと5分おきにデータを取得
        # share.FREQUENCY_TYPE_HOUR, 1だと1時間おきにデータを取得
        # share.FREQUENCY_TYPE_DAY, 1 だと1日おきにデータを取得
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                              period_type,
                                              share.FREQUENCY_TYPE_DAY,
                                              period)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    # print(type(symbol_data))
    df = pd.DataFrame(symbol_data)
    # タイムスタンプを変換
    df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")
    # 日本時間へ変換
    df["datetime_jst"] = df["datetime"] + datetime.timedelta(hours=9)
    df["date"] = df['datetime_jst'].dt.strftime('%Y%m%d')

    path = f'data/yahoo/price_{company_code}'
    if os.path.isfile(path):
        p_data = pd.read_csv(path, header=0)
        print(f'更新前：{len(p_data)}件')
        # 過去データと今回データをマージ
        df = pd.concat([df, p_data])
        # print(p_data.dtypes) date : int64
        # print(df.dtypes)     date : object
        # date列の型を変換
        df['date'] = df['date'].astype(str)
        df = df.sort_values('date')
        # 重複削除
        # 引数inplace=Trueにすると元のdfから重複した行が削除される。
        # デフォルトはinplace=False 削除したdtを戻り値として返す
        # print(df.duplicated(subset=['date'], keep='last'))
        df.drop_duplicates(subset=['date'], keep="last", inplace=True)
        print(f'更新後：{len(df)}件')

    # 休場日を埋める
    # 連続した日付を作成
    df_calender = pd.DataFrame()
    # 'yyyyMMdd'
    date_start = df['date'].min()
    date_end = df['date'].max()
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    df_calender['date'] = date_index.to_series().dt.strftime("%Y%m%d")
    # 連続した日付とプライスを結合
    df = pd.merge(df_calender, df, on='date', how='left')
    # 日付でソート デフォルトは昇順。降順にしたい場合は引数ascendingをFalseにする。
    df = df.sort_values('date')
    # 休場フラグ(0:取引日、1:休場日)
    df.loc[(df['close'].notna()) & (df['closed_flag'].isna()), 'closed_flag'] = 0
    df.loc[(df['close'].isna()) & (df['closed_flag'].isna()), 'closed_flag'] = 1
    # 欠損プライスを前日のプライスで埋める
    df.loc[:, 'close'] = df[['close']].fillna(method='ffill')

    # ファイル出力
    # timestamp,open,high,low,close,volume,datetime,year_month
    df.to_csv(path, index=False)


if __name__ == '__main__':
    # 日本株については.Tをつける：code + ".T"
    # 1306 ＮＥＸＴ ＦＵＮＤＳ ＴＯＰＩＸ連動型上場投信
    # 1321 ＮＥＸＴ ＦＵＮＤＳ 日経225連動型上場投信
    company_stock(30, 1, '1306.T')
