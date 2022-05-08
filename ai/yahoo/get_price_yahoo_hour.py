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
                                              share.FREQUENCY_TYPE_HOUR,
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
    df["date"] = df['datetime_jst'].dt.strftime('%Y%m%d%H')

    path = 'data/price_hour'
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

    # ファイル出力
    # timestamp,open,high,low,close,volume,datetime,year_month
    df.to_csv(path, index=False)


if __name__ == '__main__':
    # 日本株については.Tをつける：code + ".T"
    # 1306 ＮＥＸＴ ＦＵＮＤＳ ＴＯＰＩＸ連動型上場投信
    # 1321 ＮＥＸＴ ＦＵＮＤＳ 日経225連動型上場投信
    company_stock(10, 1, '1321.T')
