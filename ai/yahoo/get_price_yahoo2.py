import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime
import pandas as pd


def company_stock(period_type, period, company_code):
    my_share = share.Share(company_code)

    try:
        # share.PERIOD_TYPE_DAY, 10 だと過去10日間のデータを取得
        # share.FREQUENCY_TYPE_MINUTE, 5 だと5分おきにデータを取得
        # share.FREQUENCY_TYPE_DAY, 1 だと1日おきにデータを取得
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                              period_type,
                                              share.FREQUENCY_TYPE_DAY,
                                              period)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    df = pd.DataFrame(symbol_data)
    df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")

    # timestamp,open,high,low,close,volume,datetime
    df.to_csv("price", index=False)


# 日本株については.Tをつける：code + ".T"
# 1306 ＮＥＸＴ ＦＵＮＤＳ ＴＯＰＩＸ連動型上場投信
# 1321 ＮＥＸＴ ＦＵＮＤＳ 日経225連動型上場投信
company_stock(60, 1, '1321.T')
