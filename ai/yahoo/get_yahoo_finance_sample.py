import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def company_stock(period_type, period, company_code):
    my_share = share.Share(company_code)

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                              period_type,
                                              share.FREQUENCY_TYPE_MINUTE,
                                              period)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    date = symbol_data["timestamp"]

    old_date = date[0]
    now_date = date[len(date) - 1]

    old_time = datetime.fromtimestamp(old_date / 1000)
    now_time = datetime.fromtimestamp(now_date / 1000)

    price = symbol_data["close"]
    old_price = price[0]
    now_price = price[len(date) - 1]
    print(str(old_time) + "の時の株価： " + str(old_price))
    print(str(now_time) + "の時の株価： " + str(now_price))

    df = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    df.index = pd.DatetimeIndex(df.timestamp, name='timestamp').tz_localize('UTC').tz_convert('Asia/Tokyo')

    plt.title(company_code, color='black', size=15, loc='center')  # title(タイトル, 線の色, 背景色, フォントサイズ,　タイトル位置)
    plt.plot(df.index, price, label='close', color='blue')

    plt.show()


plt.figure(figsize=(10, 5))

company_stock(4, 1, '7203.T')
