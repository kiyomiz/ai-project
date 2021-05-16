import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime


def company_stock(period_type, period, company_code):
    my_share = share.Share(company_code)

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                              period_type,
                                              share.FREQUENCY_TYPE_DAY,
                                              period)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    # エポックミリ秒
    date = symbol_data["timestamp"]
    price = symbol_data["close"]

    # ファイル出力
    with open('price', "w", encoding="utf-8") as f:

        for i in zip(date, price):
            # エポック秒から日付へ変換
            dt = datetime.fromtimestamp(i[0] / 1000)
            print(str(dt) + "の時の株価： " + str(i[1]))
            f.write(str(i[1]) + "\n")


# 1306 ＮＥＸＴ　ＦＵＮＤＳ　ＴＯＰＩＸ連動型上場投信
company_stock(60, 1, '1306.T')
