import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
import matplotlib.pyplot as plt


company_code = '7203.T'
my_share = share.Share(company_code)
symbol_data = None

try:
    symbol_data = my_share.get_historical(
        share.PERIOD_TYPE_YEAR, 100,
        share.FREQUENCY_TYPE_DAY, 1)
except YahooFinanceError as e:
    print(e.message)
    sys.exit(1)

df = pd.DataFrame(symbol_data)
df["datatime"] = pd.to_datetime(df.timestamp, unit="ms")
# df.head()

plt.title(company_code, color='black', size=15, loc='center')  # title(タイトル, 線の色, 背景色, フォントサイズ,　タイトル位置)
plt.plot(df.index, symbol_data["close"], label='close', color='blue')

plt.show()
