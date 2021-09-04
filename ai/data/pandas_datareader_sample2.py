import pandas_datareader.data as pdr
import mplfinance as mpf


def make_data_frame(code):
  df_temp = pdr.DataReader("{}.JP".format(code), "stooq")
  df = df_temp.loc[:,['Open','High','Low','Close','Volume']].sort_values('Date')
  return df


df = make_data_frame(3558).tail(100)
print(df.head())

# ローソク足チャートを表示
# mpf.plot(df, type='candle')
# 出来高を表示
# mpf.plot(df, type='candle', volume=True)
# 移動平均線を表示 5, 25, 75日線を表示する例
# mpf.plot(df, type='candle', mav=(5, 25, 75), volume=True)

from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low
data = df['Close'].values.tolist()
period = 20
bb_up = bb_up(data,period)
bb_mid = bb_mid(data,period)
bb_low = bb_low(data,period)
df['bb_up'] = bb_up
df['bb_mid'] = bb_mid
df['bb_low'] = bb_low

apd = mpf.make_addplot(df[['bb_up', 'bb_mid', 'bb_low']])
mpf.plot(df, type='candle', addplot=apd, volume=True)

