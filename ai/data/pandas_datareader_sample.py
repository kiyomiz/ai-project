import matplotlib
import matplotlib.pyplot as plt
import pandas_datareader as pdr

df = pdr.DataReader("NIKKEI225","fred")
print(df.head())

# title(タイトル, 線の色, 背景色, フォントサイズ,　タイトル位置)
# plt.title("NIKKEI225", color='black', size=15, loc='center')
# plt.plot(df.index, df["NIKKEI225"], label='NIKKEI225', color='blue')
# plt.show()

# 初期化 (表示用figure)
plt.figure()
df.plot()
# 実際にグラフウィンドウを表示する
plt.show()
# 保存
# plt.savefig('pandas_iris_line.png')
plt.close('all')
