import os
import pandas as pd

if __name__ == '__main__':
    base_path = 'data/ml_base_data'

    date_start = '20220509'
    date_end = '20220831'
    # date_indexのデータ型：datetime64
    date_index = pd.date_range(start=date_start, end=date_end, freq="D")
    # date_aryは、pandas.core.series.Series
    date_ary = date_index.to_series().dt.strftime("%Y%m%d")

    df_date = pd.DataFrame()
    # for文+enumerate関数で配列から要素とインデックスを順に取り出す
    for index, date in enumerate(date_ary.values):
        print(f'{index}={date}')
        # date = '20220414'
        date_path = f'data/ml_data_{date}'
        if os.path.isfile(date_path):
            df_date = df_date.append(pd.read_csv(date_path, header=0, engine='python'), ignore_index=True)

    df_base = pd.DataFrame()
    if os.path.isfile(base_path):
        df_base = pd.read_csv(base_path, header=0)

    df = df_base.append(df_date, ignore_index=True)
    # df_newもdfもcsvから取得しているので列のデータ型が異なることはない。
    df.drop_duplicates(subset=['id'], keep="last", inplace=True)
    # データをファイルに出力
    df.to_csv(base_path, index=False)
