import os

import pandas as pd
import tweepy
import datetime
import ai.ai.twitter.settings as settings
import csv


def get_twitter_data(keyword, str_datetime):
    # python で Twitter APIを使用するためのConsumerキー、アクセストークン設定
    # <Twitter API申請して取得したConsumer_key>
    Consumer_key = settings.CONSUMER_KEY
    # <Twitter API申請して取得したConsumer_secret>
    Consumer_secret = settings.CONSUMER_SECRET
    # <Twitter API申請して取得したAccess_token>
    Access_token = settings.ACCESS_TOKEN
    # <Twitter API申請して取得したAccess_secret>
    Access_secret = settings.ACCESS_SECRET

    # 認証
    auth = tweepy.OAuthHandler(Consumer_key, Consumer_secret)
    auth.set_access_token(Access_token, Access_secret)

    # wait_on_rate_limit = True　しておくと、利用禁止期間の解除を待機してくれるようになる。
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # 検索キーワード設定
    q = keyword

    # 日付設定
    tmp_datetime = datetime.datetime.strptime(str_datetime, "%Y%m%d%H%M%S")
    str_date = datetime.datetime.strftime(tmp_datetime, "%Y%m%d")
    since_date = datetime.datetime.strftime(tmp_datetime, "%Y-%m-%d_%H:%M:%S_JST")
    until_date = datetime.datetime.strftime(tmp_datetime, "%Y-%m-%d_23:59:59_JST")

    # デバック用
    # print(f'since_date:{since_date}')
    # print(f'until_date:{until_date}' )

    # つぶやきを格納するリスト
    tweets_data = []

    # 制限
    cnt = 0

    # カーソルを使用してデータ取得
    # TODO sinceは、2022/5/15時点、API referenceにない。期待する結果が得られない
    #      ai/ai/v1に移行
    for tweet in tweepy.Cursor(api.search, q=q,
                               tweet_mode='extended',
                               since=since_date,
                               until=until_date,
                               lang='ja').items():
        cnt += 1
        # デバック用
        print(f'tweet.created_at:{tweet.created_at}')
        # 時間がUTCのため、JSTに変換
        jst_time = tweet.created_at + datetime.timedelta(hours=9)

        # 取得
        # tweet.id　ツイートのID。ユニークなもので絶対に重複しない。ツイートの識別として利用できる。
        # tweet.user.screen_name　ユーザーネーム（英字のやつ）
        # tweet.created_at　ツイートされた時間
        # tweet.full_text　ツイート内容。核。tweet_mode = ‘extended’をつけていないと、full_textではなくtextとなるため注意。
        # tweet.favorite_count　ツイートのいいねの数
        # tweet.retweet_count　ツイートのリツイートされた数
        tweets_data.append([tweet.id,
                            tweet.user.screen_name,
                            jst_time,
                            tweet.full_text,
                            tweet.favorite_count,
                            tweet.retweet_count,
                            ])

        if cnt == 900:
            break

    print(f'data count : {len(tweets_data)}')

    # 出力ファイル名
    file_name = r"'" + str_date + "'"
    file_name = file_name.replace("'", "")

    # ファイル出力
    with open('data/' + file_name, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(tweets_data)


if __name__ == '__main__':
    # input_keyword = '日経平均 OR TOPIC'
    input_keyword = '＃日経平均 -filter:retweets'
    input_datetime = '20220514'

    path = f'data/{input_datetime}'
    # print(input_datetime[:8])
    if os.path.isfile(path):
        t_data = pd.read_csv(path, header=None, names=['id', 'screen_name', 'jst_time', 'full_text', 'favorite_count', 'retweet_count'])
        # print(t_data.head())
        max_date = pd.to_datetime(t_data["jst_time"]).max()
        max_date = max_date + datetime.timedelta(seconds=1)
        since_datetime = max_date.strftime("%Y%m%d%H%M%S")
    else:
        since_datetime = f'{input_datetime}000000'

    print(since_datetime)
    # ====== Enter date [ex. 20220306010101] yyyyMMddhh24miss=====
    get_twitter_data(input_keyword, since_datetime)
