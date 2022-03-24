import tweepy
import datetime
import time
import ai.ai.twitter.settings as settings


def get_twitter_data(file_name):
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

    api = tweepy.API(auth, wait_on_rate_limit=True)

    # 日付
    sinceDate = datetime.datetime.strptime(file_name, '%Y%m%d')  # この日付以降のツイートを取得する
    untilDate = datetime.datetime.strptime(file_name, '%Y%m%d') + datetime.timedelta(days=1)  # この日付以前のツイートを取得する
    print(sinceDate)
    print(untilDate)

    # つぶやきを格納するリスト
    tweets_data = []

    # 現在の自分の上限とwindowごと使用回数の確認
    print(api.rate_limit_status()["resources"]["search"])

    # ユーザリスト取得
    # 'tarou'という名称を部分的に設定しているuserを検索する
    users = api.search_users(q='aaa')

    # 取得したuserのscreen_nameを表示する。実userなのでマスク処理を施す
    for name in users:
        n = name.screen_name
        print(n.replace(n[0:3], '***'))

    trends = api.trends_available()
    for trend in trends:
        print(trend)

    # 現在の自分の上限とwindowごと使用回数の確認
    print(api.rate_limit_status()["resources"]["search"])


if __name__ == '__main__':
    # 出力ファイル名を入力(相対パス or 絶対パス)
    print('====== Enter Tweet Data file =====')
    data_file = input('>  ')

    get_twitter_data(data_file)
