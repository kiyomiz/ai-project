import tweepy
import datetime
import ai.ai.twitter.settings as settings


def gettwitterdata(keyword, dfile):
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

    # 検索キーワード設定
    q = keyword

    # つぶやきを格納するリスト
    tweets_data = []

    # カーソルを使用してデータ取得
    # for tweet in tweepy.Cursor(api.search, q=q, count=100, tweet_mode='extended', lang='ja').items():
    for tweet in tweepy.Cursor(api.search, q=q, tweet_mode='extended', lang='ja').items(100):
        # つぶやき時間がUTCのため、JSTに変換  ※デバック用のコード
        jsttime = tweet.created_at + datetime.timedelta(hours=9)
        print(jsttime)
        # つぶやきテキスト(FULL)を取得
        tweets_data.append(tweet.full_text + '\n')

    # 出力ファイル名
    fname = r"'" + dfile + "'"
    fname = fname.replace("'", "")

    # ファイル出力
    with open(fname, "w", encoding="utf-8") as f:
        f.writelines(tweets_data)


if __name__ == '__main__':
    # 検索キーワードを入力  ※リツイートを除外する場合 「キーワード -RT 」と入力
    print('====== Enter Serch KeyWord   =====')
    keyword = input('>  ')

    # 出力ファイル名を入力(相対パス or 絶対パス)
    print('====== Enter Tweet Data file =====')
    dfile = input('>  ')

    gettwitterdata(keyword, dfile)
