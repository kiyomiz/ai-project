import tweepy
import datetime
import time
import ai.ai.twitter.settings as settings


def get_twitter_data(screen_name_list, file_name):
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

    # ユーザリストのデータ取得
    for name in screen_name_list:
        try:
            print(name)
            # カーソルを使用してデータ取得
            for tweet in tweepy.Cursor(api.user_timeline,
                                       screen_name=name,
                                       exclude_replies=True
                                       ).items(20):
                print('Get api.user_timeline')
                if sinceDate <= tweet.created_at < untilDate:
                    tweets_data.append(",".join(map(str, [tweet.id, tweet.created_at, tweet.text.replace('\n', ''), tweet.favorite_count, tweet.retweet_count])))
                    tweets_data.append('\n')

            # 10秒停止
            time.sleep(10)
        except Exception as e:
            print(e)


    # 出力ファイル名
    out_file_name = r"'" + file_name + "'"
    out_file_name = out_file_name.replace("'", "")

    # ファイル出力
    with open(out_file_name, "w", encoding="utf-8") as f:
        f.writelines(tweets_data)


if __name__ == '__main__':
    # 出力ファイル名を入力(相対パス or 絶対パス)
    print('====== Enter Tweet Data file =====')
    data_file = input('>  ')

    # ユーザーリスト
    user_list = [
        '2okutameo',
        'cissan_9984',
        'tesuta001',
        'kabu1000',
        'jackjack2010',
        'matsunosuke_jp',
        'sp500500',
        'yuunagi_dan',
        'Akito8868',
        'investorduke',
        'DAIBOUCHO',
        'nikkei',
        'Invesdoctor',
        'toushisenrigan',
        'aryarya',
        'BloombergJapan',
        'ReutersJapan',
        'AdamSmith2sei',
        'Voodoochile2',
        'the_phoenix_777',
        'nicoDisclosure',
        'yuzz__',
        'WSJJapan',
        '4ki4',
        'kabuojisan28',
        'buffett_code',
        'sak_07_',
        'KabukaPick',
        'toushi_kirby',
        'Money_tweet777',
        'kabukautokyo',
        'motti1234',
        'matenrou_nyattu',
        'kabutrader_J',
        'latte_koime',
        'stpedia',
        'worldworld4',
        'Akito8868',
        'Nagoya_Tyouki',
        'takosensei2019',
        'cjhiking',
        'stockprayer',
        'skew123',
        'shikihojp',
        'rikachan67',
        'make_life_rich',
        ]
    get_twitter_data(user_list, data_file)
