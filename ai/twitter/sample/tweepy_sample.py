import tweepy
import ai.ai.twitter.settings as settings

# 認証に必要なキーとトークン
API_KEY = settings.CONSUMER_KEY
API_SECRET = settings.CONSUMER_SECRET
ACCESS_TOKEN = settings.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = settings.ACCESS_SECRET

# APIの認証
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# キーワードからツイートを取得
api = tweepy.API(auth)
tweets = api.search(q=['Python'], count=10)

for tweet in tweets:
    print('-----------------')
    print(tweet.text)
