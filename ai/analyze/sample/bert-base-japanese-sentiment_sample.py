# 必要なライブラリのインストール
# pip install -q transformers
# pip install fugashi
# pip install ipadic
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch


tokenizer = AutoTokenizer.from_pretrained("daigo/bert-base-japanese-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("daigo/bert-base-japanese-sentiment")
# シンプルな動作確認
print(pipeline("sentiment-analysis",model="daigo/bert-base-japanese-sentiment",tokenizer="daigo/bert-base-japanese-sentiment")("私は幸福である。"))

list_text = [
             'この人は、この世の中で、いちばんしあわせな人にちがいありません。',
             '芝居小屋もすばらしいし、お客さんもすばらしい人たちでした。',
             'もし中世の時代だったら、おそらく、火あぶりにされたでしょうよ。',
             'みんなのうるさいことといったら、まるで、ハエがびんの中で、ブンブンいっているようでした。',
             'われわれ人間が、こういうことを考えだすことができるとすれば、われわれは、地の中にうめられるまでに、もっと長生きできてもいいはずだが'
]

sentiment_analyzer = pipeline("sentiment-analysis",model="daigo/bert-base-japanese-sentiment",tokenizer="daigo/bert-base-japanese-sentiment")
print(list(map(sentiment_analyzer, list_text)))

# 上記すべてポジティヴ判定になってしまったので、他の例でも試してみる
print(list(map(sentiment_analyzer, ['最悪だ', '今日は暑い', 'こんにちは', 'ふつう'])))