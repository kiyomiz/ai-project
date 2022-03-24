from asari.api import Sonar
import sklearn

print(sklearn.__version__)

list_text = [
    'この人は、この世の中で、いちばんしあわせな人にちがいありません。',
    '芝居小屋もすばらしいし、お客さんもすばらしい人たちでした。',
    'もし中世の時代だったら、おそらく、火あぶりにされたでしょうよ。',
    'みんなのうるさいことといったら、まるで、ハエがびんの中で、ブンブンいっているようでした。',
    'われわれ人間が、こういうことを考えだすことができるとすれば、われわれは、もっと長生きできていいはずだが'
]

sonar = Sonar()
res = sonar.ping(text="広告多すぎる")
print(res)
res = sonar.ping(text="サイコー")
print(res)

res = list(map(sonar.ping, list_text))
print(res)
