# デフォルトの辞書では2,100語の辞書によるパターンマッチングで
# {喜, 怒, 哀, 怖, 恥, 好, 厭, 昂, 安, 驚}の10種類の感情を推定します。
# この2,100語は、感情表現辞典に基づいているそうです。
# 加えて間投詞、擬態語、がさつな言葉、顔文字、「！」や「？」の数で感情の強さを推定します。

import mlask
import subprocess

list_text = [
    'この人は、この世の中で、いちばんしあわせな人にちがいありません。',
    '芝居小屋もすばらしいし、お客さんもすばらしい人たちでした。',
    'もし中世の時代だったら、おそらく、火あぶりにされたでしょうよ。',
    'みんなのうるさいことといったら、まるで、ハエがびんの中で、ブンブンいっているようでした。',
    'われわれ人間が、こういうことを考えだすことができるとすれば、われわれは、地の中にうめられるまでに、もっと長生きできていいはずだが'
]

emotion_analyer = mlask.MLAsk()
ans = emotion_analyer.analyze('彼のことは嫌いではない')
print(ans)
ans = emotion_analyer.analyze('嫌い')
print(ans)

cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         shell=True).communicate()[0]).decode('utf-8')

emotion_analyer_2 = mlask.MLAsk('-d {0}'.format(path))
print(list(map(emotion_analyer_2.analyze, list_text)))
# 結果
# [
#  {'text': 'この人は、この世の中で、いちばんしあわせな人にちがいありません。',
#   'emotion': defaultdict(<class 'list'>, {'yorokobi': ['しあわせ']}),
#   'orientation': 'POSITIVE',
#   'activation': 'NEUTRAL',
#   'emoticon': None,
#   'intension': 0,
#   'intensifier': {},
#   'representative': ('yorokobi', ['しあわせ'])
#  },
#  {'text': '芝居小屋もすばらしいし、お客さんもすばらしい人たちでした。', 'emotion': None},
#  {'text': 'もし中世の時代だったら、おそらく、火あぶりにされたでしょうよ。', 'emotion': None},
#  {'text': 'みんなのうるさいことといったら、まるで、ハエがびんの中で、ブンブンいっているようでした。', 'emotion': None},
#  {'text': 'われわれ人間が、こういうことを考えだすことができるとすれば、われわれは、地の中にうめられるまでに、もっと長生きできていいはずだが', 'emotion': None}
# ]
