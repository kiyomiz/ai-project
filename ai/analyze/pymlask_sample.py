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
