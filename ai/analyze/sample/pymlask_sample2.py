# デフォルトの辞書では2,100語の辞書によるパターンマッチングで
# {喜, 怒, 哀, 怖, 恥, 好, 厭, 昂, 安, 驚}の10種類の感情を推定します。
# この2,100語は、感情表現辞典に基づいているそうです。
# 加えて間投詞、擬態語、がさつな言葉、顔文字、「！」や「？」の数で感情の強さを推定します。

import mlask

emotion_analyzer = mlask.MLAsk()
result = emotion_analyzer.analyze(
    '居酒屋「もう駄目かも」、旅行会社「ショック」…ＧｏＴｏ見直しに懸念と困惑。新型コロナウイルスの感染が急拡大する中で迎えた３連休初日の２１日、政府が需要喚起策「Ｇｏ　Ｔｏ　キャンペーン」の運用を見直す方針を打ち出した。観光業者や旅行者からは、回復傾向にある観光への打撃を懸念する声や、方針変更に戸惑う声が相次いだ。')

print("text:{}".format(result['text']))
print("emotion:{}".format(result['emotion']))
for s in result['emotion'].items():
    print(f'emotion:{s}')
print("orientation:{}".format(result['orientation']))
print("activation:{}".format(result['activation']))
print("emoticon:{}".format(result['emoticon']))
print("intension:{}".format(result['intension']))
print("intensifier:{}".format(result['intensifier']))
print("representative:{}".format(result['representative']))
