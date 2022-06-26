import MeCab
# Owakati : 文章を単語別に分かち書きするのみ
# Ochasen : 分かち書きとあわせて、品詞などの形態素解析に必要な情報が得られる
tagger = MeCab.Tagger("-Ochasen")
print(tagger.parse("今日はいい天気ですね。"))
