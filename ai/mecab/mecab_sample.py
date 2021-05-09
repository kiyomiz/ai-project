import MeCab
tagger = MeCab.Tagger("-Ochasen")
print(tagger.parse("今日はいい天気ですね。"))
