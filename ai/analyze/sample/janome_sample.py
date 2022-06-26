# janome.tokenizerをインポート
from janome.tokenizer import Tokenizer
# Tokenizerオブジェクトを生成
t = Tokenizer()
# 形態素解析
tokens = t.tokenize('わたしPythonのプログラムです')
# 解析結果のリストから抽出
for token in tokens:
    print(token)

print(tokens[0].surface)
print(tokens[1].surface)
print(tokens[2].surface)
print(tokens[3].surface)

# 内包表記を使って文章の中の全ての形態素の見出しを取り出す
# []で囲っているのでlistになる
token_list = [token.surface for token in tokens]
print(token_list)
