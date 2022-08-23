import MeCab
import warnings
warnings.filterwarnings('ignore')

# -Owakati：文章を単語別に分かち書きするのみ
# -Ochasen：分かち書きと併せて、品詞などの形態素解析に必要な情報が得られる
mecab = MeCab.Tagger("-Ochasen")


def get_nouns(text):
    nouns = []
    res = mecab.parse(text)
    # [:-2]は、EOS(End Of String)と空白('')が入っており、文章とは関係がないので取り除く
    words = res.split('\n')[:-2]
    # print(words)
    for word in words:
        part = word.split('\t')
        # print(part)
        if '名詞' in part[3]:
            nouns.append(part[0])
    return nouns


# 例文
text1 = 'キカガクでは、ディープラーニングを含んだ機械学習や人工知能のキカガク流教育を行なっています。'
text2 = 'キカガクの代表の吉崎は大学院では機械学習・ロボットのシステム制御、画像処理の研究に携わっていました。'
text3 = '機械学習は微分や線形代数を始めとした数学が不可欠で、数学が機械学習の知識を支える土台となります。'

nouns1 = get_nouns(text1)
nouns2 = get_nouns(text2)
nouns3 = get_nouns(text3)

# 特徴量への変換
from gensim import corpora, matutils

# 名詞リスト
word_collect = [nouns1, nouns2, nouns3]
print(word_collect)

# 辞書の作成
dictionary = corpora.Dictionary(word_collect)
print(len(dictionary))
# for word in dictionary.items():
#    print(word)

# 全単語数を取得
n_words = len(dictionary)

# BoWによる特徴ベクトルの作成
# 単語のone-hot表現
x = []
for nouns in word_collect:
    # dictionary.doc2bowでBowに変換
    # listの要素が、tupleで、tupleの値は (単語id, 出現回数)
    bow_id = dictionary.doc2bow(nouns)
    # matutils.corpus2denseで、全単語数の長さに変換 listの要素のインデックスが単語id、要素がlistで出現回数で縦方向に保持（複数行、一列）
    # 横方向（一行、複数列）に変換　Tを使うと転置できる
    # ベクトル化するため最初の要素のみ抽出します。
    bow = matutils.corpus2dense([bow_id], n_words).T[0]
    x.append(bow)

print(x[0])
print(x[1])
print(x[2])
