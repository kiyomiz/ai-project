from gensim.models.word2vec import Word2Vec
sentences = []
model = Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=1)

# 先ほど作成したモデルを用いて「家族」に近い単語を確認してみます。
for i in model.wv.most_similar('わたし'):
    print(i)
