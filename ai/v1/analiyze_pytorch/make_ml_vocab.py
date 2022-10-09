import os
import MeCab
import pandas as pd
import torchtext
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T


class Collate:
    mecab = MeCab.Tagger('-Owakati')
    text_name = 'test'
    label_name = 'label'

    def tokenize(x):
        return Collate.mecab.parse(x).split(' ')[:-1]

    def __init__(self, v_path, ml_base_data):
        if ml_base_data is None:
            # v_pathからvocabを読み込む
            self.text_vocab = self.read_vocab(f'{v_path}_{Collate.text_name}')
            self.text_vocab.set_default_index(self.text_vocab['<unk>'])
            self.label_vocab = self.read_vocab(f'{v_path}_{Collate.label_name}')

        else:
            # v_pathへvocabを保存する

            # 辞書作成 <unk>や<pad>も辞書に含める
            # 文字列のみ
            # textの辞書(text_vocab)
            self.text_vocab = build_vocab_from_iterator(ml_base_data['text'], specials=['<unk>', '<pad>'])
            self.text_vocab.set_default_index(self.text_vocab['<unk>'])
            # labelの辞書(label_vocab)
            self.label_vocab = build_vocab_from_iterator(ml_base_data['label'])

            # 辞書の保存
            self.save_vocab(self.text_vocab, f'{v_path}_{Collate.text_name}')
            self.save_vocab(self.label_vocab, f'{v_path}_{Collate.label_name}')

        # transform生成
        # テキストは、辞書による変換(数値化)とパディング、Tensor型への変換を行います。パディングは、ミニバッチごとに系列長を統一するため不足部分がパディングされます。
        # ラベルは、辞書による変換(数値化)とTensor型への変換を行います。
        text_transform = T.Sequential(
            T.VocabTransform(self.text_vocab),
            T.ToTensor(padding_value=self.text_vocab['<pad>'])
        )
        label_transform = T.Sequential(
            T.VocabTransform(self.label_vocab),
            T.ToTensor()
        )
        self.text_transform = text_transform
        self.label_transform = label_transform

    # ミニバッチ時のデータ変換関数
    # リスト内包表記
    # x = [リストの要素を計算する式 for 計算で使用する変数 in 反復可能オブジェクト]
    def collate_batch(self, batch):
        # 2次元の場合、LSTMに入れるため、shape(Batch_size[行], vocabrary[列])を転置する
        texts = self.text_transform([text for (text, label) in batch]).T
        labels = self.label_transform([label for (text, label) in batch])
        return texts, labels

    # 辞書vocabの保存
    def save_vocab(self, vocab, path):
        if os.path.isfile(path):
            os.remove(path)

        with open(path, 'w+', encoding='utf-8', newline='\n') as f:
            for i, (token, index) in enumerate(vocab.get_stoi().items()):
                f.write(f'{index}\t{token}\n')

    # 辞書vocabの読込み
    def read_vocab(self, path):
        vocab = dict()
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                index, token = line.split('\t')
                token = token.replace('\n', '')
                vocab[token] = int(index)

        # 値(index)で降順にソート
        vocab2 = sorted(vocab.items(), key=lambda x: x[1])
        # vocabの作成
        vocab3 = torchtext.vocab.vocab(dict())
        for i, (token, index) in enumerate(vocab2):
            if i != index:
                print(f'index:{index}、token:{token}')
            vocab3.append_token(token)
        return vocab3


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    date_start = 20220509
    date_end = 20220923
    vocab_dir = 'vocab2'
    vocab_path = f'{vocab_dir}/{date_start}-{date_end}'
    data_dir = 'data2-6'
    data_path = f'{data_dir}/ml_base_data'

    ml_base_data = pd.read_csv(data_path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]
    del ml_base_data['id']
    del ml_base_data['s_date']

    # 分かち書きを実行
    ml_base_data['text'] = ml_base_data['text'].apply(Collate.tokenize)
    # labelはfloatから文字列に変換(vocabが文字列のみのため)
    ml_base_data['label'] = ml_base_data['label'].map('{:.0f}'.format)

    # vocabの作成＆保存
    Collate(vocab_path, ml_base_data)
    # vocabの読込み
    Collate(vocab_path, None)
