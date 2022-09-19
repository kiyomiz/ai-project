from sklearn.model_selection import train_test_split
import MeCab
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import warnings

warnings.filterwarnings('ignore')

mecab = MeCab.Tagger('-Owakati')


def tokenize(x):
    return mecab.parse(x).split(' ')[:-1]


class Collate:

    def __init__(self, ml_base_data):
        # 分かち書きを実行
        ml_base_data['text'] = ml_base_data['text'].apply(tokenize)
        # labelは文字列に変換
        ml_base_data['label'] = ml_base_data['label'].map('{:.0f}'.format)

        # 辞書作成 <unk>や<pad>も辞書に含める
        # 文字列のみ
        # textの辞書(text_vocab)
        self.text_vocab = build_vocab_from_iterator(ml_base_data['text'], specials=['<unk>', '<pad>'])
        self.text_vocab.set_default_index(self.text_vocab['<unk>'])
        # labelの辞書(label_vocab)
        self.label_vocab = build_vocab_from_iterator(ml_base_data['label'])

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


class Net(pl.LightningModule):

    def __init__(self, n_input, n_embed, n_hidden, n_layers, n_output):
        super().__init__()
        self.embed = nn.Embedding(n_input, n_embed, padding_idx=1)
        # 双方向LSTM : bidirectional=True
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, bidirectional=True)
        # 前方向と後ろ方向の最後の隠れ層ベクトルを結合したものを受け取るので、n_hiddenは2倍にしている
        self.fc = nn.Linear(n_hidden * 2 , n_output)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.embed(x)
        # (h, c)はタブルのそれぞれの要素を分けて取得
        x, (h, c) = self.lstm(x)
        # 双方向かつlayersが1の場合、
        # h[0]がforward（前から後ろへ）
        # h[1]がbackward（後ろから前へ）
        h_forward = h[::2, :, :]
        h_backward = h[1::2, :, :]
        bih = torch.cat([h_forward[-1], h_backward[-1]], dim=1)
        x = self.fc(bih)
        return x

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y, t), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(y, t), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc(y, t), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    date_start = 20220509
    date_end = 20220531
    path = 'data/ml_base_data'
    output_ml_result_dir = f'model/{date_start}'
    ml_base_data = pd.read_csv(path, header=0)
    ml_base_data = ml_base_data.loc[(ml_base_data['s_date'] >= date_start) & (ml_base_data['s_date'] <= date_end), :]
    del ml_base_data['id']
    del ml_base_data['s_date']

    collate = Collate(ml_base_data)

    train_data, val_data = train_test_split(ml_base_data, test_size=0.4, random_state=0)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=0)
    print(f'Train:{len(train_data)}件 Val:{len(val_data)}件 Test:{len(test_data)}件')
    print(f'price_status Train  3:{len(train_data.loc[train_data["label"]=="3"])}件')
    print(f'price_status Train  2:{len(train_data.loc[train_data["label"]=="2"])}件')
    print(f'price_status Train  1:{len(train_data.loc[train_data["label"]=="1"])}件')
    print(f'price_status Train  0:{len(train_data.loc[train_data["label"]=="0"])}件')
    print(f'price_status Val    3:{len(val_data.loc[val_data["label"]=="3"])}件')
    print(f'price_status Val    2:{len(val_data.loc[val_data["label"]=="2"])}件')
    print(f'price_status Val    1:{len(val_data.loc[val_data["label"]=="1"])}件')
    print(f'price_status Val    0:{len(val_data.loc[val_data["label"]=="0"])}件')
    print(f'price_status Test   3:{len(test_data.loc[test_data["label"]=="3"])}件')
    print(f'price_status Test   2:{len(test_data.loc[test_data["label"]=="2"])}件')
    print(f'price_status Test   1:{len(test_data.loc[test_data["label"]=="1"])}件')
    print(f'price_status Test   0:{len(test_data.loc[test_data["label"]=="0"])}件')

    # バッチサイズ
    batch_size = 200

    # Data Loadkerを用意
    train_loader = DataLoader(train_data.values, batch_size, shuffle=True, collate_fn=collate.collate_batch)
    val_loader = DataLoader(val_data.values, batch_size, collate_fn=collate.collate_batch)
    test_loader = DataLoader(test_data.values, batch_size, collate_fn=collate.collate_batch)

    # 詳細設定
    n_input = len(collate.text_vocab)
    n_embed = 100
    n_hidden = 100
    n_layers = 3
    # n_outputは、labelの種類の数を指定
    n_output = 4
    # 学習回数
    epoch = 2

    # 学習の実行
    pl.seed_everything(0)
    net = Net(n_input, n_embed, n_hidden, n_layers, n_output)
    trainer = pl.Trainer(max_epochs=epoch)
    trainer.fit(net, train_loader, val_loader)

    # テストデータに対する検証
    results = trainer.test(dataloaders=test_loader)
    print(results)

    # モデルの保存
    # パラメータのみ
    # torch.save(net.state_dict(), output_ml_result_dir)
    # モデル全体
    torch.save(net, output_ml_result_dir)
