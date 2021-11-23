from tensorflow.keras import datasets
from tensorflow.keras import models, layers, optimizers, regularizers

# 目的：入力画像が10種類のどの種類に属するか分類する

# Fashion-MNISTデータセットの読み込み
# Fashion-MNISTは、10種類のファッションアイテムのモノクロ画像が、訓練用として60,000枚、テスト用として10,000枚収録されている
(x_train, t_train), (x_test, t_test) = datasets.fashion_mnist.load_data()

# 訓練データを正規化
x_train = x_train / 255

# (60000, 28, 28)の3階テンソルを(60000, 28, 28, 1)の4階テンソルに変換
# 60000が画像数、28,28が1画像、1が1画像で、フィルタして枚数が増える
# -1を指定した次元は要素数を変えない
x_train = x_train.reshape(-1, 28, 28, 1)

# CNNモデルの定義
# tf.keras.models.Sequentialは、tf.keras.Sequentialのエイリアス
model = models.Sequential()

# 正則化の係数
weight_decay = 1e-4

# (第１層）畳み込み層１
# ニューロン数：64
# 出力：1ニューロンあたり（28, 28, 1)の３階テンソルを64出力するので
#      (28, 28, 64)の出力となる
model.add(
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        input_shape=(28, 28, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(
            weight_decay),
        activation='relu'
    ))

# (第2層）畳み込み層2
# ニューロン数：32
# 出力：1ニューロンあたり（28, 28, 1)の３階テンソルを32出力するので
#      (28, 28, 32)の出力となる
model.add(
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer=regularizers.l2(
            weight_decay),
        activation='relu'
    ))

# (第3層)プーリング層１
# ニューロン数：32
# 出力：1ニューロンあたり（14, 14, 1)の3階テンソルを32出力するので
#      (14, 14, 32)の出力となる
model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)))

# (第4層）畳み込み層3
# ニューロン数：16
# 出力：1ニューロンあたり（14, 14, 1)の３階テンソルを32出力するので
#      (14, 14, 16)の出力となる
model.add(
    layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer=regularizers.l2(
            weight_decay),
        activation='relu'
    ))

# (第5層)プーリング層2
# ニューロン数：16
# 出力：1ニューロンあたり（7, 7, 1)の3階テンソルを16出力するので
#      (7, 7, 16)の出力となる
model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)))

# ドロップアウト40%
model.add(layers.Dropout(0.4))

# Flattern
# ニューロン数＝7x7x16=784
model.add(layers.Flatten())

# （第６層）全結合層
# ニューロン数：128
# 出力：128の1階テンソルを出力
model.add(
    layers.Dense(
        128,
        activation='relu'))

# 学習率
learning_rate = 0.1
# モデルのコンパイル
model.compile(
    # 損失関数はスパースラベル対応クロスエントロピー誤差
    loss='sparse_categorical_crossentropy',
    # オプティマイザーはSGD
    optimizer=optimizers.SGD(lr=learning_rate),
    # 学習評価として正解率を指定
    metrics=['accuracy'])

# サマリーを表示
model.summary()

# エポック数
epoch = 5
# ミニバッチのサイズ
batch_size = 64

history = model.fit(
    # 訓練データ
    x_train,
    # 正解データ
    t_train,
    # ミニバッチのサイズを設定 イテレーション　データを複数に分けて重みの更新をする
    batch_size=batch_size,
    # エポック数を設定 学習回数、同じデータでの学習を何回繰り返すか？ 重みの更新回数 = イテレーション × エボック
    epochs=epoch,
    # 進捗状況を出力する
    verbose=1,
    # 20パーセントのデータを検証に使用
    validation_split=0.2,
    shuffle=True
)

with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())
