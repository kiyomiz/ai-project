from tensorflow import keras
import tensorflow as tf

# 損失関数はスパースラベル対応クロスエントロピー誤差
cce = keras.losses.SparseCategoricalCrossentropy()
# 勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = keras.optimizers.SGD(learning_rate=0.1)


class CNN(keras.Model):
    """ 畳み込みニューラルネットワーク

    Attributes:
      c1(Conv2D): 畳み込み層１
      c2(Conv2D): 畳み込み層２
      p1(MaxPooling2D): プーリング層１
      c3(Conv2D): 畳み込み層３
      p2(MaxPooling2D): プーリング層２
      d1(Dropout): ドロップアウト１
      f1(Flatten): Flatten
      l1(Dense): 全結合層
      l2(Dense): 出力層
    """

    def __init__(self, input_shape, output_dim):
        """
        Parameters:
          output_dim(int): 出力層のユニット数（次元）
        """
        super().__init__()

        # 正規化の係数
        weight_decay = 1e-4

        # (第１層）畳み込み層１
        # ニューロン数：64
        # 出力：1 ニューロンあたり（28, 28, 1)の3階テンソルを64出力するので
        # (28, 28, 64)の出力となる
        self.c1 = keras.layers.Conv2D(
            filters=64,  # フィルター数64
            kernel_size=(3, 3),  # 3 x 3 のフィルター
            padding='same',  # ゼロパディング
            input_shape=input_shape,  # 入力データの形状
            kernel_regularizer=keras.regularizers.l2(weight_decay),  # 正則化
            activation='relu'  # 活性化関数はReLU
        )

        # (第２層）畳み込み層２
        # ニューロン数：32
        # 出力：1 ニューロンあたり（28, 28, 1)の3階テンソルを32出力するので
        # (28, 28, 32)の出力となる
        self.c2 = keras.layers.Conv2D(
            filters=32,  # フィルター数32
            kernel_size=(3, 3),  # 3 x 3 のフィルター
            padding='same',  # ゼロパディング
            kernel_regularizer=keras.regularizers.l2(weight_decay),  # 正則化
            activation='relu'  # 活性化関数はReLU
        )

        # (第３層）プーリング層１
        # ニューロン数：32
        # 出力：1 ニューロンあたり（14, 14, 1)の3階テンソルを32出力するので
        # (14, 14, 32)の出力となる
        self.p1 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))  # 縮小対象の領域は2x2

        # (第４層）畳み込み層３
        # ニューロン数：16
        # 出力：1 ニューロンあたり（14, 14, 1)の3階テンソルを16出力するので
        # (14, 14, 16)の出力となる
        self.c3 = keras.layers.Conv2D(
            filters=16,  # フィルター数16
            kernel_size=(3, 3),  # 3 x 3 のフィルター
            padding='same',  # ゼロパディング
            kernel_regularizer=keras.regularizers.l2(weight_decay),  # 正則化
            activation='relu'  # 活性化関数はReLU
        )

        # (第５層）プーリング層２
        # ニューロン数：16
        # 出力：1 ニューロンあたり（7, 7, 1)の3階テンソルを16出力するので
        # (7, 7, 16)の出力となる
        self.p2 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))  # 縮小対象の領域は2x2

        # ドロップアウト40%
        self.d1 = keras.layers.Dropout(0.4)

        # Flatten
        # ニューロン数＝7x7x16=794
        # (7, 7, 16)を（784）にフラット化
        self.f1 = keras.layers.Flatten()

        # (第6層)全結合層
        # ニューロン数：128
        # 出力：(128）の1階テンソルを出力
        self.l1 = keras.layers.Dense(
            128,  # ニューロン数は128
            activation='relu')  # 活性化関数はReLU

        # (第7層）出力層
        # ニューロン数：10
        # 出力：要素数（10）の1階テンソルを出力
        self.l2 = keras.layers.Dense(
            output_dim,  # 出力層のニューロン数は10
            activation='softmax')  # 活性化関数はソフトマックス

        # すべての層をリストにする
        self.ls = [self.c1, self.c2, self.p1, self.c3,
                   self.p2, self.d1, self.f1, self.l1, self.l2]

    def __call__(self, x):
        """MLPのインスタンスからコールバックされる関数

        Parameters: x(ndarray(float32)):訓練データ、または検証データ
        Returns(float32): CNNの出力として要素数10の1階テンソル
        """
        for layer in self.ls:
            x = layer(x)

        return x


def loss(t, y):
    """損失関数
    Parameters: t(ndarray(float32)):正解ラベル
                y(ndarray(float32)):予測値

    Returns: クロスエントロピー誤差
    """
    return cce(t, y)


def train_step(model, train_loss, train_acc, x, t):
    """学習を１回行う

    Parameters: x(ndarray(float32)):訓練データ
                t(ndarray(float32)):正解ラベル

    Returns:
      ステップごとのクロスエントロピー誤差
    """
    # 自動微分による勾配計算を記録するブロック
    with tf.GradientTape() as tape:
        # モデルに入力して順伝播の出力値を取得
        outputs = model(x)
        # 出力値と正解ラベルの誤差
        tmp_loss = loss(t, outputs)

    # tapeに記録された操作を使用して誤差の勾配を計算
    grads = tape.gradient(
        # 現在のステップの誤差
        tmp_loss,
        # バイアス、重みのリストを取得
        model.trainable_variables)
    # 勾配降下法の更新式を適用してバイアス、重みを更新
    optimizer.apply_gradients(zip(grads,
                                  model.trainable_variables))

    # 損失をMeanオブジェクトに記録
    train_loss(tmp_loss)
    # 精度をSparseCategoricalAccuracyオブジェクトに記録
    train_acc(t, outputs)


def measure_step(model, record_loss, record_acc, x, t):
    """データをモデルに入力して損失と精度を測定

    Parameters: x(ndarray(float32)):検証データ or テストデータ
                t(ndarray(float32)):正解ラベル
    """
    # 検証データの予測値を取得
    preds = model(x)
    # 出力値と正解ラベルの誤差
    tmp_loss = loss(t, preds)
    # 損失をMeanオブジェクトに記録
    record_loss(tmp_loss)
    # 精度をSparseCategoricalAccuracyオブジェクトに記録
    record_acc(t, preds)
