# tensorflow.keras のインポート
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from ai.ai.tensorflow.cnn_tensorflow_common import CNN, train_step, measure_step

# Fashion-MNISTデータセットの読み込み
(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()

# 訓練データを正規化
x_train = x_train.astype('float32') / 255
# (60000, 28, 28)の3階テンソルを(60000, 28, 28, 1)の4階テンソルに変換
x_train = x_train.reshape(-1, 28, 28, 1)


# 損失を記録するオブジェクトを生成
train_loss = keras.metrics.Mean()
# 精度を記録するオブジェクトを生成
train_acc = keras.metrics.SparseCategoricalAccuracy()
# 検証時の損失を記録するオブジェクトを生成
val_loss = keras.metrics.Mean()
# 検証時の精度を記録するオブジェクトを生成
val_acc = keras.metrics.SparseCategoricalAccuracy()

# 訓練データと検証データに8:2の割合で分割
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.2)


# エポック数
epochs = 1
# ミニバッチのサイズ
batch_size = 64
# 訓練データのステップ数
steps = x_train.shape[0] // batch_size
# 検証データのステップ数
steps_val = x_val.shape[0] // batch_size

# 出力層10ニューロンのモデルを生成
model = CNN(x_train[0].shape, 10)

# 学習を行う
for epoch in range(epochs):
    # 訓練データと正解レベルをシャッフル
    x_, t_ = shuffle(x_train, t_train, random_state=1)

    # 1ステップにおけるミニバッチを使用した学習
    for step in range(steps):
        start = step * batch_size  # ミニバッチの先頭インデックス
        end = start + batch_size  # ミニバッチの末尾インデックス
        # ミニバッチでバイアス、重みを更新
        train_step(model, train_loss, train_acc, x_[start:end], t_[start:end])

    # 検証データによるモデルの評価
    for step_val in range(steps_val):
        start = step_val * batch_size  # ミニバッチの先頭インデックス
        end = start + batch_size  # ミニバッチの末尾インデックス
        # 検証データのミニバッチで損失と精度を測定
        measure_step(model, val_loss, val_acc, x_[start:end], t_[start:end])

    # 1エポックごとに結果を出力
    print('epoch({}) train_loss: {:.4} train_acc: {:.4} '
          'val_loss: {:.4} val_acc: {:.4}'
          .format(epoch + 1,
                  train_loss.result(),
                  train_acc.result(),
                  val_loss.result(),
                  val_acc.result()
                  )
          )

# 学習済みの重みを保存する
model.save_weights('model_weights', save_format='tf')
