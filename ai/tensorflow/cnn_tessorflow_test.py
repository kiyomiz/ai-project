from tensorflow import keras
from ai.ai.tensorflow.cnn_tensorflow_common import CNN, measure_step


# Fashion-MNISTデータセットの読み込み
(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()

# テストデータを正規化
x_test = x_test.astype('float32') / 255
# (10000, 28, 28)の3階テンソルを(10000, 28, 28, 1)の4階テンソルに変換
x_test = x_test.reshape(-1, 28, 28, 1)

# 学習に使用したモデルと同じ構造のモデルを生成
model = CNN(x_test[0].shape, 10)
# モデルをコンパイル
model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=["accuracy"])
# 保存した重みを読み込む
model.load_weights('model_weights')

# 損失を記録するオブジェクトを生成
test_loss = keras.metrics.Mean()
# 精度を記録するオブジェクトを生成
test_acc = keras.metrics.SparseCategoricalAccuracy()


# テストデータで予測して損失と精度を取得
measure_step(model, test_loss, test_acc, x_test, t_test)

print('test_loss: {:.4f}, test_acc: {:.4f}'.format(
    test_loss.result(),
    test_acc.result()
))
