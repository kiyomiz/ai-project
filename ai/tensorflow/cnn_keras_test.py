from tensorflow.keras import datasets
from tensorflow.keras.models import model_from_json

# Fashion-MNISTデータセットの読み込み
# Fashion-MNISTは、10種類のファッションアイテムのモノクロ画像が、訓練用として60,000枚、テスト用として10,000枚収録されている
(x_train, t_train), (x_test, t_test) = datasets.fashion_mnist.load_data()

# テストデータを正規化
x_test = x_test / 255
# (10000, 28, 28)の3階テンソルを(10000, 28, 28, 1)の4階テンソルに変換
x_test = x_test.reshape(-1, 28, 28, 1)

model = model_from_json(open('model.json', 'r').read())

# 学習済みモデルにテストデータを入力して損失と精度を取得
test_loss, test_acc = model.evaluate(x_test, t_test, verboss=0)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
