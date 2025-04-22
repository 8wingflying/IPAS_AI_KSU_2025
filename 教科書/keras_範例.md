# 
- [新一代 Keras 3.x 重磅回歸：跨 TensorFlow 與 PyTorch 建構 Transformer、CNN、RNN、LSTM 深度學習模型|陳會安](https://www.tenlong.com.tw/products/9789863127871?list_name=srh)
## keras 3==> high-level + backend(Tensorflow|PyTorch |Jax)
- https://keras.dev.org.tw/about/
- neural network(MLP) ==> 分類 + 回歸
- Computer vision ==> CNN
- NLP ==> SimpleRNN + LSTM + GRU()
## neural network(MLP) ==> 分類
```python
import pandas as pd

# 載入資料集
df = pd.read_csv("./iris.csv")
# 顯示資料集的形狀
print(df.shape)
```
```python
print(df.head())
df.head().to_html("ch6-1-1a_01.html")
print("--------------------------")
# 顯示資料集的描述資料
print(df.describe())
df.describe().to_html("ch6-1-1a_02.html")****
```
```python
colmap = np.array(["r", "g", "y"])
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.subplots_adjust(hspace = .5)
plt.scatter(df["sepal_length"], df["sepal_width"], color=colmap[y])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.subplot(1, 2, 2)
plt.scatter(df["petal_length"], df["petal_width"], color=colmap[y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

# 使用Seaborn顯示視覺化圖表
sns.pairplot(df, hue="target")
```
```python
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical

np.random.seed(7)  # 指定亂數種子
# 載入資料集
df = pd.read_csv("./iris_data.csv")
target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}
df["target"] = df["target"].map(target_mapping)
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:,0:4].astype(float)
y = to_categorical(dataset[:,4])

# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)


# 分割成訓練和測試資料集 ==> train_test_split
X_train, y_train = X[:120], y[:120]     # 訓練資料前120筆
X_test, y_test = X[120:], y[120:]       # 測試資料後30筆

# 建立Keras的Sequential模型
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(6, activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(3, activation="softmax"))

# 檢視模型參數
model.summary()   # 顯示模型摘要資訊

# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# 訓練模型
print("Training ...")
model.fit(X_train, y_train, epochs=100, batch_size=5)

# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("準確度 = {:.2f}".format(accuracy))

# 儲存Keras模型
print("Saving Model: iris.keras ...")
model.save("iris.keras")
```
### CNN
- 第 8 章 打造你的卷積神經網路
- 8-1 認識 MNIST 手寫數字資料集
- 8-2 使用 MLP 打造 MNIST 手寫辨識
- 8-3 使用 CNN 打造 MNIST 手寫辨識


```python
# 8-2 使用 MLP 打造 MNIST 手寫辨識
import numpy as np
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(7)

# 載入資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 轉換成 2**28 = 784 的向量
X_train = X_train.reshape(X_train.shape[0], 28*28).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28*28).astype("float32")

# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train = X_train / 255
X_test = X_test / 255
# One-hot編碼
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 定義模型
model = Sequential()
model.add(Input(shape=(28*28,)))
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")

# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=10, batch_size=128, verbose=2)

# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

# 顯示圖表來分析模型的訓練過程
import matplotlib.pyplot as plt
# 顯示訓練和驗證損失
loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

```PYTHON
# 8-3 使用 CNN 打造 MNIST 手寫辨識
import numpy as np
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(7)
# 載入資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 將圖片轉換成 4D 張量
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train = X_train / 255
X_test = X_test / 255
# One-hot編碼
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 定義模型
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(16, kernel_size=(5, 5), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=10, batch_size=128, verbose=2)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

# 儲存Keras模型
print("Saving Model: mnist.keras ...")
model.save("mnist.keras")

# 顯示圖表來分析模型的訓練過程
import matplotlib.pyplot as plt

# 顯示訓練和驗證損失
loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

```
