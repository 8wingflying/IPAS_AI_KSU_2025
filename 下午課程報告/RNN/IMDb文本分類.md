# RNN_NLP_IMDb
- [IMDB Dataset of 50K Movie Reviews|Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  - IMDB 數據集具有 50K 條電影評論，用於自然語言處理或文本分析。
  - 這是一個用於二元情感分類的數據集，包含的數據比以前的基準數據集多得多。
  - 一組 25,000 篇高度極性的電影評論用於培訓，25,000 篇用於測試。
  - 因此，使用分類或深度學習演算法預測正面和負面評論的數量。
  - 有關數據集的更多資訊，請訪問以下連結 http://ai.stanford.edu/~amaas/data/sentiment/
- [Towards Tensorflow 2.0 系列](https://ithelp.ithome.com.tw/users/20119971/ironman/2254?page=1)
- 傳統做法: word2vec | tfidf | bow
  - https://www.kaggle.com/code/jagarapusiva/imdb-movie-reviews-word2vec-tfidf-bow
- MLP
- CNN
- SimpleRNN
- LSTM
- GRU
- 各種變形LSTM+CNN
- 官方程式碼[Text classification with an RNN ](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
- 官方程式碼[Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)

### 其他分析
- [NLP Text Preprocessing Tutorial](https://www.kaggle.com/code/rudraneelsannigrahi/nlp-text-preprocessing-tutorial)
### 載入資料
```python
from keras.datasets import imdb

# 載入 IMDb 資料集, 如果是第一次載入會自行下載資料集
top_words = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# 形狀
print("X_train: (",len(X_train),",",len(X_train[0]), ")")
print("y_train: (",len(y_train),",)")
print("X_test: (",len(X_test),",",len(X_test[0]), ")")
print("y_test: (",len(y_test),",)")
print("--------------------------")

# 顯示 Numpy 陣列內容
print(X_train[0])
print("----------------")
print(y_train[0])   # 標籤資料
```
### 預處理 ==>[tf.keras.utils.pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences#:~:text=This%20function%20transforms%20a%20list%20(of%20length%20num_samples),length%20of%20the%20longest%20sequence%20in%20the%20list.)
```
tf.keras.utils.pad_sequences(
    sequences,
    maxlen=None,
    dtype='int32',
    padding='pre',
    truncating='pre',
    value=0.0
)
```
```python
from keras.datasets import imdb
from keras.utils import pad_sequences

# 載入 IMDb 資料集, 如果是第一次載入會自行下載資料集
top_words = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data( num_words=top_words)

max_words = 100
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
```
### 預處理 ==>Embedding  [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
```python
tf.keras.layers.Embedding(
    input_dim,
    output_dim,
    embeddings_initializer='uniform',
    embeddings_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    weights=None,
    lora_rank=None,
    **kwargs
)
```
### MLP
```python
import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, Flatten

np.random.seed(10)  # 指定亂數種子
# 載入 IMDb 資料集
top_words = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# 資料預處理
max_words = 100
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

# 定義模型
model = Sequential()
model.add(Input(shape=(max_words, )))
model.add(Embedding(top_words, 32))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")

# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=["accuracy"])

# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)
print("--------------------------")

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
```

### SIMPLERNN/LSTM/GRU
```python
import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, SimpleRNN,  LSTM, GRU

np.random.seed(7)  # 指定亂數種子
# 載入 IMDb 資料集
top_words = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# 資料預處理
max_words = 100
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

# 定義模型
model = Sequential()
model.add(Input(shape=(max_words, )))
model.add(Embedding(top_words, 32))
model.add(Dropout(0.25))
model.add(SimpleRNN(32, return_sequences=False,activation="tanh"))
# model.add(LSTM(32, return_sequences=False,activation="tanh"))
# model.add(GRU(32, return_sequences=False,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")

# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="rmsprop",  metrics=["accuracy"])
### optimizer ==> adam, rmsprop

# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)
print("--------------------------")

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))


# 顯示訓練和驗證損失圖表
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 儲存模型結構和權重
model.save("imdb_rnn.keras")
```
### LSTM +CNN
```python
import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D

np.random.seed(7)  # 指定亂數種子

# 載入 IMDb 資料集
top_words = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# 資料預處理
max_words = 500
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)


# 定義模型LSTM+CNN
model = Sequential()
model.add(Input(shape=(max_words, )))
model.add(Embedding(top_words, 32))
model.add(Dropout(0.25))
model.add(Conv1D(filters=32, kernel_size=3, padding="same",activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=False,
               activation="tanh"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")

# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 訓練模型
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)
print("--------------------------")

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))


# 顯示訓練和驗證損失圖表
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```


