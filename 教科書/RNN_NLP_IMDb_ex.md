# RNN_NLP_IMDb
- MLP
- CNN
- SimpleRNN
- LSTM
- GRU
- 各種變形

###
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
###
```python

```

###
```python

```
###
```python

```
###
```python

```
###
```python

```
###
```python

```
###
```python

```
###
```python

```
###
```python

```
###
```python

```
###
```python

```


