# Transformer_範例
- 使用預訓練模型
- 微調預訓練模型

### 使用預訓練模型最簡單的方法就是使用pipeline函數：
- 情感分析(Sentiment analysis)：一段文本是正面還是負面的情感傾向
- 文本生成(Text generation)：給定一段文本，讓模型補充後面的內容
- 命名實體識別(Name entity recognition)：識別文字中出現的人名地名的命名實體
- 問答(Question answering)：給定一段文本以及針對它的一個問題，從文本中抽取答案
- 填詞(Filling masked text)：把一段文字的某些部分mask住，然後讓模型填空
- 摘要(Summarization)：根據一段長文本中生成簡短的摘要
- 翻譯(Translation)：把一種語言的文字翻譯成另一種語言
- 特徵提取(Feature extraction)：把一段文字用一個向量來表示

### 情感分析(Sentiment analysis)
- https://fancyerii.github.io/2020/07/08/huggingface-transformers/
```PYTHON
from transformers import pipeline
classifier = pipeline('sentiment-analysis')

# 可使用PROXY
# classifier = pipeline('sentiment-analysis', proxies={"http": "http://localhost:1080"})


# 也可一次预测多个结果：

results = classifier(["We are very happy to show you the 🤗 Transformers library.",
           "We hope you don't hate it."])

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```
