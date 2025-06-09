# Transformer_範例
- 使用預訓練模型
  - KERAS 範例
    - [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/)
    - [Named Entity Recognition using Transformers](https://keras.io/examples/nlp/ner_transformers/)
    - [Text classification with Switch Transformer](https://keras.io/examples/nlp/text_classification_with_switch_transformer/)
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

## 參考資料
- [Hugging Face’s notebooks](https://huggingface.co/docs/transformers/notebooks)
  - 有一堆範例程式
  - 入門推薦 [快速入門Quick tour](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/quicktour.ipynb)
- [Natural Language Processing with Transformers, Revised Edition](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
  - https://github.com/nlp-with-transformers/notebooks
  - CH1

### 範例1:情感分析(Sentiment analysis)
- https://fancyerii.github.io/2020/07/08/huggingface-transformers/
- !pip install transformers
```PYTHON
from transformers import pipeline
classifier = pipeline("sentiment-analysis")

# 可使用PROXY
# classifier = pipeline('sentiment-analysis', proxies={"http": "http://localhost:1080"})

# 也可一次预测多个结果：
results = classifier(["We are very happy to show you the 🤗 Transformers library.",
           "We hope you don't hate it."])

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```
```
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use cpu
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```
#### 範例2:text-classification文本分類

```python

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

#hide_output
from transformers import pipeline

classifier = pipeline("text-classification")

import pandas as pd

outputs = classifier(text)
pd.DataFrame(outputs)  
```

#### 範例3: 命名實體識別(Name entity recognition)：識別文字中出現的人名地名的命名實體
```python
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)    
```


#### 範例4:問答(Question answering)：給定一段文本以及針對它的一個問題，從文本中抽取答案
```python
reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])    
```


#### 範例5:text Summarization(文本摘要)
```python
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```


#### 範例6: machine Translation(機器翻譯)
```python
translator = pipeline("translation_en_to_de", 
                      model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])
```

#### 範例7:Text Generation(文本生成)
```python

#hide
from transformers import set_seed
set_seed(42) # Set the seed to get reproducible results

generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
```


