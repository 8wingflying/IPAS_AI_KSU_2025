# BERT 
- BERT（Bidirectional Encoder Representations from Transformers）是由Google發佈的預訓練語言模型。
- BERT的架構基於Transformer的編碼器部分，採用多層堆疊的方式。
- BERT模型沒有使用解碼器層，因此沒有遮罩多頭注意力子層。
- BERT的訓練包括兩個主要任務：遮罩語言建模（Masked Language Modeling, MLM）和下一句預測（Next Sentence Prediction, NSP）。
- BERT-base模型包含12層編碼器，BERT-large模型包含24層編碼器。
- BERT的輸入包括特殊的[CLS]和[SEP]標記，用於分類和句子分隔。
- BERT的預訓練過程使用了大規模無監督語料，之後可以通過微調（fine-tuning）應用於各種下游NLP任務。
- [[1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## 導讀
- https://zhuanlan.zhihu.com/p/403495863
- https://zhuanlan.zhihu.com/p/52282552
- https://medium.com/@pocheng0118/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-d751cdbaff02
- https://www.woshipm.com/ai/6223371.html
- https://nlp.stanford.edu/seminar/details/jdevlin.pdf

## 應用
