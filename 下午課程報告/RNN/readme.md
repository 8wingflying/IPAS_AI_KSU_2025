## RNN 與 自然語言處理
- NLP
  - https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/ 
- [Text Representation or Text Embedding Techniques(word ==> vector) ](NLP_WordVector.md)
  - https://medium.com/ml-note/word-embedding-3ca60663999d 
- RNN
  - Vanilla RNN
  - LSTM
  - GRU 
- Neural Machine Translation 與 seq2seq model
  - 2014 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
  - 2014[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- 2017 Transformer
  -  
- Pre-Trained Language Models ==> LLM
  - **2018 BERT Bidirectional Encoder Representations from Transforme**
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805v2)
  - 上下文無關模型（如word2vec或GloVe）為詞彙表中的每個單詞生成一個詞向量表示，因此容易出現單詞的歧義問題。
  - BERT考慮到單詞出現時的上下文。例如，詞「水分」的word2vec詞向量在「植物需要吸收水分」和「財務報表裡有水分」是相同的，但BERT根據上下文的不同提供不同的詞向量，詞向量與句子表達的句意有關。
  - BERTology
- NLP|文本分類(TEXT Classofication)
  - [IMDb文本分類](IMDb文本分類.md)
  - Sentiment Analysis
- NLP|文本生成(TEXT Generation) ==> GenAI


## 教科書:第16章
- Generating Shakespearean Text Using a Character RNN
- Sentiment Analysis
- An Encoder–Decoder Network for Neural Machine Translation
- Attention Mechanisms
- An Avalanche of Transformer Models
- Vision Transformers
- Hugging Face’s Transformers Library

## 教科書:第15章Processing Sequences Using RNNs and CNNs ==> 時間序列分析
- Recurrent Neurons and Layers
- Training RNNs
- Forecasting a Time Series
- Handling Long Sequences
