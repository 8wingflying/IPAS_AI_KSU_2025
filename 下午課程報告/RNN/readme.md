## RNN 與 自然語言處理
- NLP
  - https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/ 
- [Text Representation or Text Embedding Techniques(word ==> vector) ](NLP_WordVector.md)
  - https://medium.com/ml-note/word-embedding-3ca60663999d 
- RNN
  - Vanilla RNN
  - LSTM
  - GRU (Gated Recurrent Unit)
  - https://www.geeksforgeeks.org/rnn-vs-lstm-vs-gru-vs-transformers/
- Neural Machine Translation 與 seq2seq model
  - 2014 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
  - 2014 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- 2017 Transformer
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - 基本構建單元 ==> 縮放點積注意力（scaled dot-product attention）單元
- Pre-Trained Language Models ==> LLM
  - **2018 BERT Bidirectional Encoder Representations from Transforme**
  - GOOGLE [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805v2)
  - 上下文無關模型（如word2vec或GloVe）為詞彙表中的每個單詞生成一個詞向量表示，因此容易出現單詞的歧義問題。
  - BERT考慮到單詞出現時的上下文。例如，詞「水分」的word2vec詞向量在「植物需要吸收水分」和「財務報表裡有水分」是相同的，但BERT根據上下文的不同提供不同的詞向量，詞向量與句子表達的句意有關。
  - BERTology
  - OPENAI GPT (Generative Pre-trained Transformer)
    - GPT模型是基於Transformer模型的類神經網路，在大型未標記文字資料集上進行預訓練，並能夠生成類似於人類自然語言的文字。
    - 截至2023年，大多數LLM都具備這些特徵，並廣泛被稱為GPT
    - GPT-1(2018)
    - GPT-4是一個多模態LLM，能夠處理文字和圖像輸入（儘管其輸出僅限於文字）
    - GPT-4o(2024年5月發布)
    - GPT-5 ?? MLLM
  - Google PaLM
  - 2023 Meta AI | LLaMA(Large Language Model Meta AI)(2023年2月)
    - https://zh.wikipedia.org/zh-tw/LLaMA
    - LLaMA 2(2023年7月)
    - Code Llama(2023年8月)
    - Llama 3 (2024年4月18日)
    - Llama 4 (2025年4月5日)
      - 架構已更改為混合專家模型。
      - 具備多模態（文字和圖像輸入，文字輸出）和多語言（12種語言）特性 
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
