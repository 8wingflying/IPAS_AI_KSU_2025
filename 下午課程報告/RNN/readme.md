![RNN.png](RNN.png)

## RNN 與 自然語言處理|報告主題
- 需涵蓋底下`1`到`5`內容(未編號者(加分項):可不做)
- `1`.NLP研究主題
  - https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/ 
- Text Preprocessing(文本預處理) | Text Feature Engineering
  - [Text Preprocessing(文本預處理)基本觀念](TextPreprocessing.md)
  - [使用NLTK](IMDb文本分類_NLTK.md)
    - 移除Stop Words（停用詞）
    - Text Normalization ==> `詞幹提取(stemming)`和`詞形還原(lemmatization)`
- `2`.[Text Representation or Text Embedding Techniques(word ==> vector) ](NLP_WordVector.md)
  - https://medium.com/ml-note/word-embedding-3ca60663999d
  - Bag of Word
  - TF-IDF
  - Word Embedding 
- `3`.RNN model  ==> The Problem of Long-Term Dependencies
  - Vanilla RNN
  - LSTM|Long Short-Term Memory|長短期記憶(1997)
    - 【導讀】[Understanding LSTM Networks(2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    - https://ithelp.ithome.com.tw/articles/10193924 
  - GRU | Gated Recurrent Unit(2014)
    - https://zhuanlan.zhihu.com/p/20310288990
    - https://ithelp.ithome.com.tw/articles/10194201 
  - https://www.geeksforgeeks.org/rnn-vs-lstm-vs-gru-vs-transformers/
- `4`.從seq2seq model到Transformer
  - Neural Machine Translation 與 seq2seq model
    - 2014 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
    - 2014 [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)
    - https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/
  - Attention
    - 2014【經典論文】 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
    - 2015 【經典論文】[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
    - 👍【導讀】[Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
    - 【導讀】 ==> 28. 注意力機制（Attention mechanism）
  - 2017 [Transformer](Transformer.md)
    - 👍👍👍【經典論文】[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - 【論文】[[1607.06450] Layer Normalization | Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton](https://arxiv.org/abs/1607.06450)
    - 【導讀】[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
    - 【模型應用】[Transformer_HuggingFace範例](Transformer_HuggingFace範例.md)
- `5`.[Pre-Trained Language Models](Pre-Trained_Language_Models.md) ==> LLM
  - **2018 BERT Bidirectional Encoder Representations from Transforme**
    - 【經典論文】[[1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - 2018 GPT
    - GPT-1
    -  [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf]()
    - GPT-2
    - GPT-3
    - GPT-3.5
    - GPT-4 
  - LLAMA
    - https://zh.wikipedia.org/zh-tw/LLaMA
    - LLAMA-1(2023年2月)
    - LLAMA-2(2023年7月)
    - Code Llama(2023年8月)
    - LLAMA-3(2024年4月18日)
    - Llama-4(2025年4月5日) 

- `5`.範例應用NLP|文本分類(TEXT Classofication)
  - IMDb文本分類
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
    - SimpleRNN
    - LSTM
    - GRU
    - 各種變形LSTM+CNN
    - Keras官方範例[Bidirectional LSTM on IMDB](https://keras.io/examples/nlp/bidirectional_lstm_imdb/)
    - Keras官方範例[Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/)
    - Keras官方範例[Text Classification using FNet](https://keras.io/examples/nlp/fnet_classification_with_keras_hub/)
    - TF官方範例[Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch/)
    - TF官方範例[Text classification with an RNN ](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
    - TF官方範例[Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
  - IMDb文本分類
    - [IMDb文本分類_NLTK](IMDb文本分類_NLTK.md)
    - [IMDb文本分類_文字向量化](IMDb文本分類_文字向量化.md)
    - [IMDb文本分類_傳統作法](IMDb文本分類_傳統作法.md)
      - 模型1 ==> 使用CountVectorizer + MultinomialNB
      - 模型2 ==> 使用TfidfVectorizer + MultinomialNB
      - 模型3 ==> 使用 wordvector + SVC | RandomForestClassifier
    - [IMDb文本分類_RNN](IMDb文本分類_RNN.md)
    - Sentiment Analysis 情緒分析 ==> 負評 vs 正評
  - 以內容為基礎的推薦系統
  - [主題建模Topic Modeling](TopicModeling主題建模.md)

### 其他分析
- [NLP Text Preprocessing Tutorial](https://www.kaggle.com/code/rudraneelsannigrahi/nlp-text-preprocessing-tutorial)

## 教科書相關章節
#### 教科書:第15章Processing Sequences Using RNNs and CNNs ==> [時間序列分析](時間序列分析.md)
- Recurrent Neurons and Layers
- Training RNNs
- Forecasting a Time Series
- Handling Long Sequences

#### 教科書:第16章
- Generating Shakespearean Text Using a Character RNN
- Sentiment Analysis
- An Encoder–Decoder Network for Neural Machine Translation
- Attention Mechanisms
- An Avalanche of Transformer Models
- Vision Transformers
- Hugging Face’s Transformers Library

## 延伸閱讀
- [Natural Language Processing in Action, Second Edition(2025)](https://learning.oreilly.com/library/view/natural-language-processing/9781617299445/)
- [Natural Language Processing with Transformers, Revised Edition(2022)](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
- [Python Natural Language Processing Cookbook - Second Edition(2024)](https://learning.oreilly.com/library/view/python-natural-language/9781803245744/)
- [Practical Natural Language Processing(2020)](https://learning.oreilly.com/library/view/practical-natural-language/9781492054047/)
- 傳統NLP ==> NLTK, spaCy, sklearn, and gensim
  - [Python Natural Language Processing Cookbook(2021)](https://learning.oreilly.com/library/view/python-natural-language/9781838987312/)
  - [Natural Language Processing: Python and NLTK(2016)](https://learning.oreilly.com/library/view/natural-language-processing/9781787285101/) 


