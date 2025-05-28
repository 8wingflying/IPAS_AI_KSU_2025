# Natural Language Processing (NLP) Tutorial
- 自然語言處理 (NLP) 是人工智慧 (AI) 的一個分支，它使機器能夠理解和處理人類語言。
- 人類語言可以採用文字或音訊格式。
- [Natural Language Processing (NLP) Tutorial](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- https://www.geeksforgeeks.org/artificial-intelligence-natural-language-generation/

#### NLP的應用
- Alexa、Siri 和 Google Assistant 等語音助理使用 NLP 進行語音辨識和互動。
- Grammarly、Microsoft Word 和 Google Docs 等工具會應用 NLP 進行語法檢查和文字分析。
- 透過 Google 和 DuckDuckGo 等搜尋引擎擷取資訊。
- 網站機器人和客戶支援聊天機器人利用 NLP 進行自動對話和查詢處理。
- Google 翻譯和類似服務使用 NLP 進行語言之間的即時翻譯。
- 文字摘要
- 文本生成 
#### 自然語言處理階段(Phases of Natural Language Processing)
- 自然語言處理有兩個組成部分：
  - 自然語言理解(Natural Language Understanding)
  - 自然語言生成(Natural Language Generation)

#### 常用函式庫
- https://www.geeksforgeeks.org/nlp-libraries-in-python/
- NLTK (Natural Language Toolkit)
- spaCy
- Gensim
- Stanza(StanfordNLP)
- Hugging Face Transformer
- AllenNLP

#### Text Representation or Text Embedding Techniques in NLP
- One-Hot Encoding
Bag of Words (BOW)
N-Grams
Term Frequency-Inverse Document Frequency (TF-IDF)
N-Gram Language Modeling with NLTK

在 NLP 中規範化文字數據
文字規範化將文字轉換為一致的格式，提高了品質並使其更易於在 NLP 任務中處理。

文字規範化的關鍵步驟包括：

1. 正規表示式（RE）是定義搜尋模式的字元序列。

如何寫正規表示式？
正規表示式的屬性
Python 中的 RegEx
使用 RE 提取電子郵件
2.標記化是將文字拆分成稱為標記的較小單元的過程。

標記文字、句子和單字的工作原理
字標記化
基於規則的標記化
子詞標記化
基於字典的標記化
空格標記
WordPiece 標記化
3.詞形還原將單字還原為其基本形式或字根形式。

4.詞幹提取透過刪除後綴將作品簡化為字根。詞幹擷取器的類型包括：

波特·施泰默
蘭開斯特·施泰默
滾雪球詞幹分析器
洛維斯·施泰默
基於規則的詞幹提取
5.停用詞刪除是從文件中刪除常用詞的過程。

6.詞性（POS）標註根據定義和上下文為句子中的每個單字分配一個詞性。

NLP 中的文字表示或文字嵌入技術
文字表示將文字資料轉換為數值向量，透過以下方法處理：

獨熱編碼
詞袋模型（BOW）
N-Grams
詞頻-逆文檔頻率（TF-IDF）
使用 NLTK 進行 N-Gram 語言建模
文字嵌入技術是指用於建立這些向量表示的方法和模型，包括傳統方法（如 TFIDF 和 BOW）和更高級的方法：

1.詞嵌入

Word2Vec（SkipGram，連續詞袋 - CBOW）
GloVe（用於詞語表示的全域向量）
快速文字
2. 預訓練嵌入

ELMo（語言模型的嵌入）
BERT（來自 Transformer 的雙向編碼器表示）
3.文檔嵌入 - Doc2Vec
