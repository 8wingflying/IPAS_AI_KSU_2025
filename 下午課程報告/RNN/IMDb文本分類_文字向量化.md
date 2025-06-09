![image](https://github.com/user-attachments/assets/2b553914-ba3e-4475-bc27-978c400c6ed8)# 文字向量化 ==> Representing text as numbers
- One-hot encoding (OHE)
- scikit-learn
  - CountVectorizer ==> [](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    - https://blog.csdn.net/weixin_38278334/article/details/82320307 
  - TF-IDF ==> [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    - https://blog.csdn.net/qq_43391414/article/details/112912107 
- word embedding tools.
  - Word2vec
    - Continuous bag-of-words (CBOW)
    - Skip-gram
    - https://serokell.io/blog/word2vec
  - fastText
  - GloVe
- Pretrained model ==> Keras `Embedding` Layers
  - The Embedding layer can be understood as a lookup table that maps from integer indices (which stand for specific words) to dense vectors (their embeddings).
  - The dimensionality (or width) of the embedding is a parameter you can experiment with to see what works well for your problem, much in the same way you would experiment with the number of neurons in a Dense layer.
  - 參看 Tensorflow官方範例 [Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
  - 參看 keras 官方範例 [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/)
  - https://ithelp.ithome.com.tw/articles/10254164
  - embedding_layer = Embedding(200, 32, input_length=50)
    - 200 第一個參數－字彙數目或文章中 unique words 數目。
    - 32 第二個參數－每個字彙向量的維度。
    - 50 第二個參數－每個輸入(input)句子的長度。
    - Embedding()會產生一個2D向量(2D vector)，列代表字彙，行顯示相對應的維度。

# scikit-learn
### CountVectorizer
- https://blog.csdn.net/weixin_38278334/article/details/82320307
### TF-IDF
- [Understanding TF-IDF (Term Frequency-Inverse Document Frequency)](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/)
- https://blog.csdn.net/qq_43391414/article/details/112912107
- 詞頻（TF）：測量單詞在文檔中出現的頻率。
  - 頻率越高，重要性越高。
  - 如果某個術語在文檔中頻繁出現，則它可能與文檔的內容相關。
  - 公式：
  - 單獨使用 TF 的限制：
    - TF 不考慮術語在整個語料庫中的全域重要性。
    - “the”或“and”等常用詞 可能具有較高的 TF 分數，但在區分文檔方面沒有意義。
- 反向文檔頻率 （IDF）：減少多個文檔中常見單詞的權重，同時增加稀有單詞的權重。
  - 如果術語出現在較少的文檔中，則它更有可能有意義和具體。
  - 公式：IDF 公式
    - 對數是用於抑制非常大或非常小的值的影響，確保IDF分數適當縮放。
    - 它還有助於平衡出現在極少或極多文檔中的術語的影響。
  - 單獨使用 IDF 的限制：
    - IDF 不考慮術語在特定文檔中出現的頻率。
    - 一個術語在語料庫中可能很少見（高IDF），但在特定文檔中（低TF）中無關緊要。
- TF-IDF 分數越高，意味著該術語在該特定文檔中更重要

