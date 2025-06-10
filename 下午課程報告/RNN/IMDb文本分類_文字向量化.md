# 文字向量化 ==> Representing text as numbers
- One-hot encoding (OHE)
- scikit-learn
  - CountVectorizer ==> [](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    - https://blog.csdn.net/weixin_38278334/article/details/82320307 
  - TF-IDF ==> [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    - https://blog.csdn.net/qq_43391414/article/details/112912107 
- word embedding tools ==> [Genism](https://radimrehurek.com/gensim/apiref.html#api-reference)
  - 通過對向量的運算，比如歐幾裡得距離或者cosine相似度，可以計算出兩個單詞之間的語義相似性。
  - Word2vec(2013/4) ==> [models.word2vec – Word2vec embeddings](https://radimrehurek.com/gensim/models/word2vec.html)
    - Continuous bag-of-words (CBOW)
    - Skip-gram
    - https://serokell.io/blog/word2vec
    - Gensim 則是 Google 於 2013 提出的 Word2Vec 論文的 Python 實現
  - fastText(2016) ==> [models.fasttext – FastText model](https://radimrehurek.com/gensim/models/fasttext.html)
    - https://radimrehurek.com/gensim/models/fasttext.html
    - [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
    - https://blog.csdn.net/qq_38890412/article/details/104710375
  - GloVe(Global Vectors for Word Representation)(2014)
    - 由 Stanford NLP Group 所發表的一篇論文、同時也是開放出來的一個 Pre-trained 詞嵌入模型
    - [GloVe-Gensim](https://clay-atlas.com/blog/2020/07/01/python-cn-glove-convert-gensim/) ==> 把 GloVe Glove 模型轉換成 Gensim 可以讀取的格式
    - https://clay-atlas.com/blog/2020/07/01/python-cn-glove-convert-gensim/
    - https://zhuanlan.zhihu.com/p/79573970
- Pretrained model ==> Keras `Embedding` Layers
  - The Embedding layer can be understood as a lookup table that maps from integer indices (which stand for specific words) to dense vectors (their embeddings).
  - The dimensionality (or width) of the embedding is a parameter you can experiment with to see what works well for your problem, much in the same way you would experiment with the number of neurons in a Dense layer.
  - [Understanding Word Embeddings with Keras](https://medium.com/@hsinhungw/understanding-word-embeddings-with-keras-dfafde0d15a4)
  - 參看 Tensorflow官方範例 [Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
  - 參看 keras 官方範例 [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/)
  - [在Keras模型中使用預訓練的詞向量](https://keras-cn.readthedocs.io/en/latest/legacy/blog/word_embedding/)
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


## Gensim
- Gensim comes with several already pre-trained models, in the Gensim-data repository
- Colab安裝genism ==> 注意版本匹配
```
# Uninstall existing versions of numpy, scipy, and gensim
!pip uninstall numpy scipy gensim -y

# Install a compatible set of versions
# We are installing numpy==1.23.5 and scipy==1.9.3 as these are known to work well together
# and are within reasonable ranges for gensim compatibility.
!pip install numpy==1.23.5 scipy==1.9.3 gensim
```
- 範例:
```PYTHON
import gensim.downloader
# Show all available models in gensim-data
print(list(gensim.downloader.info()['models'].keys()))
```
```
['fasttext-wiki-news-subwords-300',
'conceptnet-numberbatch-17-06-300',
'word2vec-ruscorpora-300', 'word2vec-google-news-300',
'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200',
'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200',
'__testing_word2vec-matrix-synopsis']
```
- 範例:
```python
import gensim.downloader as api

info = api.info()  # show info about available models/datasets
model = api.load("glove-twitter-25")  # download the model and return as object ready for use
model.most_similar("cat")
```
```
[==================================================] 100.0% 104.8/104.8MB downloaded
[('dog', 0.9590820074081421),
 ('monkey', 0.920357882976532),
 ('bear', 0.9143136739730835),
 ('pet', 0.9108031392097473),
 ('girl', 0.8880629539489746),
 ('horse', 0.8872726559638977),
 ('kitty', 0.8870542049407959),
 ('puppy', 0.886769711971283),
 ('hot', 0.886525571346283),
 ('lady', 0.8845519423484802)]
```
- 範例: load a corpus and use it to train a Word2Vec model
```python
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

corpus = api.load('text8')  # download the corpus and return it opened as an iterable
model = Word2Vec(corpus)  # train a model from the corpus
model.most_similar("car")
```


## Wordvector
- [models.word2vec – Word2vec embeddings](https://radimrehurek.com/gensim/models/word2vec.html)

```python
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")


model = Word2Vec.load("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)

vector = model.wv['computer']  # get numpy vector of a word
sims = model.wv.most_similar('computer', topn=10)  # get other similar words
```


## Fasttext
- https://radimrehurek.com/gensim/models/fasttext.html
```python
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences

print(common_texts[0])

print(len(common_texts))

model = FastText(vector_size=4, window=3, min_count=1)  # instantiate
model.build_vocab(corpus_iterable=common_texts)
model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=10)  # train
```
