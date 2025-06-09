## NLTK主題
- 移除Stop Words（停用詞）
  - 使用NLTK
  - 使用Spacy
  - 使用genism 
- Text Normalization ==> `詞幹提取(stemming)`和`詞形還原(lemmatization)`
  - 使用NLTK
  - 使用Spacy
  - 使用TextBlob
- https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/

#### 移除 Stop Words（停用詞）
- https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/ 
- NLP與資訊檢索中，為節省儲存空間和提高搜尋效率，在自然語言處理資料（或文字）之前或之後會自動過濾掉某些字或詞，這些字或詞即被稱為Stop Words（停用詞）。
  - 停用詞是任何自然語言中最常見的詞。
  - 為了分析文本數據和構建 NLP 模型，這些停用詞可能不會為文檔的含義增加太多價值。
  - 一般來說，文本中最常用的詞是 “the”， “is”， “in”， “for”， “where”， “when”， “to”， “at” 等。 
- nltk.download('stopwords')
```python
# The following code is to remove stop words from sentence using nltk
# Created by - ANALYTICS VIDHYA

# importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
set(stopwords.words('english'))


# sample sentence
text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

# set of stop words
stop_words = set(stopwords.words('english')) 

# tokens of words  
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 



print("\n\nOriginal Sentence \n\n")
print(" ".join(word_tokens)) 

print("\n\nFiltered Sentence \n\n")
print(" ".join(filtered_sentence)) 
```

## Text Normalization
-  ‘ate’, ‘eat’, or ‘eaten’  ==> eat
-  text normalization is a process of transforming a word into a single canonical form.
-  This can be done by two processes, stemming and lemmatization.
-  https://spotintelligence.com/2023/03/25/nlp-feature-engineering/
- 詞幹提取(stemming) ==> NTLK PorterStemmer()
  - 詞幹提取是一種文本規範化技術，它通過考慮單詞中可能找到的常見前綴或後綴清單來截斷單詞的結尾或開頭
  - 這是一個基於規則的基本過程，用於從單詞中去除後綴（“ing”、“ly”、“es”、“s”等）
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

set(stopwords.words('english'))

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

Stem_words = []
ps =PorterStemmer()
for w in filtered_sentence:
    rootWord=ps.stem(w)
    Stem_words.append(rootWord)
print(filtered_sentence)
print(Stem_words)
```
- `詞形還原(lemmatization)`  ==> NTLK WordNetLemmatizer()
  - 詞形還原是獲取單詞詞根形式的有序且逐步的過程。
  - 它使用詞彙 （單詞的詞典重要性） 和形態分析 （單詞結構和語法關係） 
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from nltk.stem import WordNetLemmatizer
set(stopwords.words('english'))

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
print(filtered_sentence) 

lemma_word = []
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
for w in filtered_sentence:
    word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
    word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
    word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
    lemma_word.append(word3)
print(lemma_word)
```
- v 代表動詞，a 代表形容詞，n 代表名詞。
- 詞形還原器僅對那些與 lemmatize 方法的 pos 參數匹配的單詞進行詞形還原。
