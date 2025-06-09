##
- 移除Stop Words（停用詞）
  - 使用NLTK
  - 使用Spacy
  - 使用genism 
- Text Normalization
#### 移除 Stop Words（停用詞）
- https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/ 
- NLP與資訊檢索中，為節省儲存空間和提高搜尋效率，在自然語言處理資料（或文字）之前或之後會自動過濾掉某些字或詞，這些字或詞即被稱為Stop Words（停用詞）。
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
