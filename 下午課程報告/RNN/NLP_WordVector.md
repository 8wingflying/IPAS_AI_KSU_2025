## NLP_WordVector
- 資料來源
- [Applied Recommender Systems with Python: Build Recommender Systems with Deep Learning, NLP and Graph-Based Techniques](https://learning.oreilly.com/library/view/applied-recommender-systems/9781484289549/)
- [簡中譯本](https://www.tenlong.com.tw/products/9787302657408?list_name=srh)
- https://github.com/Apress/applied-recommender-systems-python
- Ch 3. Content-Based Recommender Systems內容過濾推薦系統
  - 數據收集和下載詞嵌入
  - 將數據導入為DataFrame
  - 預處理數據 
  - 文本(Text)轉為特徵
    - OHE 
    - 詞頻向量器CountVectorizer
    - TF-IDF
    - 詞嵌入
  - 相似性度量
    - 歐幾里得距離 
    - 餘弦相似度 
    - 曼哈頓距離
  - 模型構建
    - 使用CountVectorizer構建模型
    - 使用TF-IDF特徵構建模型
    - 使用Word2vec特徵構建模型
    - 使用fastText特徵構建模型
    - 使用GloVe特徵構建模型
    - 使用共現矩陣構建模型
###
```python
# -*- coding: utf-8 -*-


#  Installing and Importing the Required Libraries


# Commented out IPython magic to ensure Python compatibility.
#Importing the libraries

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from gensim import models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
# %matplotlib inline
from gensim.models import FastText as ft
from IPython.display import Image
import os
```
# Links to download required word embeddings
```
#### download w2v
 gdown https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

#### download glove
 wget https://nlp.stanford.edu/data/glove.6B.zip

#### download fastext
gdown https://drive.google.com/uc?id=1vz6659Atv9OOXiakzj1xaKhZ9jxJkeFF
```
# Importing the data

df = pd.read_csv("Rec_sys_content.csv")
# Viewing Top 5 Rows
df.head(5)

# Data Info
df.info()

# Total Null Values in Data
df.isnull().sum(axis = 0)

# Droping Null Values
df.dropna().reset_index(inplace = True)

# Data Shape
df.shape

df

"""## Loading pretrained models"""

# Importing Word2Vec
word2vecModel = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# # Importing FastText
fasttext_model=ft.load_fasttext_format("cc.en.300.bin.gz")


# # Import Glove
glove_df = pd.read_csv('glove.6B.300d.txt', sep=" ",
                       quoting=3, header=None, index_col=0)
glove_model = {key: value.values for key, value in glove_df.T.items()}

"""## Importing Count Vectorizer and TFIDF"""

# Importing Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')


# Importing IFIDF
tfidf_vec = TfidfVectorizer(stop_words='english', analyzer='word', ngram_range=(1,3))

"""# Preprocessing"""

# Combining Product and Description
df['Description'] = df['Product Name'] + ' ' +df['Description']

# Dropping Duplicates and keeping first record
unique_df = df.drop_duplicates(subset=['Description'], keep='first')

# Converting String to Lower Case
unique_df['desc_lowered'] = unique_df['Description'].apply(lambda x: x.lower())

# Remove Stop special Characters
unique_df['desc_lowered'] = unique_df['desc_lowered'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Coverting Description to List
desc_list = list(unique_df['desc_lowered'])

unique_df= unique_df.reset_index(drop=True)

unique_df.reset_index(inplace=True)
```
# Similarity Measures
"""
## Manhattan distance

It is calculated as the sum of the absolute differences between the two vectors.
![image.png](attachment:image.png)

## Euclidean distance

It is calculated as the square root of the sum of the squared differences between the two vectors.
![image.png](attachment:image.png)

## Cosine similarity

It is the cosine of the angle between two n-dimensional vectors in an n-dimensional space.
It is the dot product of the two vectors divided by the product of the two vectors' lengths (or magnitudes).


![image.png](attachment:image.png)
"""

# Functions for Ranking

def find_euclidean_distances(sim_matrix, index, n=10):

    # Getting Score and Index
    result = list(enumerate(sim_matrix[index]))

    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:10+1]

    # Mapping index with data
    similar_products =  [{'value': unique_df.iloc[x[0]]['Product Name'], 'score' : round(x[1], 2)} for x in sorted_result]

    return similar_products

def find_similarity(cosine_sim_matrix, index, n=10):

    # calculate cosine similarity between each vectors
    result = list(enumerate(cosine_sim_matrix[index]))

    # Sorting the Score
    sorted_result = sorted(result,key=lambda x:x[1],reverse=True)[1:n+1]

    similar_products =  [{'value': unique_df.iloc[x[0]]['Product Name'], 'score' : round(x[1], 2)} for x in sorted_result]

    return similar_products

def find_manhattan_distance(sim_matrix, index, n=10):

    # Getting Score and Index
    result = list(enumerate(sim_matrix[index]))

    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:10+1]

    # Mapping index with data
    similar_products =  [{'value': unique_df.iloc[x[0]]['Product Name'], 'score' : round(x[1], 2)} for x in sorted_result]

    return similar_products
```
# Recommendation functions using various features

# Count Vectorizer
```
"""
CountVectorizer is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text. This is helpful when we have multiple such texts, and we wish to convert each word in each text into vectors. CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is nothing but the count of the word in that particular text sample.
Eg.
document = [ “One Geek helps Two Geeks”, “Two Geeks help Four Geeks”, “Each Geek helps many other Geeks at GeeksforGeeks.”]

![image.png](attachment:image.png)
"""

product_id = 'Vickerman 14" Finial Drop Christmas Ornaments, Pack of 2'

# Comparing similarity to get the top matches using count Vec

def get_recommendation_cv(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    index = list(row.index)[0]
    description = row['desc_lowered'].loc[index]

    #Create vector using Count Vectorizer

    count_vector = count_vectorizer.fit_transform(desc_list)

    if similarity == "cosine":
        sim_matrix = cosine_similarity(count_vector)
        products = find_similarity(sim_matrix , index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(count_vector)
        products = find_manhattan_distance(sim_matrix , index)

    else:
        sim_matrix = euclidean_distances(count_vector)
        products = find_euclidean_distances(sim_matrix , index)

    return products

# Cosine Similarity
get_recommendation_cv(product_id, unique_df, similarity = "cosine", n=10)

# Manhattan Similarity
get_recommendation_cv(product_id, unique_df, similarity = "manhattan", n=10)

# Euclidean Similarity
get_recommendation_cv(product_id, unique_df, similarity = "euclidean", n=10)
```
# tf-Idf
```
"""
TF-IDF (term frequency-inverse document frequency) is a statistical measure that
evaluates how relevant a word is to a document in a collection of documents.
This is done by multiplying two metrics:
how many times a word appears in a document,
and the inverse document frequency of the word across a set of documents.


![image.png](attachment:image.png)

documentA = 'the man went out for a walk'

documentB = 'the children sat around the fire'

![image.png](attachment:image.png)
"""

# Comparing similarity to get the top matches using TF-IDF

def get_recommendation_tfidf(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    index = list(row.index)[0]
    description = row['desc_lowered'].loc[index]

    #Create vector using tfidf

    tfidf_matrix = tfidf_vec.fit_transform(desc_list)

    if similarity == "cosine":
        sim_matrix = cosine_similarity(tfidf_matrix)
        products = find_similarity(sim_matrix , index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(tfidf_matrix)
        products = find_manhattan_distance(sim_matrix , index)

    else:
        sim_matrix = euclidean_distances(tfidf_matrix)
        products = find_euclidean_distances(sim_matrix , index)

    return products

# Cosine Similarity
get_recommendation_tfidf(product_id, unique_df, similarity = "cosine", n=10)

# Manhattan Similarity
get_recommendation_tfidf(product_id, unique_df, similarity = "manhattan", n=10)

# Euclidean Similarity
get_recommendation_tfidf(product_id, unique_df, similarity = "euclidean", n=10)
```
# Word2vec
```
"""
Word2Vec is a shallow, two-layer neural networks which is trained to reconstruct linguistic contexts of words. It takes as its input a large corpus of words and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space. Word2Vec is a particularly computationally-efficient predictive model for learning word embeddings from raw text.

![image.png](attachment:image.png)
"""

# Comparing similarity to get the top matches using Word2Vec pretrained model

def get_recommendation_word2vec(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    input_index = list(row.index)[0]
    description = row['desc_lowered'].loc[input_index]

    #create vectors for each desc using word2vec
    vector_matrix = np.empty((len(desc_list), 300))
    for index, each_sentence in enumerate(desc_list):
        sentence_vector = np.zeros((300,))
        count  = 0
        for each_word in each_sentence.split():
            try:
                sentence_vector += word2vecModel[each_word]
                count += 1
            except:
                continue

        vector_matrix[index] = sentence_vector


    if similarity == "cosine":
        sim_matrix = cosine_similarity(vector_matrix)
        products = find_similarity(sim_matrix , input_index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(vector_matrix)
        products = find_manhattan_distance(sim_matrix , input_index)

    else:
        sim_matrix = euclidean_distances(vector_matrix)
        products = find_euclidean_distances(sim_matrix , input_index)

    return products

# Cosine Similarity
get_recommendation_word2vec(product_id, unique_df, similarity = "cosine", n=10)

# Manhattan Similarity
get_recommendation_word2vec(product_id, unique_df, similarity = "manhattan", n=10)

# Euclidean Similarity
get_recommendation_word2vec(product_id, unique_df, similarity = "euclidean", n=10)
```
### fastText
```
"""
fastText is another word embedding method that is an extension of the word2vec model. Instead of learning vectors for words directly, fastText represents each word as an n-gram of characters. So, for example, take the word, “artificial” with n=3, the fastText representation of this word is <ar, art, rti, tif, ifi, fic, ici, ial, al>, where the angular brackets indicate the beginning and end of the word. This helps capture the meaning of shorter words and allows the embeddings to understand suffixes and prefixes.

![image.png](attachment:image.png)
"""

# Comparing similarity to get the top matches using fastText pretrained model

def get_recommendation_fasttext(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    input_index = list(row.index)[0]
    description = row['desc_lowered'].loc[input_index]

    #create vectors for each description using fasttext
    vector_matrix = np.empty((len(desc_list), 300))
    for index, each_sentence in enumerate(desc_list):
        sentence_vector = np.zeros((300,))
        count  = 0
        for each_word in each_sentence.split():
            try:
                sentence_vector += fasttext_model.wv[each_word]
                count += 1
            except:
                continue

        vector_matrix[index] = sentence_vector

    if similarity == "cosine":
        sim_matrix = cosine_similarity(vector_matrix)
        products = find_similarity(sim_matrix , input_index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(vector_matrix)
        products = find_manhattan_distance(sim_matrix , input_index)

    else:
        sim_matrix = euclidean_distances(vector_matrix)
        products = find_euclidean_distances(sim_matrix , input_index)

    return products

# Cosine Similarity
get_recommendation_fasttext(product_id, unique_df, similarity = "cosine", n=10)

# Manhattan Similarity
get_recommendation_fasttext(product_id, unique_df, similarity = "manhattan", n=10)

# Euclidean Similarity
get_recommendation_fasttext(product_id, unique_df, similarity = "euclidean", n=10)

```
### Glove
```
"""
The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors. But keep in mind that there’s quite a bit of synergy between the GloVe and Word2vec.

![image.png](attachment:image.png)
"""

# Comparing similarity to get the top matches using Glove pretrained model

def get_recommendation_glove(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    input_index = list(row.index)[0]
    description = row['desc_lowered'].loc[input_index]

    #using glove embeddings to create vectors
    vector_matrix = np.empty((len(desc_list), 300))
    for index, each_sentence in enumerate(desc_list):
        sentence_vector = np.zeros((300,))
        count  = 0
        for each_word in each_sentence.split():
            try:
                sentence_vector += glove_model[each_word]
                count += 1

            except:
                continue

        vector_matrix[index] = sentence_vector


    if similarity == "cosine":
        sim_matrix = cosine_similarity(vector_matrix)
        products = find_similarity(sim_matrix , input_index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(vector_matrix)
        products = find_manhattan_distance(sim_matrix , input_index)

    else:
        sim_matrix = euclidean_distances(vector_matrix)
        products = find_euclidean_distances(sim_matrix , input_index)

    return products

# Cosine Similarity
get_recommendation_glove(product_id, unique_df, similarity = "cosine", n=10)

# Manhattan Similarity
get_recommendation_fasttext(product_id, unique_df, similarity = "manhattan", n=10)

# Euclidean Similarity
get_recommendation_fasttext(product_id, unique_df, similarity = "euclidean", n=10)

```
### co-occurrence matrix
```
"""
The purpose of this matrix is to present the number of times each word appears in the same context. .

Roses are red. Sky is blue.

We'll have the following matrix as below:

![image.png](attachment:image.png)
"""

# create cooccurence matrix

#preprocessing
df = df.head(250)

# Combining Product and Description
df['Description'] = df['Product Name'] + ' ' +df['Description']
unique_df = df.drop_duplicates(subset=['Description'], keep='first')
unique_df['desc_lowered'] = unique_df['Description'].apply(lambda x: x.lower())
unique_df['desc_lowered'] = unique_df['desc_lowered'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
desc_list = list(unique_df['desc_lowered'])

co_ocr_vocab = []
for i in desc_list:
    [co_ocr_vocab.append(x) for x in i.split()]

co_occur_vector_matrix = np.zeros((len(co_ocr_vocab), len(co_ocr_vocab)))

for _, sent in enumerate(desc_list):
    words = sent.split()
    for index, word in enumerate(words):
        if index != len(words)-1:
            co_occur_vector_matrix[co_ocr_vocab.index(word)][co_ocr_vocab.index(words[index+1])] += 1

# Comparing similarity to get the top matches using cooccurence matrix

def get_recommendation_coccur(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    input_index = list(row.index)[0]
    description = row['desc_lowered'].loc[input_index]

    vector_matrix = np.empty((len(desc_list), len(co_ocr_vocab)))
    for index, each_sentence in enumerate(desc_list):
        sentence_vector = np.zeros((len(co_ocr_vocab),))
        count  = 0
        for each_word in each_sentence.split():
            try:
                sentence_vector += co_occur_vector_matrix[co_ocr_vocab.index(each_word)]
                count += 1

            except:
                continue

        vector_matrix[index] = sentence_vector/count


    if similarity == "cosine":
        sim_matrix = cosine_similarity(vector_matrix)
        products = find_similarity(sim_matrix , index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(vector_matrix)
        products = find_manhattan_distance(sim_matrix , index)

    else:
        sim_matrix = euclidean_distances(vector_matrix)
        products = find_euclidean_distances(sim_matrix , index)

    return products

# Cosine Similarity
get_recommendation_coccur(product_id, unique_df, similarity = "cosine", n=10)

# Manhattan Similarity
get_recommendation_coccur(product_id, unique_df, similarity = "manhattan", n=10)

# Euclidean Similarity
get_recommendation_coccur(product_id, unique_df, similarity = "euclidean", n=10)

```
