# IMDb文本分類_傳統作法

```python
import pandas as pd
import numpy as np

## 載入檔案
messages=pd.read_csv()

messages.head()


import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus=[]
for i in range(len(messages)):
    string=re.sub('[^a-zA-Z]',' ',messages['review'][i])
    string=string.lower()
    string=string.split()
    string=[ps.stem(word) for word in string if not word in stopwords.words('english') ]
    string=' '.join(string)
    corpus.append(string)

## 模型1 ==> 使用CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
y=le.fit_transform(messages['sentiment'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB().fit(X_train,y_train)

y_pred=nb.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))


## 模型2 => 使用TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=2500)
X=tfidf.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
nbt=MultinomialNB().fit(X_train,y_train)
y_pred=nbt.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))

## 模型2 =>使用wordvector

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

import nltk
nltk.download('wordnet')

import nltk
nltk.download('omw-1.4')

!pip install gensim

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['review'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[0]

from nltk import sent_tokenize
from gensim.utils import simple_preprocess

words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))

words

model3.epochs
model3.corpus_count
model3.wv.similar_by_word('art')

def avg_word2vec(doc):
    return np.mean([model3.wv[word] for word in doc if word in model3.wv.index_to_key], axis=0)

!pip install tqdm

from tqdm import tqdm
X3 = []
for i in tqdm(range(len(words))):
    X3.append(avg_word2vec(words[i]))

type(X3)

X_new=np.array(X3)

X_new.shape

print(X_new[3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
model4 = SVC(kernel='rbf', random_state=0).fit(X_train, y_train)

from sklearn.metrics import accuracy_score,classification_report
y_pred = model4.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_pred,y_test))

from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 150, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10,300,10)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)

rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
### fit the randomized model
rf_randomcv.fit(X_train,y_train)

rf_randomcv.best_params_

best_random_grid=rf_randomcv.best_estimator_

from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred=best_random_grid.predict(X_test)
print(metrics.confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))
```
