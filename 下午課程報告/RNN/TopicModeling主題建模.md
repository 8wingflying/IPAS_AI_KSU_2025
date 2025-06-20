# Topic Modeling 主題建模
- 瞭解主題建模
- 主題建模的重要性
- 主題模型如何運作？
- 主題建模技術的類型
  - 潛在語義分析 （LSA）
  - 潛在狄利克雷分配 （LDA）
  - BERTopic
- 如何實現主題建模？
- 主題建模的應用
## 導讀
- https://www.geeksforgeeks.org/what-is-topic-modeling/

## 範例
```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
```
## 範例
- Hands-On-Large-Language-Models/Chapter 5 - Text Clustering and Topic Modeling
```python


```
