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

## BERTopic
- https://maartengr.github.io/BERTopic/index.html
- https://blog.csdn.net/m0_52069102/article/details/145950351
- https://github.com/MaartenGr/BERTopic
- https://towardsdatascience.com/meet-bertopic-berts-cousin-for-advanced-topic-modeling-ea5bf0b7faa3/
- 在Python中使用BERTopic進行機器學習輔助主題建模
- https://geekdaxue.co/read/thinkdot@tech/7f0425a8a61e7d728ce2de027734ca96
- https://zhuanlan.zhihu.com/p/693625240

## 範例
- https://github.com/MaartenGr/BERTopic
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
