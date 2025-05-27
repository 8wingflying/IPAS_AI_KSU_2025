## Hierarchical clustering(階層式分群)
- [Hierarchical Clustering in Machine Learning](https://www.geeksforgeeks.org/hierarchical-clustering/)
  - 階層式分群法（Hierarchical Clustering）==>透過階層架構的方式，將資料一層層地反覆進行分裂或者聚合，來產生最後的樹狀結構
  - 常見的方式有聚合式階層和分裂式階層分群法兩種分群法。
    - `1`.Agglomerative Clustering(聚合式階層) ==> bottom-up approach or hierarchical agglomerative clustering (HAC).
    - `2`.Divisive clustering(分裂式階層分群法) ==> top-down approach

#### 範例學習
```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])


clustering = AgglomerativeClustering(n_clusters=4).fit(X)

print(clustering.labels_)
```
