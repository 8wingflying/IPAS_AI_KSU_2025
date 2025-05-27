## Hierarchical clustering(階層式分群)
- [Hierarchical Clustering in Machine Learning](https://www.geeksforgeeks.org/hierarchical-clustering/)
  - 階層式分群法（Hierarchical Clustering）==>透過階層架構的方式，將資料一層層地反覆進行分裂或者聚合，來產生最後的樹狀結構
  - 常見的方式有聚合式階層和分裂式階層分群法兩種分群法。
    - `1`.Agglomerative Clustering(聚合式階層) ==> bottom-up approach or hierarchical agglomerative clustering (HAC).
    - `2`.Divisive clustering(分裂式階層分群法) ==> top-down approach
  - [Difference Between Agglomerative clustering and Divisive clustering](https://www.geeksforgeeks.org/difference-between-agglomerative-clustering-and-divisive-clustering/)
#### 範例學習1.Agglomerative Clustering(聚合式階層)
```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])


clustering = AgglomerativeClustering(n_clusters=4).fit(X)

print(clustering.labels_)
```
#### 範例學習2.Divisive clustering(分裂式階層分群法)
```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

Z = linkage(X, 'ward') # Ward Distance

dendrogram(Z) #plotting the dendogram

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()
```
#### 範例學習3.
- https://www.geeksforgeeks.org/implementing-agglomerative-clustering-using-sklearn/
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

## 載入資料集
data = pd.read_csv('CC_GENERAL.csv')

data.drop('CUST_ID', axis=1, inplace=True)

data.fillna(method='ffill', inplace=True)

print(data.head())

## 第 3 步：預處理數據

## 縮放==>使要素具有可比性==>這很重要，因為聚類取決於距離
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

## 規範化有助於聚類分析演算法更好地工作
X_normalized = normalize(X_scaled)

X_normalized = pd.DataFrame(X_normalized)

## 第4步：降低數據的維度
## 使用 PCA 將許多列特徵減少到只有 2 個

pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']


## 第 5 步：製作樹狀圖
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))

## 步驟 6：對 k 的不同值應用凝聚聚類
for k in range(2, 7):  # Try values from 2 to 6
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X_pca)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca['P1'], X_pca['P2'], c=labels, cmap='rainbow')
    plt.title(f'Agglomerative Clustering (k={k})')
    plt.xlabel('P1')
    plt.ylabel('P2')
    plt.show()
## 第 7 步：評估模型並可視化結果
## Silhouette 分數告訴我們數據的分組程度。分數越高，模型越好。
k = [2, 3, 4, 5, 6]

silhouette_scores = []
silhouette_scores.append(
        silhouette_score(X_principal, ac2.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac3.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac4.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac5.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac6.fit_predict(X_principal)))

plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()
```
