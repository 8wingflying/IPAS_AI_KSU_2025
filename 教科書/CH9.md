# CH9
- Clustering
- GMM

## Clustering
- https://scikit-learn.org/stable/modules/clustering.html
- [sklearn.cluster模組](https://scikit-learn.org/stable/api/sklearn.cluster.html)
- Cluster quality metrics
  - 範例 :https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py 
  - from sklearn import metrics
    - metrics.homogeneity_score,
    - metrics.completeness_score,
    - metrics.v_measure_score,
    - metrics.adjusted_rand_score,
    - metrics.adjusted_mutual_info_score
    - metrics.silhouette_score
  - https://zhuanlan.zhihu.com/p/609722957
  - https://www.geeksforgeeks.org/homogeneity_score-using-sklearn-in-python/

### Classification vs Clustering
- 使用Classification演算法
- 使用Clustering演算法 ==> 三個cluster 
```python
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)

mapping = {}
for class_id in np.unique(y):
    mode, _ = stats.mode(y_pred[y==class_id])
    mapping[mode] = class_id

y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(X[y_pred==0, 2], X[y_pred==0, 3], "yo", label="Cluster 1")
plt.plot(X[y_pred==1, 2], X[y_pred==1, 3], "bs", label="Cluster 2")
plt.plot(X[y_pred==2, 2], X[y_pred==2, 3], "g^", label="Cluster 3")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="upper left")
plt.grid()
plt.show()
```
### K-means
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)
```
class sklearn.cluster.KMeans
(n_clusters=8, *,
init='k-means++',  ==>
n_init='auto',
max_iter=300,
tol=0.0001,
verbose=0,
random_state=None,
copy_x=True,
algorithm='lloyd')
```
- 參數調叫與
  - Centroid initialization methods
    - [k-means++: The Advantages of Careful Seeding|David Arthur ∗ Sergei Vassilvitskii†](http://ilpubs.stanford.edu:8090/778/)
  - Accelerated K-Means
    - [Using the Triangle Inequality to Accelerate](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf) 
  - Mini-Batch K-Means
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans 

- [Notice of Violation of IEEE Publication Principles: K-means versus k-means ++ clustering technique](https://ieeexplore.ieee.org/abstract/document/6199061)

### K-means 範例
- [sklearn.datasets](https://scikit-learn.org/stable/api/sklearn.datasets.html)
  - 資料載入器(Loaders)有許多 load_XXX()   fetch_YYY()
    - load_iris
    - fetch_olivetti_faces 
  - 樣本產生器(Sample generators) ==> make_XXXX()
    - make_moons ==> Make two interleaving half circles. 
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# extra code – the exact arguments of make_blobs() are not important
blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                         [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std,
                  random_state=7)

k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

## 訓練
y_pred = kmeans.fit_predict(X)

## 展示 5 centroids (i.e., cluster centers) 
kmeans.cluster_centers_

## 預測
import numpy as np

X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)
```
- Figure 9-3. k-means decision boundaries (Voronoi tessellation)
- Hard Clustering vs Soft Clustering
- 演算法複雜度
- Fig 9-5 ==> 不同初始化(質心) 會有不同行為
- 如何選擇初始化(質心 Centroid initialization methods)??  ==> 使用inertia

```python
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
```
```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)
minibatch_kmeans.fit(X)

```

#### Finding the optimal number of clusters
- 方法1:劃出inertia與k的關係 ==> elbow Fig 9-8
- 方法2:使用silhouette_score ==> elbow Fig 9-9
- 方法3:silhouette diagram ==> Figure 9-10
```python
from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)
```
## k-means應用
- k-means應用1:圖像分割(Image Segmentation)
- k-means應用2:半監督學習

#### DBSCAN
```python

```

## 高斯混合模型（Gaussian Mixture Model， GMM）
- 高斯混合模型（Gaussian Mixture Model， GMM）
- 原理:[Gaussian Mixture Models (GMMs)](https://www.datasciencebase.com/unsupervised-ml/probabilistic-models/gausian-mixture-models-gmms)
- https://medium.com/@juanc.olamendy/understanding-gaussian-mixture-models-a-comprehensive-guide-df30af59ced7
- 機器學習_學習筆記系列(92)：高斯混合模型(Gaussian Mixture Models)
- [2.1. Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html)
- [sklearn.mixture](https://scikit-learn.org/stable/api/sklearn.mixture.html)
  - GaussianMixture ==> Gaussian Mixture
  - BayesianGaussianMixture ==>Variational Bayesian estimation of a Gaussian mixture.
- k-means vs Gaussian Mixture Model

### Gaussian Mixtures model
- GMM = n_components 個Guassian model渾成的模型
- 模型訓練 ==> Expectation-Maximization (EM) Algorithm
  - Expectation 步驟
  - Maximization 步驟
- [How GMM Works](https://www.geeksforgeeks.org/gaussian-mixture-model/)
  - Initialization:從每個高斯分佈的均值、協方差和混合係數的初始猜測開始。
  - E-step:對於每個數據點，計算它屬於每個高斯分佈（集群）的概率
  - M-step:使用在 E 步中計算的概率更新參數（均值、協方差、混合係數）。
  - Repeat:繼續在 E 步和 M 步之間交替，直到數據的對數似然（衡量模型與數據擬合程度的度量）收斂。
- 高斯混合模型 （GMM） 的優勢
  - 靈活的集群形狀：與假設球形集群的 K-Means 不同，GMM 可以對具有任意形狀的集群進行建模。
  - 軟分配：GMM 為每個數據點分配一個概率，使其屬於每個聚類，而 K-Means 將每個點分配給一個聚類。
  - 處理重疊數據：當集群重疊或具有不同的密度時，GMM 表現良好。由於它使用概率分佈，因此它可以將一個點分配給具有不同概率的多個聚類
- GMM 的局限性
  - 計算複雜性：GMM 的計算成本往往很高，尤其是對於大型數據集，因為它需要像期望最大化 （EM） 演算法這樣的反覆運算過程來估計參數。
  - 選擇集群數量：與其他集群方法一樣，GMM 要求您預先指定集群數量
    - 貝葉斯資訊準則 （BIC） 和 Akaike 資訊準則 （AIC） 等方法可以幫助根據數據選擇最佳聚類數
####
```
class sklearn.mixture.GaussianMixture(n_components=1,
*,
covariance_type='full', ==>covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
   String describing the type of covariance parameters to use. Must be one of:
   ‘full’: each component has its own general covariance matrix.
   ‘tied’: all components share the same general covariance matrix.
   ‘diag’: each component has its own diagonal covariance matrix.
   ‘spherical’: each component has its own single variance.
tol=0.001,
reg_covar=1e-06,
max_iter=100,
n_init=1,  ==> 記得調高點 ==>  n_init=10
init_params='kmeans',
weights_init=None,
means_init=None,
precisions_init=None,
random_state=None,
warm_start=False,
verbose=0,
verbose_interval=10)
```
```python
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

## 檢視 EM演算法估計的參數
gm.weights_

gm.means_

gm.covariances_

gm.converged_

gm.n_iter_

## 預測 ==> 使用該模型來預測每個實例屬於哪個聚類（硬聚類）或它來自每個聚類的機率

## Hard Clustering
gm.predict(X)

## Soft Clustering
gm.predict_proba(X).round(3)

## 評估指標
gm.aic(X)

gm.bic(X)
```
- GMM ==> IRIS
  - https://github.com/ParthShethSK/GMM-Iris-Clustering 
- 應用:Anomaly Detection ==> Anomaly Detection Using Gaussian Mixtures
  - Anomaly Detection | Outlier detection  ==> 評估指標另訂 
    - 正常 99.99%  Anomaly:0.01%
- 關鍵主題:選擇聚類數量Selecting the Number of Clusters


#### 應用:異常檢測
- 高斯混合可用於異常檢測==>位於低密度區域的實例可視為異常
- 必須定義要使用的密度閾值。
- 在一家試圖檢測缺陷產品的製造公司中，缺陷產品的比例通常是眾所周知的。假設它等於 2%，那麼您可以將密度閾值設定為導致 2% 的實例位於低於該閾值密度的區域的值：
- 使用score_samples()方法==>估計任何位置的密度的對數：
```python
# 計算密度
densities = gm.score_samples(X)
# 設定密度的閥值
density_threshold = np.percentile(densities, 2)
# 定義anomaly(異常)
anomalies = X[densities < density_threshold]
```

#### 關鍵主題:選擇聚類數量Selecting the Number of Clusters(k)
- 不能使用 inertia 或 the silhouette score(輪廓分數)==>因為它們都假設cluster是`球形`。
- 使用最小化理論資訊準則的模型：
  - Bayesian Information Criterion (BIC)貝葉斯資訊準則
    - https://en.wikipedia.org/wiki/Bayesian_information_criterion 
  - the Akaike Information Criterion (AIC) 赤池資訊準則
    - https://en.wikipedia.org/wiki/Bayesian_information_criterion 
- 如何選擇聚類數量Selecting the Number of Clusters(k)? ==> aic或bic最小時的k see Fig-9.20

### Bayesian Gaussian Mixture Models
- GaussianMixture ==> 需手動搜尋最佳聚類數量Selecting the Number of Clusters(k)
- 使用BayesianGaussianMixture能夠為不必要的聚類賦予等於（或接近）零的權重的類別。
- 只需將組件數量設定為您認為大於最佳聚類數量的值，演算法就會自動消除不必要的聚類
```python
class sklearn.mixture.BayesianGaussianMixture(*,
n_components=1,
covariance_type='full',
tol=0.001,
reg_covar=1e-06,
max_iter=100,
n_init=1,
init_params='kmeans',
weight_concentration_prior_type='dirichlet_process',==>
weight_concentration_prior=None,
mean_precision_prior=None,
mean_prior=None,
degrees_of_freedom_prior=None,
covariance_prior=None,
random_state=None,
warm_start=False,
verbose=0,
verbose_interval=10)
```

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
bgm.weights_.round(2)
```
# 作業 ==> Olivetti faces dataset
- [Olivetti Faces - Kaggle](https://www.kaggle.com/datasets/sahilyagnik/olivetti-faces)

#### 作業
- 11. Using Clustering as Preprocessing for Classification
- 13. Using Dimensionality Reduction Techniques for Anomaly Detection
