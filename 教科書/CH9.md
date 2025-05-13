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
```

```
```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, n_init=3, random_state=42)
minibatch_kmeans.fit(X)

```

#### Finding the optimal number of clusters
```python

```

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
