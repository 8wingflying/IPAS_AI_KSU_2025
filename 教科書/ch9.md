#

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

#### Gaussian Mixtures

#### Anomaly Detection Using Gaussian Mixtures
#### Selecting the Number of Clusters

#### Bayesian Gaussian Mixture Models

# 作業 ==> Olivetti faces dataset
- [Olivetti Faces - Kaggle](https://www.kaggle.com/datasets/sahilyagnik/olivetti-faces)

#### 作業
- 11. Using Clustering as Preprocessing for Classification
- 13. Using Dimensionality Reduction Techniques for Anomaly Detection
