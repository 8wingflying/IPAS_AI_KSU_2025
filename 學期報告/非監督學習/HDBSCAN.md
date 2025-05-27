# HDBSCAN
- 論文 []()
- https://blog.csdn.net/zhixiting5325/article/details/90021024
- [How HDBSCAN Works¶](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

####
- [sklearn.cluster.HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
#### 範例學習
- [Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)]()
- pip install hdbscan
```python
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 產生樣本
X, y_true = make_blobs(n_samples=1000, centers=5,
                       cluster_std=0.6, random_state=42)


# Initialize HDBSCAN with additional parameters
clusterer = hdbscan.HDBSCAN(min_cluster_size=5,
                            min_samples=5,
                            cluster_selection_method='eom',
                            allow_single_cluster=True,
                            metric='euclidean',
                            algorithm='best',
                            leaf_size=30)
# Fit the model to the data
clusterer.fit(X)

# Evaluate clustering
labels = clusterer.labels_
silhouette_avg = silhouette_score(X, labels)
ari = adjusted_rand_score(y_true, labels)

# Evaluation metrics
print("Silhouette Coefficient: {:.2f}".format(silhouette_avg))
print("Adjusted Rand Index: {:.2f}".format(ari))

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_,
            cmap='viridis', s=50, alpha=0.7, edgecolors='k')
plt.colorbar()
plt.title('HDBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```
