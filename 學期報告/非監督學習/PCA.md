#

#### 原理
- 機器/統計學習:主成分分析(Principal Component Analysis, PCA)
- https://ithelp.ithome.com.tw/articles/10211877
- https://en.wikipedia.org/wiki/Principal_component_analysis
- https://leemeng.tw/essence-of-principal-component-analysis.html
- https://www.geeksforgeeks.org/principal-component-analysis-pca/
- https://www.geeksforgeeks.org/recovering-feature-names-of-explainedvarianceratio-in-pca-with-sklearn/
- [Difference between PCA VS t-SNE](https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/)
#### 範例學習:PCA vs Kernel Principal Component Analysis (KPCA)
- KPCA is a technique used in machine learning for nonlinear dimensionality reduction

```python

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

## 產生資料集
X, y = make_moons(n_samples=500, noise=0.02, random_state=417)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

## PCA分析
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.title("PCA")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

## KernelPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

plt.title("Kernel PCA")
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.show()
```
- In the kernel space the two classes are linearly separable.
- Kernel PCA uses a kernel function to project the dataset into a higher-dimensional space, where it is linearly separable. Finally, we applied the kernel PCA to a non-linear dataset using scikit-learn.
