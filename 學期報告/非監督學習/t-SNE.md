# t-distributed stochastic neighbor embedding (t-SNE)
- 論文 [Visualizing data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) | L.v.d. Maaten, **G. Hinton**. |Journal of Machine Learning Research, Vol 9(Nov), pp. 2579—2605. 2008
- [Accelerating t-SNE using tree-based algorithms(2014)](https://jmlr.org/papers/volume15/vandermaaten14a/vandermaaten14a.pdf)
  - Barnes-Hut algorithm
  - dual-tree algorithm 
- https://www.mropengate.com/2019/06/t-sne.html
- https://zhuanlan.zhihu.com/p/148170862

![t-SNE.png](t-SNE.png)

## 原理
- 機器學習_學習筆記系列(78)：t-隨機鄰近嵌入法(t-distributed stochastic neighbor embedding)
- 把原本資料點和點之間的`歐幾里得距離`，改成以`機率`的形式來表示
- 高維分佈(Normal Distribution) ==> 低微分佈(Normal Distribution==> 可改用 t-distribution) 
- t-SNE最小化了兩個分佈之間關於嵌入點位置的Kullback-Leibler（KL)散度
- 使用梯度下降法來解
- scikit-learn implements t-SNE with both exact solutions and the Barnes-Hut approximation.
- https://developer.aliyun.com/article/62946
- https://distill.pub/2016/misread-tsne/
- Kullback-Leibler（KL)散度
- https://ricardokleinklein.github.io/2021/09/05/understanding-kl.html

## 範例學習
- https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/
- [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
```python
class sklearn.manifold.TSNE(
n_components=2, *,
perplexity=30.0,
early_exaggeration=12.0,
learning_rate='auto',
max_iter=None,
n_iter_without_progress=300,
min_grad_norm=1e-07,
metric='euclidean',
metric_params=None,
init='pca',
verbose=0,
random_state=None,
method='barnes_hut',
angle=0.5,
n_jobs=None,
n_iter='deprecated')
```
```python
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

## 使用mnist資料集
mnist = fetch_openml('mnist_784', version=1)

d = mnist.data  
l = mnist.target  

df = pd.DataFrame(d)
df['label'] = l  

print(df.head(4))

## standardize the data
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(df)
print(standardized_data.shape)

##
data_1000 = standardized_data[0:1000, :]
labels_1000 = l[0:1000]

model = TSNE(n_components = 2, random_state = 0)

tsne_data = model.fit_transform(data_1000)

tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data = tsne_data,
     columns =("Dim_1", "Dim_2", "label"))

sn.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2',
               hue='label', palette="bright")
plt.show()
```
## sklearn測試


```python
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

X, y = load_digits(return_X_y=True)

tsne = TSNE()
X_embedded = tsne.fit_transform(X)

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
```

```python
## 經過Gemini 修正數次後的程式

import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
# from sklearn.manifold.t_sne import _joint_probabilities # Remove this line
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

X, y = load_digits(return_X_y=True)

tsne = TSNE()
X_embedded = tsne.fit_transform(X)

# Pass x and y coordinates as keyword arguments
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, legend='full', palette=palette)
```
## 範例學習 手把手實作
- 機器學習_學習筆記系列(78)：t-隨機鄰近嵌入法(t-distributed stochastic neighbor embedding)
```python

## t-隨機鄰近嵌入法(t-distributed stochastic neighbor embedding)
## 引入我們需要的packages
import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

## 載入MNIST Dataset
from sklearn.datasets import load_digits
digits = load_digits()
X=(digits.data/16)
y=digits.target
plt.rcParams["figure.figsize"] = (18,18)
plt.gray() 
for i in range(100):
    plt.subplot(20, 20, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.show() 

## Nearest Neighbor
from scipy.spatial.distance import cdist
def Nearest_Neighbor(X,n_neighbors):
    distance = cdist(X,X,"euclidean")
    neighbors_index=np.argsort(distance,axis=1)[:,1:n_neighbors+1]
    neighbors=np.sort(distance,axis=1)[:,1:n_neighbors+1]
    return neighbors,neighbors_index
## Binary search for sigmas of conditional Gaussians

import math

def _binary_search_perplexity(neighbors,neighbors_index,perplexity,verbose):
    EPSILON_DBL = 1e-8
    PERPLEXITY_TOLERANCE = 1e-5
    n_steps = 100
    n_samples,n_neighbors  = neighbors.shape
    using_neighbors = n_neighbors < n_samples
    beta_sum = 0.0
    desired_entropy = math.log(perplexity)
    P = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in tqdm(range(n_samples)):
        beta_min = -np.Inf
        beta_max = np.Inf
        beta = 1.0
        for l in range(n_steps):
            sum_Pi = 0.0
            P[i,neighbors_index[i]]=np.exp(-neighbors[i]* beta)
            sum_Pi=np.sum(P[i,:])
            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            P[i,:]/=sum_Pi
            sum_disti_Pi=np.sum(P[i,neighbors_index[i]]*neighbors[i])
            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy
            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break
            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.Inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.Inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
        beta_sum += beta
    return P


## Principal Component Anlysis
def PCA(X,n_components,N):
    X_center=X-np.mean(X,axis=0)
    W,D,V=np.linalg.svd(X_center.T)
    X_embedded=np.dot(X_center,W[:,:n_components])
    return X_embedded

## KL Divergence
from scipy.spatial.distance import pdist

def _kl_divergence(X_embedded,P,N,n_components):
    MACHINE_EPSILON_ARRAY=np.ones(N)*np.finfo(np.double).eps
    dist=(cdist(X_embedded,X_embedded,"euclidean")**2+1)**-1
    Q = np.maximum(dist/np.sum(dist.ravel()),MACHINE_EPSILON_ARRAY)
    PQd = (P-Q)*dist
    grad=np.zeros(X_embedded.shape)
    for i in range(N):
        grad[i]=4*np.dot(PQd[i],X_embedded[i]-X_embedded)
    return grad

## Gradient Descent
def gradient_descent(Y,P,gradient,max_iter,learning_rate,momentum):
    for t in tqdm(range(max_iter)):
        if(t<1):
            pre_pre_Y=Y.copy()
            Y=pre_pre_Y-learning_rate*gradient
            pre_Y=Y.copy()
        else:
            Y=pre_Y-learning_rate*gradient+momentum*(pre_Y-pre_pre_Y)
            pre_pre_Y=pre_Y.copy()
            pre_Y=Y.copy()
        gradient=_kl_divergence(Y,P,N,n_components)
    return Y

## t-SNE
N,M=X.shape
n_neighbors=30
n_components=2
neighbors,neighbors_index=Nearest_Neighbor(X,n_neighbors)
early_exaggeration=12.0
MACHINE_EPSILON=np.finfo(np.double).eps
neighbors = neighbors.reshape(N, -1)
neighbors = neighbors.astype(np.float32, copy=False)
conditional_P = _binary_search_perplexity(neighbors,neighbors_index,30, 0)
indptr=np.linspace(0,N*n_neighbors,N+1).astype(int)
P = conditional_P + conditional_P.T
P/=(2*N)
X_embedded=PCA(X,n_components,N)
grad=_kl_divergence(X_embedded,P,N,n_components)
Y=X_embedded.copy()
gradient=grad.copy()
max_iter=750
learning_rate=200
momentum=0.2
X_sub=gradient_descent(Y,P*early_exaggeration,gradient,max_iter,learning_rate,momentum)
HBox(children=(IntProgress(value=0, max=1797), HTML(value='')))
HBox(children=(IntProgress(value=0, max=750), HTML(value='')))

## Plot
color=["#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF",
       "#FF00FF","#FF0088","#FF8800","#00FF99","#7700FF"]
plt.rcParams["figure.figsize"] = (18,18)
for i in range(0,10):
    BOOL=(y==i)
    plt.scatter(X_sub[BOOL,0],X_sub[BOOL,1],c=color[i],label=i)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.show()
```
```python
## COLAB GEMINI 修正版
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

## 載入MNIST Dataset
from sklearn.datasets import load_digits
digits = load_digits()
X=(digits.data/16)
y=digits.target
plt.rcParams["figure.figsize"] = (18,18)
plt.gray()
for i in range(100):
    plt.subplot(20, 20, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.show()

## Nearest Neighbor
from scipy.spatial.distance import cdist
def Nearest_Neighbor(X,n_neighbors):
    distance = cdist(X,X,"euclidean")
    neighbors_index=np.argsort(distance,axis=1)[:,1:n_neighbors+1]
    neighbors=np.sort(distance,axis=1)[:,1:n_neighbors+1]
    return neighbors,neighbors_index
## Binary search for sigmas of conditional Gaussians

import math

def _binary_search_perplexity(neighbors,neighbors_index,perplexity,verbose):
    EPSILON_DBL = 1e-8
    PERPLEXITY_TOLERANCE = 1e-5
    n_steps = 100
    n_samples,n_neighbors  = neighbors.shape
    using_neighbors = n_neighbors < n_samples
    beta_sum = 0.0
    desired_entropy = math.log(perplexity)
    P = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in tqdm(range(n_samples)):
        # Changed np.Inf to np.inf for compatibility with newer NumPy versions
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        for l in range(n_steps):
            sum_Pi = 0.0
            P[i,neighbors_index[i]]=np.exp(-neighbors[i]* beta)
            sum_Pi=np.sum(P[i,:])
            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            P[i,:]/=sum_Pi
            sum_disti_Pi=np.sum(P[i,neighbors_index[i]]*neighbors[i])
            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy
            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break
            # Changed np.Inf to np.inf for compatibility with newer NumPy versions
            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                # Changed np.Inf to np.inf for compatibility with newer NumPy versions
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
        beta_sum += beta
    return P


## Principal Component Anlysis
def PCA(X,n_components,N):
    X_center=X-np.mean(X,axis=0)
    W,D,V=np.linalg.svd(X_center.T)
    X_embedded=np.dot(X_center,W[:,:n_components])
    return X_embedded

## KL Divergence
from scipy.spatial.distance import pdist

def _kl_divergence(X_embedded,P,N,n_components):
    MACHINE_EPSILON_ARRAY=np.ones(N)*np.finfo(np.double).eps
    dist=(cdist(X_embedded,X_embedded,"euclidean")**2+1)**-1
    Q = np.maximum(dist/np.sum(dist.ravel()),MACHINE_EPSILON_ARRAY)
    PQd = (P-Q)*dist
    grad=np.zeros(X_embedded.shape)
    for i in range(N):
        grad[i]=4*np.dot(PQd[i],X_embedded[i]-X_embedded)
    return grad

## Gradient Descent
def gradient_descent(Y,P,gradient,max_iter,learning_rate,momentum):
    for t in tqdm(range(max_iter)):
        if(t<1):
            pre_pre_Y=Y.copy()
            Y=pre_pre_Y-learning_rate*gradient
            pre_Y=Y.copy()
        else:
            Y=pre_Y-learning_rate*gradient+momentum*(pre_Y-pre_pre_Y)
            pre_pre_Y=pre_Y.copy()
            pre_Y=Y.copy()
        gradient=_kl_divergence(Y,P,N,n_components)
    return Y

## t-SNE
N,M=X.shape
n_neighbors=30
n_components=2
neighbors,neighbors_index=Nearest_Neighbor(X,n_neighbors)
early_exaggeration=12.0
MACHINE_EPSILON=np.finfo(np.double).eps
neighbors = neighbors.reshape(N, -1)
neighbors = neighbors.astype(np.float32, copy=False)
conditional_P = _binary_search_perplexity(neighbors,neighbors_index,30, 0)
indptr=np.linspace(0,N*n_neighbors,N+1).astype(int)
P = conditional_P + conditional_P.T
P/=(2*N)
X_embedded=PCA(X,n_components,N)
grad=_kl_divergence(X_embedded,P,N,n_components)
Y=X_embedded.copy()
gradient=grad.copy()
max_iter=750
learning_rate=200
momentum=0.2
X_sub=gradient_descent(Y,P*early_exaggeration,gradient,max_iter,learning_rate,momentum)

## Plot
color=["#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF",
       "#FF00FF","#FF0088","#FF8800","#00FF99","#7700FF"]
plt.rcParams["figure.figsize"] = (18,18)
for i in range(0,10):
    BOOL=(y==i)
    plt.scatter(X_sub[BOOL,0],X_sub[BOOL,1],c=color[i],label=i)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.show()
```

