### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/
- 基本原理==>https://myapollo.com.tw/blog/dbscan/
- 將資料點分為三種類型：
  - `1`.在指定半徑(eplison)內具有足夠數量的連續點的核心點(CORE)
  - `2`.靠近核心點但缺乏足夠多的鄰居本身作為核心點的邊界點(Reachable points)
  - `3`.不屬於任何集群的噪點(noise)。
- DBSCAN 演算法步驟:
  - `1`.確定核心點：對於數據集中的每個點，計算其 eps 鄰域內的點數。如果計數達到或超過 MinPts，則將該點標記為核心點(CORE)。
  - `2`.形成群集：對於尚未分配給群集的每個核心點，創建一個新群集。遞歸地查找所有密度連接的點，即核心點的 eps 半徑內的點，並將它們添加到集群中。
  - `3`.密度連通性：如果存在一個點鏈，其中每個點都在下一個點的 eps 半徑內，並且鏈中至少有一個點是核心點，則兩個點 a 和 b 是密度連通點。此鏈接過程可確保群集中的所有點都通過一系列密集區域進行連接。
  - `4`.標籤噪點(noise)：處理完所有點后，任何不屬於集群的點都將被標記為噪點。

### 範例學習 ==> iris資料集
- 不要再用K-means！ 超實用分群法DBSCAN詳解
- https://caveofpython.com/machine-learning/clustering-irises-with-dbscan/
```python
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn

iris = load_iris(as_frame=True)

df = iris['data']

X = StandardScaler().fit_transform(df)
y = np.choose(iris['target'], iris['target_names'])

model = DBSCAN(eps=0.75, min_samples=8)
model.fit(X)

y_predicted = model.labels_

plot_x = 0
plot_y = 2

x_col = df.columns[plot_x]
y_col = df.columns[plot_y]

fig, axes = plt.subplots(ncols=2)

fig.suptitle("Iris Flower DBSCAN Clustering")

axes[0].set_xlabel(x_col)
axes[0].set_ylabel(y_col)
axes[1].set_xlabel(x_col)
axes[1].set_ylabel(y_col)

sn.scatterplot(data=df, x=x_col, y=y_col, hue=y, palette='husl', ax=axes[0])
sn.scatterplot(data=df, x=x_col, y=y_col, hue=y_predicted, palette='husl', ax=axes[1])

plt.show()
```
