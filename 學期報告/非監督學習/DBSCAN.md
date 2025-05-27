### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)


### 範例學習 ==> iris資料集
- 不要再用K-means！ 超實用分群法DBSCAN詳解
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
