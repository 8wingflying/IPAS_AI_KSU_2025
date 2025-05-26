# 非監督式學習之維度縮減演算法_教科書範例
- PCA
- Random Projection
- LLE

### LLE
- LLE（Locally Linear Embedding）演算法，即局部線性嵌入演算法。
- 該演算法是針對非線性信號特徵向量維數的優化方法，這種維數優化並不是僅僅在數量上簡單的約簡，而是在保持原始資料性質不變的情況下，將高維空間的信號映射到低維空間上，即特徵值的二次提
- https://www.geeksforgeeks.org/locally-linear-embedding-in-machine-learning/
- https://shunliz.gitbooks.io/machine-learning/content/ml/clean-feature/lle.html
- 關鍵步驟：
  - `1`. LLE 查找每個點的最近鄰域。
  - `2`.計算出顯示每個點與其相鄰點的關係的權重。這些權重有助於捕獲數據的本地結構。
  - `3`.LLE 使用這些權重來創建數據的低維版本。
  - 它通過根據所需的維數選擇最佳方向（特徵向量）來保持重要的形狀。
### 數學推導
### 範例 ==> 瑞士捲

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

## 產生瑞士捲
n_samples = 1000
n_neighbors = 10
X, _ = make_swiss_roll(n_samples=n_samples)

## 建立LLE
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2)

X_reduced = lle.fit_transform(X)

## 檢視原始資料與轉換後的資料
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(122)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
plt.title("Reduced Data (LLE)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.tight_layout()
plt.show()

## 
```

```python


```

```python


```

```python


```

```python


```
### LLE:教科書範例
```python
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)

X_unrolled = lle.fit_transform(X_swiss)
```
- 底下是畫圖
```python
# Define the save_fig function
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Saves the current matplotlib figure."""
    import os
    PROJECT_ROOT_DIR = "." # You might need to adjust this path
    CHAPTER_ID = "dimensionality_reduction" # You might need to adjust this
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

```python
# extra code – this cell generates and saves Figure 8–10
import matplotlib.cm as cm

darker_hot = cm.hot

plt.scatter(X_unrolled[:, 0], X_unrolled[:, 1],c=t, cmap=darker_hot)
plt.xlabel("$z_1$")
plt.ylabel("$z_2$", rotation=0)
plt.axis([-0.055, 0.060, -0.070, 0.090])
plt.grid(True)

save_fig("lle_unrolling_plot")
plt.title("Unrolled swiss roll using LLE")
plt.show()
```
