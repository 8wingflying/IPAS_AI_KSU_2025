# 非監督式學習之維度縮減演算法_教科書範例
- PCA
- Random Projection
- LLE

### LLE
- https://www.geeksforgeeks.org/locally-linear-embedding-in-machine-learning/
```python


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
