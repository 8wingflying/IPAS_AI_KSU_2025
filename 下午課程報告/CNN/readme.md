# [Transfer learning with TensorFlow Hub](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)

#### 圖形處理常用套件
- [Python Image Processing Libraries](https://www.geeksforgeeks.org/python-image-processing-libraries/)
- OpenCV
- Scikit-Image
- Pillow/PIL
- Matplotlib
- [seaborn: statistical data visualization](https://seaborn.pydata.org/)
  - 【學習指南(tutorial)】https://seaborn.pydata.org/tutorial.html
  - 【範例】https://seaborn.pydata.org/examples/index.html

#### 範例學習
- [Image Processing with SciPy and NumPy in Python](https://www.geeksforgeeks.org/image-processing-with-scipy-and-numpy-in-python/)
- 使用 NumPy 和 SciPy 等核心科學模組在 Python 中進行影像處理。
- 圖像由 NumPy ndarrays 組成，因此我們可以處理和操作圖像
- SciPy 提供了子模組 scipy.ndimage，它提供了可以在 NumPy 數組上操作的函數
- 安裝
  - pip install numpy
  - pip install scipy 
- SciPy 的misc packag附帶一些預先載入的圖片。
- 我們將使用這些圖像來了解圖像處理。
- face() 函數提供了一個這樣的圖像。
- face() 函數將取得浣熊(raccoon)臉的彩色影像。
- 範例：使用 SciPy 儲存影像
```python
from scipy import misc
import imageio
import matplotlib.pyplot as plt

# reads a raccoon face
face = misc.face()

# save the image
imageio.imsave('raccoon.png', face)

plt.imshow(face)
plt.show()
```
- 範例：從圖像建立 NumPy 數組
```python
from scipy import misc
import imageio
import matplotlib.pyplot as plt

img = imageio.imread('raccoon.png')

print(img.shape)
print(img.dtype)

plt.imshow(img)
plt.show()
```
- 裁切影像
```python
from scipy import misc
import matplotlib.pyplot as plt

# for grascaling the image
img = misc.face(gray = True)


x, y = img.shape

# 裁切影像(Cropping the image)
crop = img[x//3: - x//8, y//3: - y//8]

plt.imshow(crop)
plt.show()
```
- 模糊影像
```python
from scipy import misc,ndimage
import matplotlib.pyplot as plt

img = misc.face()

# 模糊影像
blur_G = ndimage.gaussian_filter(img,sigma=7)

plt.imshow(blur_G)
plt.show()
```
