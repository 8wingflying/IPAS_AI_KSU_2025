## [Developer guides](https://keras.io/guides/)
## [Keras 3 API documentation](https://keras.io/api/)
- [Keras Applications(經典模型)](https://keras.io/api/applications/)
## Keras Tuner 調教器
## Keras RS 推薦系統
## Keras Hub(Keras社群模型)
## [Keras code examples(150)](https://keras.io/examples/)
- Computer Vision
- Natural Language Processing
- Structured Data
- Timeseries
- Generative Deep Learning
- Audio Data
- Reinforcement Learning
- Graph Data
- Quick Keras Recipes
## [Getting started with Keras](https://keras.io/getting_started/)
```python
pip install --upgrade keras-cv
pip install --upgrade keras-hub
pip install --upgrade keras
```
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```
##
```python
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
```
