#### [A walk through latent space with Stable Diffusion 3](https://keras.io/examples/generative/random_walks_with_stable_diffusion_3/)
- 預設的 「stable_diffusion_3_medium」 不包括 T5XXL 文字編碼器，因為它需要更多的 GPU 記憶體
- 
```python
!# Use the latest version of KerasHub
!!pip install -Uq git+https://github.com/keras-team/keras-hub.git
```
- [StableDiffusion3Backbone](https://keras.io/keras_hub/api/models/stable_diffusion_3/stable_diffusion_3_backbone/)
  - stable_diffusion_3_medium
  - stable_diffusion_3.5_medium
  - stable_diffusion_3.5_large
  - stable_diffusion_3.5_large_turbo	 
```python
import math

import keras
import keras_hub
import matplotlib.pyplot as plt
from keras import ops
from keras import random
from PIL import Image

height, width = 512, 512
num_steps = 28
guidance_scale = 7.0
dtype = "float16"

# 設定使用的 Stable Diffusion 3 模型 ==> stable_diffusion_3_medium
backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium", image_shape=(height, width, 3), dtype=dtype
)
## 設定預處理器
preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
    "stable_diffusion_3_medium"
)
```

- 
```python

```
- 
```python

```
