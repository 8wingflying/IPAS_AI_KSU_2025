# Stable Diffusion 3
- [Stable Diffusion 3 in KerasHub!](https://keras.io/keras_hub/guides/stable_diffusion_3_in_keras_hub/)


#### 說明
- Stable Diffusion 3 是一個強大的開源潛在擴散模型 （LDM） 
- 本範例展示透過文本提示生成高品質的新穎圖像。
- 使用由 Stability AI 的預訓練模型，它被預訓練了 10 億 圖像，並在 3300 萬張高品質美學和偏好圖像上進行微調 。
- 本範例展示KerasHub 的 Stable Diffusion 3 Medium 實現，包括文本到圖像、圖像到圖像和修復任務。

- 安裝一些依賴項並獲取演示的鏡像：
```python
!pip install -Uq keras
!pip install -Uq git+https://github.com/keras-team/keras-hub.git
!wget -O mountain_dog.png https://raw.githubusercontent.com/keras-team/keras-io/master/guides/img/stable_diffusion_3_in_keras_hub/mountain_dog.png
!wget -O mountain_dog_mask.png https://raw.githubusercontent.com/keras-team/keras-io/master/guides/img/stable_diffusion_3_in_keras_hub/mountain_dog_mask.png
```

```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import time

import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
```

```python

text_to_image = keras_hub.models.StableDiffusion3TextToImage.from_preset(
    "stable_diffusion_3_medium", dtype="float16"
)
```

```python
def display_generated_images(images):
    """Helper function to display the images from the inputs.

    This function accepts the following input formats:
    - 3D numpy array.
    - 4D numpy array: concatenated horizontally.
    - List of 3D numpy arrays: concatenated horizontally.
    """
    display_image = None
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            display_image = Image.fromarray(images)
        elif images.ndim == 4:
            concated_images = np.concatenate(list(images), axis=1)
            display_image = Image.fromarray(concated_images)
    elif isinstance(images, list):
        concated_images = np.concatenate(images, axis=1)
        display_image = Image.fromarray(concated_images)

    if display_image is None:
        raise ValueError("Unsupported input format.")

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(display_image)
    plt.show()
    plt.close()


backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium", image_shape=(512, 512, 3), dtype="float16"
)
preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
    "stable_diffusion_3_medium"
)
text_to_image = keras_hub.models.StableDiffusion3TextToImage(backbone, preprocessor)
```

```python
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# When using JAX or TensorFlow backends, you might experience a significant
# compilation time during the first `generate()` call. The subsequent
# `generate()` call speedup highlights the power of JIT compilation and caching
# in frameworks like JAX and TensorFlow, making them well-suited for
# high-performance deep learning tasks like image generation.
generated_image = text_to_image.generate(prompt)
display_generated_images(generated_image)
```

##### 文字到圖像任務
```python

generated_images = text_to_image.generate([prompt] * 3)
display_generated_images(generated_images)
```

```python

num_steps = [10, 28, 50]
generated_images = []
for n in num_steps:
    st = time.time()
    generated_images.append(text_to_image.generate(prompt, num_steps=n))
    print(f"Cost time (`num_steps={n}`): {time.time() - st:.2f}s")

display_generated_images(generated_images)
```
- 指導模型生成特定的 styles 和 elements
```python

generated_images = text_to_image.generate(
    {
        "prompts": [prompt] * 3,
        "negative_prompts": ["Green color"] * 3,
    }
)
display_generated_images(generated_images)
```
- 使用guidance_scale來影響圖像生成的程度
```PYTHON
generated_images = [
    text_to_image.generate(prompt, guidance_scale=2.5),
    text_to_image.generate(prompt, guidance_scale=7.0),
    text_to_image.generate(prompt, guidance_scale=10.5),
]
display_generated_images(generated_images)
```
##### 圖像到圖像任務 ==> StableDiffusion3ImageToImage
```python
image_to_image = keras_hub.models.StableDiffusion3ImageToImage(backbone, preprocessor)

image = Image.open("mountain_dog.png").convert("RGB")
image = image.resize((512, 512))
width, height = image.size

# 特別注意 values of the image must be in the range of [-1.0, 1.0].
rescale = keras.layers.Rescaling(scale=1 / 127.5, offset=-1.0)
image_array = rescale(np.array(image))

prompt = "dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, "
prompt += "adorable, Pixar, Disney, 8k"

generated_image = image_to_image.generate(
    {
        "images": image_array,
        "prompts": prompt,
    }
)

display_generated_images(
    [
        np.array(image),//原始圖片
        generated_image,//產生的圖片
    ]
)
```
- 參數設定 ==>strength
- 

```python
generated_images = [
    image_to_image.generate(
        {
            "images": image_array,
            "prompts": prompt,
        },
        strength=0.7, //參數設定
    ),
    image_to_image.generate(
        {
            "images": image_array,
            "prompts": prompt,
        },
        strength=0.8,//參數設定
    ),
    image_to_image.generate(
        {
            "images": image_array,
            "prompts": prompt,
        },
        strength=0.9,//參數設定
    ),
]
display_generated_images(generated_images)
```
##### 圖片修復(Inpaint)任務 ==> StableDiffusion3Inpaint
- [keras_hub.models.StableDiffusion3Inpaint(backbone, preprocessor, **kwargs)](https://keras.io/keras_hub/api/models/stable_diffusion_3/stable_diffusion_3_inpaint/)
- inpaint.generate方法
- from_preset 方法
```python
inpaint = keras_hub.models.StableDiffusion3Inpaint(backbone, preprocessor)

image = Image.open("mountain_dog.png").convert("RGB")
image = image.resize((512, 512))
image_array = rescale(np.array(image))

# Note that the mask values are of boolean dtype.
mask = Image.open("mountain_dog_mask.png").convert("L")
mask = mask.resize((512, 512))
mask_array = np.array(mask).astype("bool")

prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly "
prompt += "detailed, 8k"

generated_image = inpaint.generate(
    {
        "images": image_array,
        "masks": mask_array,
        "prompts": prompt,
    }
)
display_generated_images(
    [
        np.array(image),
        np.array(mask.convert("RGB")),
        generated_image,
    ]
)
```




