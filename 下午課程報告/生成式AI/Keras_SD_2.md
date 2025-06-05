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

- 撰寫helper函數
```python
def get_text_embeddings(prompt):
    """Get the text embeddings for a given prompt."""
    token_ids = preprocessor.generate_preprocess([prompt])
    negative_token_ids = preprocessor.generate_preprocess([""])
    (
        positive_embeddings,
        negative_embeddings,
        positive_pooled_embeddings,
        negative_pooled_embeddings,
    ) = backbone.encode_text_step(token_ids, negative_token_ids)
    return (
        positive_embeddings,
        negative_embeddings,
        positive_pooled_embeddings,
        negative_pooled_embeddings,
    )


def decode_to_images(x, height, width):
    """Concatenate and normalize the images to uint8 dtype."""
    x = ops.concatenate(x, axis=0)
    x = ops.reshape(x, (-1, height, width, 3))
    x = ops.clip(ops.divide(ops.add(x, 1.0), 2.0), 0.0, 1.0)
    return ops.cast(ops.round(ops.multiply(x, 255.0)), "uint8")


def generate_with_latents_and_embeddings(
    latents, embeddings, num_steps, guidance_scale
):
    """Generate images from latents and text embeddings."""

    def body_fun(step, latents):
        return backbone.denoise_step(
            latents,
            embeddings,
            step,
            num_steps,
            guidance_scale,
        )

    latents = ops.fori_loop(0, num_steps, body_fun, latents)
    return backbone.decode_step(latents)


def export_as_gif(filename, images, frames_per_second=10, no_rubber_band=False):
    if not no_rubber_band:
        images += images[2:-1][::-1]  # Makes a rubber band: A->B->A
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )
```
- 設定==>因為要使用自定義latent和embedding生成圖像  
```python
if keras.config.backend() == "torch":
    import torch

    @torch.no_grad()
    def wrapped_function(*args, **kwargs):
        return generate_with_latents_and_embeddings(*args, **kwargs)

    generate_function = wrapped_function
elif keras.config.backend() == "tensorflow":
    import tensorflow as tf

    generate_function = tf.function(
        generate_with_latents_and_embeddings, jit_compile=True
    )
elif keras.config.backend() == "jax":
    import itertools

    import jax

    @jax.jit
    def compiled_function(state, *args, **kwargs):
        (trainable_variables, non_trainable_variables) = state
        mapping = itertools.chain(
            zip(backbone.trainable_variables, trainable_variables),
            zip(backbone.non_trainable_variables, non_trainable_variables),
        )
        with keras.StatelessScope(state_mapping=mapping):
            return generate_with_latents_and_embeddings(*args, **kwargs)

    def wrapped_function(*args, **kwargs):
        state = (
            [v.value for v in backbone.trainable_variables],
            [v.value for v in backbone.non_trainable_variables],
        )
        return compiled_function(state, *args, **kwargs)

    generate_function = wrapped_function
```


- Interpolating between text prompts
```python
prompt_1 = "A cute dog in a beautiful field of lavander colorful flowers "
prompt_1 += "everywhere, perfect lighting, leica summicron 35mm f2.0, kodak "
prompt_1 += "portra 400, film grain"
prompt_2 = prompt_1.replace("dog", "cat")
interpolation_steps = 5

encoding_1 = get_text_embeddings(prompt_1)
encoding_2 = get_text_embeddings(prompt_2)


# Show the size of the latent manifold
print(f"Positive embeddings shape: {encoding_1[0].shape}")
print(f"Negative embeddings shape: {encoding_1[1].shape}")
print(f"Positive pooled embeddings shape: {encoding_1[2].shape}")
print(f"Negative pooled embeddings shape: {encoding_1[3].shape}")
```
- 使用Spherical Linear Interpolation （slerp） 簡單線性插值
- https://en.wikipedia.org/wiki/Slerp

```python
def slerp(v1, v2, num):
    ori_dtype = v1.dtype
    # Cast to float32 for numerical stability.
    v1 = ops.cast(v1, "float32")
    v2 = ops.cast(v2, "float32")

    def interpolation(t, v1, v2, dot_threshold=0.9995):
        """helper function to spherically interpolate two arrays."""
        dot = ops.sum(
            v1 * v2 / (ops.linalg.norm(ops.ravel(v1)) * ops.linalg.norm(ops.ravel(v2)))
        )
        if ops.abs(dot) > dot_threshold:
            v2 = (1 - t) * v1 + t * v2
        else:
            theta_0 = ops.arccos(dot)
            sin_theta_0 = ops.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = ops.sin(theta_t)
            s0 = ops.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v1 + s1 * v2
        return v2

    t = ops.linspace(0, 1, num)
    interpolated = ops.stack([interpolation(t[i], v1, v2) for i in range(num)], axis=0)
    return ops.cast(interpolated, ori_dtype)


interpolated_positive_embeddings = slerp(
    encoding_1[0], encoding_2[0], interpolation_steps
)

interpolated_positive_pooled_embeddings = slerp(
    encoding_1[2], encoding_2[2], interpolation_steps
)
# We don't use negative prompts in this example, so there’s no need to
# interpolate them.
negative_embeddings = encoding_1[1]
negative_pooled_embeddings = encoding_1[3]
```
- 從每個點生成圖像
```python
latents = random.normal((1, height // 8, width // 8, 16), seed=42)

images = []
progbar = keras.utils.Progbar(interpolation_steps)
for i in range(interpolation_steps):
    images.append(
        generate_function(
            latents,
            (
                interpolated_positive_embeddings[i],
                negative_embeddings,
                interpolated_positive_pooled_embeddings[i],
                negative_pooled_embeddings,
            ),
            ops.convert_to_tensor(num_steps),
            ops.convert_to_tensor(guidance_scale),
        )
    )
    progbar.update(i + 1, finalize=i == interpolation_steps - 1)
```

- 將圖像序列導出為gif
```python
from IPython.display import Image as IImage
IImage("dog_to_cat_5.gif")

images = ops.convert_to_numpy(decode_to_images(images, height, width))
export_as_gif(
    "dog_to_cat_5.gif",
    [Image.fromarray(image) for image in images],
    frames_per_second=2,
)
```


- 做一個更細粒度的插值 使用更多步驟
```python
interpolation_steps = 64
batch_size = 4
batches = interpolation_steps // batch_size

interpolated_positive_embeddings = slerp(
    encoding_1[0], encoding_2[0], interpolation_steps
)
interpolated_positive_pooled_embeddings = slerp(
    encoding_1[2], encoding_2[2], interpolation_steps
)
positive_embeddings_shape = ops.shape(encoding_1[0])
positive_pooled_embeddings_shape = ops.shape(encoding_1[2])
interpolated_positive_embeddings = ops.reshape(
    interpolated_positive_embeddings,
    (
        batches,
        batch_size,
        positive_embeddings_shape[-2],
        positive_embeddings_shape[-1],
    ),
)
interpolated_positive_pooled_embeddings = ops.reshape(
    interpolated_positive_pooled_embeddings,
    (batches, batch_size, positive_pooled_embeddings_shape[-1]),
)
negative_embeddings = ops.tile(encoding_1[1], (batch_size, 1, 1))
negative_pooled_embeddings = ops.tile(encoding_1[3], (batch_size, 1))

latents = random.normal((1, height // 8, width // 8, 16), seed=42)
latents = ops.tile(latents, (batch_size, 1, 1, 1))

images = []
progbar = keras.utils.Progbar(batches)
for i in range(batches):
    images.append(
        generate_function(
            latents,
            (
                interpolated_positive_embeddings[i],
                negative_embeddings,
                interpolated_positive_pooled_embeddings[i],
                negative_pooled_embeddings,
            ),
            ops.convert_to_tensor(num_steps),
            ops.convert_to_tensor(guidance_scale),
        )
    )
    progbar.update(i + 1, finalize=i == batches - 1)

images = ops.convert_to_numpy(decode_to_images(images, height, width))
export_as_gif(
    "dog_to_cat_64.gif",
    [Image.fromarray(image) for image in images],
    frames_per_second=2,
)

```

- 多個圖像插值
```python
prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
prompt_3 = "The eiffel tower in the style of starry night"
prompt_4 = "An architectural sketch of a skyscraper"

interpolation_steps = 8
batch_size = 4
batches = (interpolation_steps**2) // batch_size

encoding_1 = get_text_embeddings(prompt_1)
encoding_2 = get_text_embeddings(prompt_2)
encoding_3 = get_text_embeddings(prompt_3)
encoding_4 = get_text_embeddings(prompt_4)

positive_embeddings_shape = ops.shape(encoding_1[0])
positive_pooled_embeddings_shape = ops.shape(encoding_1[2])
interpolated_positive_embeddings_12 = slerp(
    encoding_1[0], encoding_2[0], interpolation_steps
)
interpolated_positive_embeddings_34 = slerp(
    encoding_3[0], encoding_4[0], interpolation_steps
)
interpolated_positive_embeddings = slerp(
    interpolated_positive_embeddings_12,
    interpolated_positive_embeddings_34,
    interpolation_steps,
)
interpolated_positive_embeddings = ops.reshape(
    interpolated_positive_embeddings,
    (
        batches,
        batch_size,
        positive_embeddings_shape[-2],
        positive_embeddings_shape[-1],
    ),
)
interpolated_positive_pooled_embeddings_12 = slerp(
    encoding_1[2], encoding_2[2], interpolation_steps
)
interpolated_positive_pooled_embeddings_34 = slerp(
    encoding_3[2], encoding_4[2], interpolation_steps
)
interpolated_positive_pooled_embeddings = slerp(
    interpolated_positive_pooled_embeddings_12,
    interpolated_positive_pooled_embeddings_34,
    interpolation_steps,
)
interpolated_positive_pooled_embeddings = ops.reshape(
    interpolated_positive_pooled_embeddings,
    (batches, batch_size, positive_pooled_embeddings_shape[-1]),
)
negative_embeddings = ops.tile(encoding_1[1], (batch_size, 1, 1))
negative_pooled_embeddings = ops.tile(encoding_1[3], (batch_size, 1))

latents = random.normal((1, height // 8, width // 8, 16), seed=42)
latents = ops.tile(latents, (batch_size, 1, 1, 1))

images = []
progbar = keras.utils.Progbar(batches)
for i in range(batches):
    images.append(
        generate_function(
            latents,
            (
                interpolated_positive_embeddings[i],
                negative_embeddings,
                interpolated_positive_pooled_embeddings[i],
                negative_pooled_embeddings,
            ),
            ops.convert_to_tensor(num_steps),
            ops.convert_to_tensor(guidance_scale),
        )
    )
    progbar.update(i + 1, finalize=i == batches - 1)

## 在網格中顯示生成的圖像，以使其更易於解釋。

def plot_grid(images, path, grid_size, scale=2):
    fig, axs = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * scale, grid_size * scale)
    )
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")
    for ax in axs.flat:
        ax.axis("off")

    for i in range(min(grid_size * grid_size, len(images))):
        ax = axs.flat[i]
        ax.imshow(images[i])
        ax.axis("off")

    for i in range(len(images), grid_size * grid_size):
        axs.flat[i].axis("off")
        axs.flat[i].remove()

    plt.savefig(
        fname=path,
        pad_inches=0,
        bbox_inches="tight",
        transparent=False,
        dpi=60,
    )


images = ops.convert_to_numpy(decode_to_images(images, height, width))
plot_grid(images, "4-way-interpolation.jpg", interpolation_steps)
```

- 文本提示周圍走動
- 繞著潛在流形走一走 從特定提示生成的點開始
```python
images = []
progbar = keras.utils.Progbar(batches)
for i in range(batches):
    # Vary diffusion latents for each input.
    latents = random.normal((batch_size, height // 8, width // 8, 16))
    images.append(
        generate_function(
            latents,
            (
                interpolated_positive_embeddings[i],
                negative_embeddings,
                interpolated_positive_pooled_embeddings[i],
                negative_pooled_embeddings,
            ),
            ops.convert_to_tensor(num_steps),
            ops.convert_to_tensor(guidance_scale),
        )
    )
    progbar.update(i + 1, finalize=i == batches - 1)

images = ops.convert_to_numpy(decode_to_images(images, height, width))
plot_grid(images, "4-way-interpolation-varying-latent.jpg", interpolation_steps)
```
-
```python
walk_steps = 64
batch_size = 4
batches = walk_steps // batch_size
prompt = "An oil paintings of cows in a field next to a windmill in Holland"
encoding = get_text_embeddings(prompt)

walk_latent_x = random.normal((1, height // 8, width // 8, 16))
walk_latent_y = random.normal((1, height // 8, width // 8, 16))
walk_scale_x = ops.cos(ops.linspace(0.0, 2.0, walk_steps) * math.pi)
walk_scale_y = ops.sin(ops.linspace(0.0, 2.0, walk_steps) * math.pi)
latent_x = ops.tensordot(walk_scale_x, walk_latent_x, axes=0)
latent_y = ops.tensordot(walk_scale_y, walk_latent_y, axes=0)
latents = ops.add(latent_x, latent_y)
latents = ops.reshape(latents, (batches, batch_size, height // 8, width // 8, 16))

images = []
progbar = keras.utils.Progbar(batches)
for i in range(batches):
    images.append(
        generate_function(
            latents[i],
            (
                ops.tile(encoding[0], (batch_size, 1, 1)),
                ops.tile(encoding[1], (batch_size, 1, 1)),
                ops.tile(encoding[2], (batch_size, 1)),
                ops.tile(encoding[3], (batch_size, 1)),
            ),
            ops.convert_to_tensor(num_steps),
            ops.convert_to_tensor(guidance_scale),
        )
    )
    progbar.update(i + 1, finalize=i == batches - 1)

images = ops.convert_to_numpy(decode_to_images(images, height, width))
export_as_gif(
    "cows.gif",
    [Image.fromarray(image) for image in images],
    frames_per_second=4,
    no_rubber_band=True,
)
```
