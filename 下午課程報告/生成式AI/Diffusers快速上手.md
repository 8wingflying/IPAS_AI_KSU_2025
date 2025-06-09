## 🤗 Diffusers
- https://github.com/huggingface/diffusers
- 🤗 Diffusers 是最先進的預訓練擴散模型的首選庫，用於生成圖像、音訊甚至分子的 3D 結構。
- 無論您是在尋找簡單的推理解決方案還是訓練自己的擴散模型， 🤗 Diffusers 都是一個支援兩者的模組化工具箱。
- 我們的庫在設計時注重可用性而不是性能，簡單而不是簡單，以及可定製性而不是抽象。

## 🤗 Diffusers 提供三個核心元件：
- `最先進的擴散管道(State-of-the-art diffusion pipelines)` ==> 只需幾行代碼即可在推理中運行。
  - The DiffusionPipeline is a high-level end-to-end class designed to rapidly generate samples from pretrained diffusion models for inference.
- `可互換的雜訊調度器(Interchangeable noise schedulers)` ==> 適用於不同的擴散速度和輸出品質。
  - Many different schedulers - algorithms that control how noise is added for training, and how to generate denoised images during inference.
- `預訓練模型(Pretrained models)`可用作構建區塊(building blocks)，並與`調度器(schedulers)`結合使用，用於創建您自己的端到端擴散系統。
  - Popular pretrained model architectures and modules that can be used as building blocks for creating diffusion systems.

## diffusers_實戰1 ==> 更多學習 請參閱 https://huggingface.co/docs/diffusers/index
- https://github.com/huggingface/diffusers
- [使用T4 GPU](diffusers_實戰1_20250609_2.ipynb)
```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```
