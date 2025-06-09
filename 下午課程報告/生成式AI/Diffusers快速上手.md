## 🤗 Diffusers
- https://github.com/huggingface/diffusers
- 🤗 Diffusers 是最先進的預訓練擴散模型的首選庫，用於生成圖像、音訊甚至分子的 3D 結構。
- 無論您是在尋找簡單的推理解決方案還是訓練自己的擴散模型， 🤗 Diffusers 都是一個支援兩者的模組化工具箱。
- 我們的庫在設計時注重可用性而不是性能，簡單而不是簡單，以及可定製性而不是抽象。

## 🤗 Diffusers 提供三個核心元件：
- `最先進的擴散管道(State-of-the-art diffusion pipelines)` ==> 只需幾行代碼即可在推理中運行。
- `可互換的雜訊調度器(Interchangeable noise schedulers)` ==> 適用於不同的擴散速度和輸出品質。
- `預訓練模型(Pretrained models)`可用作構建區塊(building blocks)，並與`調度器(schedulers)`結合使用，用於創建您自己的端到端擴散系統。

## diffusers_實戰1
- https://github.com/huggingface/diffusers
- [使用T4 GPU](diffusers_實戰1_20250609.ipynb)
```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```
