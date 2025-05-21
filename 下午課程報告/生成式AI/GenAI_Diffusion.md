## 學習資源
- https://andy6804tw.github.io/wiki/
## 歷史發展
- 2015 Diffusion Model
  - [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.pdf)
- 2020經典[(arXiv:2006.11239)Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  - https://github.com/hojonathanho/diffusion 
- [(arXiv:2102.09672)Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- 2021 DALL-E
  - https://zh.wikipedia.org/wiki/DALL-E
  - DALL-E 2(2022-04)
  - DALL-E 3(October 2023)
  - Sora (2024-02)
  - In March 2025, DALL-E-3 was replaced in ChatGPT by GPT-4o's native image-generation capabilities.
- 2022 Midjourney
  - https://zh.wikipedia.org/wiki/Midjourney 
- 2022 Stable_Diffusion
  - https://zh.wikipedia.org/zh-tw/Stable_Diffusion
  - Stable Diffusion (2022-08)
  - Stable Diffusion 3 (2024-03)
  - Stable Video 4D (2024-07)
- 2021 Latent diffusion model (LDM)
  - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 
- 2021 CLIP (Contrastive Language-Image Pre-Training)
  - 經典[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) 
- 2021 Vision Transformer (ViT)
- 微軟提出的 Swin Transformer
  - 【論文】[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
  - 【導讀】https://www.geeksforgeeks.org/swin-transformer/
  - 【導讀】https://blog.csdn.net/qq_37541097/article/details/121119988
- Facebook 提出的 DeiT (Data-efficient image Transformer)
- LLM再學習 ==>
  - Parameter-Efficient Fine-Tuning (PEFT) ==> 🤗 [HuggingFace Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft)
  - 【綜合報告】[Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment](https://arxiv.org/abs/2312.12148)
  - 透過凍結原本的預訓練模型(e.g., GPT-3) 的權重，搭配一個小的模型進行微調就可以達到很好的 Fine-Tuning 效果，同 Adapter 的概念：透過 Freeze LLM 僅透過微調新增的小型網路，當作補丁或是插件
  - LoRA(2021): Low-Rank Adaptation of Large Language Models
    - 【論文】[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
    - https://github.com/cloneofsimo/lora
    - https://huggingface.co/spaces/lora-library/LoRA-DreamBooth-Training-UI
    - 【導讀】微調大型語言模型LLM的技術LoRA及生成式AI-Stable diffusion LoRA
    - 【導讀】【LLM專欄】All about Lora
    - https://www.accucrazy.com/lora-ai-training/
  - Dreambooth(2022)
    - 【論文】[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) 
  - ControlNet(2023)
    - 【論文】[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)  
- https://d223302.github.io/AACL2022-Pretrain-Language-Model-Tutorial/lecture_material/AACL_2022_tutorial_PLMs.pdf


## [Stable Diffusion 在線](https://stablediffusionweb.com/zh-tw)
## BOOKS
- [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
- [Using Stable Diffusion with Python](https://learning.oreilly.com/library/view/using-stable-diffusion/9781835086377/)
- [Diffusions in Architecture: Artificial Intelligence and Image Generators](https://learning.oreilly.com/library/view/diffusions-in-architecture/9781394191772/)
- [Applied Generative AI for Beginners: Practical Knowledge on Diffusion Models, ChatGPT, and Other LLMs]()
