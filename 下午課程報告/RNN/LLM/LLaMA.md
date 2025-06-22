# LLaMA(Large Language Model Meta AI)  
- Meta AI | LLaMA(Large Language Model Meta AI)
- LLaMA 1(2023年2月)
  - https://zh.wikipedia.org/zh-tw/LLaMA
  - [[2302.13971]LLaMA: Open and efficient foundation language models](https://arxiv.org/abs/2302.13971)
- LLaMA 2(2023年7月)
  - [Llama 2: Open Foundation and Fine-Tuned Chat Models]()
  - Llama 2使用了和Llama 1相同的模型架構以及tokenizer。
  - 與Llama 1不同的是，Llama 2將上下文長度擴展到了4k，並且34B和70B參數量版本使用了GQA
- Code Llama(2023年8月)
- Llama 3 (2024年4月18日)
  - [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) 
- Llama 4 (2025年4月5日)
  - 架構已更改為混合專家模型。
  - 具備多模態（文字和圖像輸入，文字輸出）和多語言（12種語言）特性
## Open Source LLM
- [Open LLMs](https://github.com/eugeneyan/open-llms)
  - https://blog.n8n.io/open-source-llm/
  - https://medium.com/@yugank.aman/top-10-open-source-llm-models-and-their-uses-6f4a9aced6af
  - https://github.com/Hannibal046/Awesome-LLM
  - https://www.geeksforgeeks.org/artificial-intelligence/top-10-open-source-llm-models/
- LLaMA(Large Language Model Meta AI) 
- Mistral 7B
  - [Mistral 7B](https://arxiv.org/abs/2310.06825) 
- Falcon
- Open LLMs for code
  - Code Llama 
  - [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
  - https://huggingface.co/codellama
  - https://github.com/Meta-Llama/codellama
  - https://ai.meta.com/blog/code-llama-large-language-model-coding/
  - https://blog.csdn.net/qq_36530891/article/details/132832476
  - [Code Llama 本地部署使用指南，并在 VSCode 和 chatbox 中使用](https://www.bingal.com/posts/code-llama-usage/)
## LLama導讀
- https://zhuanlan.zhihu.com/p/694072728
- https://blog.csdn.net/qq_41185868/article/details/137981416
- 

## LLaMA-Factory
- https://llamafactory.readthedocs.io/zh-cn/latest/index.html
- https://zhuanlan.zhihu.com/p/1919124023255741920
- LLaMA Factory 是一個簡單易用且高效的大型語言模型（Large Language Model）訓練與微調平臺。
- 通過 LLaMA Factory，可以在無需編寫任何代碼的前提下，在本地完成上百種預訓練模型的微調
- 框架特性包括：
  - 模型種類：LLaMA、LLaVA、Mistral、Mixtral-MoE、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phi 等等。
  - 訓練演算法：（增量）預訓練、（多模態）指令監督微調、獎勵模型訓練、PPO 訓練、DPO 訓練、KTO 訓練、ORPO 訓練等等。
  - 運算精度：16 比特全參數微調、凍結微調、LoRA 微調和基於 AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ 的 2/3/4/5/6/8 比特 QLoRA 微調。
  - 優化演算法：GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ 和 PiSSA。
  - 加速運算元：FlashAttention-2 和 Unsloth。
  - 推理引擎：Transformers 和 vLLM。
  - 實驗監控：LlamaBoard、TensorBoard、Wandb、MLflow、SwanLab 等等。

## 架構實作
## 應用
- https://huggingface.co/docs/transformers/main/model_doc/llama4
```python
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

messages = [
    {"role": "user", "content": "what is the recipe of mayonnaise?"},
]

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

output = pipe(messages, do_sample=False, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```
