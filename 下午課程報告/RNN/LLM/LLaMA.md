# LLaMA(Large Language Model Meta AI)  
- 2023 Meta AI | LLaMA(Large Language Model Meta AI)(2023年2月)
  - https://zh.wikipedia.org/zh-tw/LLaMA
  - [[2302.13971]LLaMA: Open and efficient foundation language models](https://arxiv.org/abs/2302.13971)
- LLaMA 2(2023年7月)
  - [Llama 2: Open Foundation and Fine-Tuned Chat Models]() 
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
## 架構實作
## 應用
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
