# GPT (Generative Pre-trained Transformer)
- OPENAI GPT (Generative Pre-trained Transformer)
  - https://en.wikipedia.org/wiki/Generative_pre-trained_transformer
  - GPT-1(2018)
    - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 
  - GPT-2(2019)
    - [Language Models are Unsupervised Multitask Learners]() 
  - GPT-3(2020)
    - [[2005.14165] Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) 
  - GPT-3.5(2022)
  - GPT-4(2023)
    - GPT-4是一個多模態LLM，能夠處理文字和圖像輸入（儘管其輸出僅限於文字）
    - 2023 [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
    - [GPT-4](https://openai.com/index/gpt-4-research/)
  - GPT-4o(2024年5月發布)
  - GPT-5 ?? MLLM
## 導讀
- [How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

## 模型實作
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
- https://github.com/jaymody/picoGPT/tree/main
- https://jiqihumanr.github.io/2023/04/13/gpt-from-scratch/
- [程式碼 gpt2.py](https://github.com/jaymody/picoGPT/blob/main/gpt2.py)
- 執行
```python
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 8
```
- 使用 Docker執行
```
docker build -t "openai-gpt-2" "https://gist.githubusercontent.com/jaymody/9054ca64eeea7fad1b58a185696bb518/raw/Dockerfile"

docker run -dt "openai-gpt-2" --name "openai-gpt-2-app"

docker exec -it "openai-gpt-2-app" /bin/bash -c 'python3 src/interactive_conditional_samples.py --length 8 --model_type 124M --top_k 1'

# paste "Alan Turing theorized that computers would one day become" when prompted
```
## 應用
