### Transformer
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Transformer的架構，主要由兩個部分組成：編碼器（Encoder） 和 解碼器（Decoder）
- 編碼器（Encoder): 元件與用途
  - Input Embedding：輸入的序列（例如詞彙）會先轉換成嵌入向量（embedding）==> 詞的向量表示。
  - Positional Encoding(位置編碼)：
    - 由於 Transformer 本身不具備序列的時間步概念，因此需要引入位置編碼（Positional Encoding）來讓模型了解輸入序列的順序。
    - 通過結合一種簡單的位置編碼方法，強調了位置資訊在 Transformer 模型中的關鍵作用，確保我們的模型能夠識別序列中元素的順序。 
  - Multi-Head Attention：這是 Transformer 的核心機制，利用**多頭自注意力（Multi-Head Self-Attention）**來讓每個輸入詞彙與其他所有詞進行關聯，捕捉整個序列的依賴關係。
  - Add & Norm(Skip Connections 和 Layer Normalization)：
    - 使用`殘差連接`，將多頭注意力的輸出和輸入相加，並進行`Layer 正規化`。
    - 整合這些基本元件以確保模型的訓練穩定性和效率，展示了它們在促進深度架構中有效學習的作用。
  - Feed Forward：每個編碼器層都有一個全連接層來進一步處理數據。 
- 解碼器（Decoder）
  - 解碼器結構與編碼器類似
  - 新增 Masked Multi-Head Attention。這是為了確保在訓練生成時，解碼器只能看到之前的輸出，而不能看到未來的輸出，避免數據洩漏。
  - 解碼器的輸出==> 通過一個線性層，再進行 Softmax，生成對應的機率分布，代表模型對下一個詞的預測。 
- 基本構建單元 ==> 縮放點積注意力（scaled dot-product attention）單元

### `1`.【架構】導讀
- 👍[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  -  【教學影片(限量免費版20250610)】[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/?utm_campaign=handsonllm-launch&utm_medium=partner) 
- 👍[Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- https://ithelp.ithome.com.tw/articles/10363257

### `2`.【架構】實作==> 核心關鍵技術的程式實作[Optional]
### `3`.Transformer工程學[Optional]
- Transfer Learning
  - 【範例】 [Transfer learning with Transformers trainer and pipeline for NLP](https://billtcheng2013.medium.com/transfer-learning-with-transformers-trainer-and-pipeline-for-nlp-8b1d2c1a8c3d)
  - 【REVIEW】202410[Transfer Learning on Transformers for Building Energy Consumption Forecasting -- A Comparative Study](https://arxiv.org/abs/2410.14107) 
- Fine-Tuning
#  `4`.範例學習==>應用
#### 範例學習:[Transformer_HuggingFace範例](Transformer_HuggingFace範例.md)
#### 作業 ==> 入門推薦 完成 [快速入門Quick tour](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/quicktour.ipynb)
#### 範例學習: 實作一個Keras Transformer
- Attention Layers in TensorFlow
  - Self-Attention (Scaled Dot-Product Attention) --> tf.keras.layers.Attention
  - Multi-Head Attention  --> tf.keras.layers.MultiHeadAttention
  - https://www.geeksforgeeks.org/attention-layers-in-tensorflow/
- [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/)

#### 範例: Tensorflow  Transformer
- [Neural machine translation with attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
- [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)

#### 範例: 實作一個PyTorch Transformer
- [Building a Simple Transformer using PyTorch [Code Included]](https://pureai.substack.com/p/building-a-simple-transformer-using-pytorch)
- https://github.com/ermattson/pure-ai-tutorials/tree/main/SimpleTransformer-PyTorch
- [Natural Language Processing with Transformers, Revised Edition](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
   - https://github.com/nlp-with-transformers/notebooks
   - 01_introduction.ipynb

#### 更多範例
- https://fancyerii.github.io/2020/07/08/huggingface-transformers/

# `5`.延伸閱讀
## 【REVIEW】
- [A Survey of Transformers](https://arxiv.org/abs/2106.04554)
- [A Survey on Visual Transformer](https://arxiv.org/abs/2012.12556)
- [Transformers in Time Series: A Survey](https://arxiv.org/abs/2202.07125)
- [Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006)

## 參考書
- [Natural Language Processing with Transformers, Revised Edition](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
  - https://github.com/nlp-with-transformers/notebooks
  - 第一版簡體中譯本
- [Hands-On Generative AI with Transformers and Diffusion Models](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
  - https://github.com/genaibook/genaibook
- [Mastering Transformers - Second Edition](https://learning.oreilly.com/library/view/mastering-transformers/9781837633784/)
  - https://github.com/PacktPublishing/Mastering-Transformers-Second-Edition
- [Transformers for Natural Language Processing and Computer Vision - Third Edition](https://learning.oreilly.com/library/view/transformers-for-natural/9781805128724/)
  - https://github.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition/
- [Hands-On Large Language Models](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/)
  - [章節內容](LLM_BOOK_Content.md) 
