### Transformer
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 基本構建單元 ==> 縮放點積注意力（scaled dot-product attention）單元
- 關鍵要素
  - 單頭自我注意機制：我們將多頭自我注意的概念提煉到其核心，展示了自我注意在處理序列中的基本作。
  - 一個簡單的位置前饋網路：通過前饋網路的極簡版本，我們說明瞭 Transformers 如何獨立地對每個位置的數據進行轉換，從而增強模型捕獲數據內關係的能力。
  - Skip Connections 和 Layer Normalization：集成這些基本元件以確保模型的訓練穩定性和效率，展示了它們在促進深度架構中有效學習的作用。
  - 簡單的位置編碼：我們通過結合一種簡單的位置編碼方法，強調了位置資訊在 Transformer 模型中的關鍵作用，確保我們的模型能夠識別序列中元素的順序。 

### 導讀
- 👍[Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- 👍[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- https://ithelp.ithome.com.tw/articles/10363257

## 範例:應用
- [Natural Language Processing with Transformers, Revised Edition](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
   - https://github.com/nlp-with-transformers/notebooks
   - 01_introduction.ipynb

## 範例: 實作一個Keras Transformer
- [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/)

## 範例: Tensorflow  Transformer
- [Neural machine translation with attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
- [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)
## 範例: 實作一個PyTorch Transformer
- [Building a Simple Transformer using PyTorch [Code Included]](https://pureai.substack.com/p/building-a-simple-transformer-using-pytorch)
- https://github.com/ermattson/pure-ai-tutorials/tree/main/SimpleTransformer-PyTorch
