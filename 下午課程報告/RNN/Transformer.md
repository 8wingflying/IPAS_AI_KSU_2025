### Transformer
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 基本構建單元 ==> 縮放點積注意力（scaled dot-product attention）單元
- 關鍵要素
  - 單頭自我注意機制：我們將多頭自我注意的概念提煉到其核心，展示了自我注意在處理序列中的基本作。
  - 一個簡單的位置前饋網路：通過前饋網路的極簡版本，我們說明瞭 Transformers 如何獨立地對每個位置的數據進行轉換，從而增強模型捕獲數據內關係的能力。
  - Skip Connections 和 Layer Normalization：集成這些基本元件以確保模型的訓練穩定性和效率，展示了它們在促進深度架構中有效學習的作用。
  - 簡單的位置編碼：我們通過結合一種簡單的位置編碼方法，強調了位置資訊在 Transformer 模型中的關鍵作用，確保我們的模型能夠識別序列中元素的順序。 

### 導讀

## 範例
- [Building a Simple Transformer using PyTorch [Code Included]](https://pureai.substack.com/p/building-a-simple-transformer-using-pytorch)
- https://github.com/ermattson/pure-ai-tutorials/tree/main/SimpleTransformer-PyTorch
