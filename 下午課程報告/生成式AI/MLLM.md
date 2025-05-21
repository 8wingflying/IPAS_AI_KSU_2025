## MMLM
- 多模態指令調整（MIT）
- 多模態上下文學習（M-ICL）
- 多模態思想鏈（M-CoT）
- LLM輔助視覺推理（LAVR）

## 導讀
## Survey
- [[2306.13549] A Survey on Multimodal Large Language Models](https://arxiv.org/abs/2306.13549)
- 一個典型的 MLLM 可以抽象為三個模組
  - 一個預先訓練的`模態編碼器(Modality encoder)`
    - 編碼器接收圖像、音訊或視頻並輸出特徵，這些功能由連接器處理，以便 LLM 能夠更好地理解 
  - 一個預先訓練的 LLM
  - 一個用於連接它們的模態介面
    - 連接器大致有三種類型：基於投影的連接器、基於查詢的連接器和基於融合的連接器。
    - 前兩種類型採用 Token 級融合，將特徵處理成 Token 並與文本 Token 一起發送
    - 最後一種類型在 LLM 內部實現特徵級融合。 
- 一些 MLM 還包括一個生成器，用於輸出除文本之外的其他模態
- `3`.訓練策略和數據
  - 一個成熟的 MLLM 會經歷三個階段的訓練，即`預訓練(Pre-training)`、`教學調整(instruction-tuning)`和`對齊調整(alignment tuning)`。
  - 訓練的每個階段都需要不同類型的數據並實現不同的目標。 
