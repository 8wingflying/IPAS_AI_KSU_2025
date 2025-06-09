### [Hands-On Large Language Models](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/)
- https://github.com/HandsOnLLM/Hands-On-Large-Language-Models

## I. 理解語言模型
```
1. 大型語言模型簡介
2. Tokens And Embeddings
3. 深入瞭解大型語言模型
```
## II. 使用預訓練語言模型(Using Pretrained Language Models)
```
4. 文字分類(Text Classification)
5. 文字聚類和主題建模(Text Clustering And Topic Modeling)
6. 提示工程(Prompt Engineering)
7. 高級文本生成技術和工具(Advanced Text Generation Techniques And Tools)
8. 語義搜索和檢索 - 增強生成(Semantic Search And Retrieval-Augmented Generation)
9. 多模態大型語言模型(Multimodal Large Language Models)
```
## III. 訓練和微調語言模型(Training And Fine-Tuning Language Models)
```
10. 建立`文字嵌入模型(Text Embedding Models)`
11. 微調分類的表示模型(Fine-Tuning Representation Models For Classification)
12. 微調生成模型
```
### [大語言模型工程師手冊：從概念到生產實踐](https://www.tenlong.com.tw/products/9787115667373?list_name=srh)
- [LLM Engineer's Handbook: Master the art of engineering large language models from concept to production](https://www.tenlong.com.tw/products/9787115667373?list_name=srh)

```
第 1章　理解LLM Twin的概念與架構　1
1.1　理解LLM Twin的概念　1
1.1.1　什麽是LLM Twin　2
1.1.2　為什麽構建LLM Twin　3
1.1.3　為什麽不使用ChatGPT（或其他類似的聊天機器人）　4
1.2　規劃LLM Twin的MVP　5
1.2.1　什麽是MVP　5
1.2.2　定義LLM Twin的MVP　5
1.3　基於特徵、訓練和推理流水線構建機器學習系統　6
1.3.1　構建生產級機器學習系統的挑戰　6
1.3.2　以往解決方案的問題　8
1.3.3　解決方案：機器學習系統的流水線　10
1.3.4　FTI流水線的優勢　11
1.4　設計LLM Twin的系統架構　12
1.4.1　列出LLM Twin架構的技術細節　12
1.4.2　使用FTI流水線設計LLM Twin架構　13
1.4.3　關於FTI流水線架構和LLM Twin架構的最終思考　17
1.5　小結　18

第 2章　工具與安裝　19
2.1　Python生態環境與項目安裝　20
2.1.1　Poetry：Python項目依賴與環境管理利器　21
2.1.2　Poe the Poet：Python 項目任務管理神器　22
2.2　MLOps與MLOps工具生態　23
2.2.1　Hugging Face：模型倉庫　23
2.2.2　ZenML：編排、工件和元數據　24
2.2.3　Comet ML：實驗跟蹤工具　33
2.2.4　Opik：提示監控　34
2.3　用於存儲NoSQL和向量數據的數據庫　35
2.3.1　MongoDB：NoSQL數據庫　35
2.3.2　Qdrant：向量數據庫　35
2.4　為AWS做準備　36
2.4.1　設置AWS賬戶、訪問密鑰和CLI　36
2.4.2　SageMaker：訓練與推理計算　37
2.5　小結　39

第3章　數據工程(Data Engineering)
3.1　設計LLM Twin的數據採集流水線　41
3.1.1　實現LLM Twin數據採集流水線　44
3.1.2　ZenML流水線及其步驟　44
3.1.3　分發器：實例化正確的爬蟲　48
3.1.4　爬蟲　50
3.1.5　NoSQL數據倉庫文檔　59
3.2　採集原始數據並存儲到數據倉庫　67
3.3　小結　71

第4章　RAG特徵流水線(RAG Feature Pipeline)
4.1　理解RAG　73
4.1.1　為什麽使用RAG　74
4.1.2　基礎RAG框架　75
4.1.3　什麽是嵌入　78
4.1.4　關於向量數據庫的更多內容　84
4.2　高級RAG技術概覽　86
4.2.1　預檢索　87
4.2.2　檢索　90
4.2.3　後檢索　91
4.3　探索LLM Twin的RAG特徵流水線架構　93
4.3.1　待解決的問題　93
4.3.2　特徵存儲　94
4.3.3　原始數據從何而來　94
4.3.4　設計RAG特徵流水線架構　94
4.4　實現LLM Twin的RAG特徵流水線　101
4.4.1　配置管理　101
4.4.2　ZenML流水線與步驟　102
4.4.3　Pydantic領域實體　109
4.4.4　分發器層　116
4.4.5　處理器　117
4.5　小結　125

第5章　監督微調(Supervised Fine-Tuning)
5.1　構建指令訓練數據集　127
5.1.1　構建指令數據集的通用框架　128
5.1.2　數據管理　130
5.1.3　基於規則的過濾　131
5.1.4　數據去重　132
5.1.5　數據凈化　133
5.1.6　數據質量評估　133
5.1.7　數據探索　136
5.1.8　數據生成　138
5.1.9　數據增強　139
5.2　構建自定義指令數據集　140
5.3　探索SFT及其關鍵技術　148
5.3.1　何時進行微調　148
5.3.2　指令數據集格式　149
5.3.3　聊天模板　150
5.3.4　參數高效微調技術　151
5.3.5　訓練參數　155
5.4　微調技術實踐　158
5.5　小結　164

第6章　偏好對齊微調(Fine-Tuning with Preference Alignment)
6.1　理解偏好數據集　165
6.1.1　偏好數據　166
6.1.2　數據生成與評估　168
6.2　構建個性化偏好數據集　171
6.3　偏好對齊　177
6.3.1　基於人類反饋的強化學習　178
6.3.2　DPO　179
6.4　實踐DPO　181
6.5　小結　187

第7章　LLM的評估方法
7.1　模型能力評估　188
7.1.1　機器學習與LLM評估的對比　188
7.1.2　通用LLM評估　189
7.1.3　領域特定LLM評估　191
7.1.4　任務特定LLM評估　193
7.2　RAG系統的評估　195
7.2.1　Ragas　196
7.2.2　ARES　197
7.3　TwinLlama-3.1-8B模型評估　198
7.3.1　生成答案　199
7.3.2　答案評估　200
7.3.3　結果分析　204
7.4　小結　207

第8章　模型推理性能優化(Inference Optimization)
8.1　模型優化方法　208
8.1.1　KV cache　209
8.1.2　連續批處理　211
8.1.3　投機解碼　212
8.1.4　優化的註意力機制　214
8.2　模型並行化　215
8.2.1　數據並行　215
8.2.2　流水線並行　216
8.2.3　張量並行　217
8.2.4　組合使用並行化方法　218
8.3　模型量化　219
8.3.1　量化簡介　219
8.3.2　基於GGUF和llama.cpp的模型量化　223
8.3.3　GPTQ和EXL2量化技術　225
8.3.4　其他量化技術　226
8.4　小結　227

第9章　RAG推理流水線(RAG Inference Pipeline)
9.1　理解LLM Twin的RAG推理流水線　229
9.2　探索LLM Twin的高級RAG技術　230
9.2.1　高級RAG預檢索優化：查詢擴展與自查詢　233
9.2.2　高級RAG檢索優化：過濾向量搜索　239
9.2.3　高級RAG後檢索優化：重排序　240
9.3　構建基於RAG的LLM Twin推理流水線　243
9.3.1　實現檢索模塊　243
9.3.2　整合RAG推理流水線　249
9.4　小結　254

第 10章　推理流水線部署(Inference Pipeline Deployment)
10.1　部署方案的選擇　256
10.1.1　吞吐量和延遲　256
10.1.2　數據　256
10.1.3　基礎設施　257
10.2　深入理解推理部署方案　258
10.2.1　在線實時推理　259
10.2.2　異步推理　260
10.2.3　離線批量轉換　260
10.3　模型服務的單體架構與微服務架構　261
10.3.1　單體架構　261
10.3.2　微服務架構　262
10.3.3　單體架構與微服務架構的 選擇　264
10.4　探索LLM Twin的推理流水線部署 方案　265
10.5　部署LLM Twin服務　268
10.5.1　基於AWS SageMaker構建LLM 微服務　268
10.5.2　使用FastAPI構建業務 微服務　282
10.6　自動縮放應對突發流量高峰　285
10.6.1　註冊縮放目標　287
10.6.2　創建彈性縮放策略　287
10.6.3　縮放限制的上下限設置　288
10.7　小結　289

第 11章　MLOps與LLMOps　290
11.1　LLMOps發展之路：從DevOps和 MLOps尋根　291
11.1.1　DevOps　291
11.1.2　MLOps　293
11.1.3　LLMOps　296
11.2　將LLM Twin流水線部署到雲端　299
11.2.1　理解基礎架構　300
11.2.2　MongoDB環境配置　301
11.2.3　Qdrant環境配置　302
11.2.4　設置ZenML雲環境　303
11.3　為LLM Twin添加LLMOps　313
11.3.1　LLM Twin的CI/CD流水線 工作流程　313
11.3.2　GitHub Actions快速概覽　316
11.3.3　CI流水線　316
11.3.4　CD流水線　320
11.3.5　測試CI/CD流水線　322
11.3.6　CT流水線　323
11.3.7　提示監控　327
11.3.8　告警　332
11.4　小結　332
附錄　MLOps原則　334
```
