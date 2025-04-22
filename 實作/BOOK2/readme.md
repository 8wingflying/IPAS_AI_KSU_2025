#### [精通機器學習｜使用 Scikit-Learn , Keras 與 TensorFlow, 3/e ](https://www.tenlong.com.tw/products/9786263246676?list_name=srh)
- https://github.com/ageron/handson-ml2
- https://github.com/ageron/handson-ml3
- https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/

#### 第一部分機器學習的基礎知識11
```
第1章機器學習概覽13
1.1 什麼是機器學習14
1.2 為什麼使用機器學習14
1.3 機器學習的應用示例16
1.4 機器學習系統的類型18
1.5 機器學習的主要挑戰32
1.6 測試與驗證38
1.7 練習題40

第2章端到端的機器學習項目42
2.1 使用真實數據42
2.2 觀察大局44
2.3 獲取數據48
2.4 從數據探索和可視化中獲得洞見60
2.5 機器學習算法的數據準備66
2.6 選擇和訓練模型74
2.7 微調模型77
2.8 啟動、監控和維護你的系統82
2.9 試試看84
2.10 練習題84

Look at the big picture.
Get the data.
Explore and visualize the data to gain insights.
Prepare the data for machine learning algorithms.
Select a model and train it.
Fine-tune your model.
Present your solution.
Launch, monitor, and maintain your system.

第3章分類(Classification)
3.1 MNIST86
3.2 訓練二元分類器(Binary Classifier)
3.3 性能測量(Performance Measures)
3.4 多類分類器(Multiclass Classification)
3.5 誤差分析(Error Analysis)
3.6 多標籤分類(Multilabel Classification)
3.7 多輸出分類(Multioutput Classificatio)

第4章訓練模型==>回歸(regression)
4.1 線性回歸(Linear Regression)
  The Normal Equation
  Computational Complexity
4.2 梯度下降(Gradient Descent)
  Batch Gradient Descent
  Stochastic Gradient Descent
  Mini-Batch Gradient Descent
4.3 多項式回歸(Polynomial Regression)
4.4 學習曲線(Learning Curves)
4.5 正則化線性模型(Regularized Linear Models)
  Ridge Regression
  Lasso Regression
  Elastic Net Regression
  Early Stopping
4.6 邏輯回歸(Logistic Regression) ==>二元分類
    Estimating Probabilities
    Training and Cost Function
    Decision Boundaries
  Softmax Regression ==> 多類別分類


第5章支持向量機143
5.1 線性SVM分類(Linear SVM Classification)
5.2 非線性SVM分類(Nonlinear SVM Classification)
  Polynomial Kernel
  Similarity Features
  Gaussian RBF Kernel
  SVM Classes and Computational Complexity
5.3 SVM回歸(SVM Regression)
5.4 工作原理
  Under the Hood of Linear SVM Classifiers
  The Dual Problem
     Kernelized SVMs



第6章決策樹162
6.1 訓練和可視化決策樹162
6.2 做出預測163
6.3 估計類概率165
6.4 CART訓練算法166
6.5 計算複雜度166
6.6 基尼不純度或熵167
6.7 正則化超參數167
6.8 回歸168
6.9 不穩定性170
6.10 練習題172
```
- Ensemble Learning|集成學習
```
第7章集成學習和隨機森林173
7.1 投票分類器(Voting Classifiers): Soft vs Hard
7.2 bagging和pasting
  Out-of-Bag Evaluation
  7.3 隨機補丁(Random Patches)和隨機子空間(Random Subspaces)
7.4 隨機森林(Random Forests)
  Extra-Trees
  Feature Importance
7.5 提升法(Boosting)
  AdaBoost
  Gradient Boosting
  Histogram-Based Gradient Boosting
7.6 堆疊法(Stacking)
```
- 非監督學習
```
第8章降維193
8.1 維度的詛咒194
8.2 降維的主要方法195
8.3 PCA198
8.4 內核PCA204
8.5 LLE206
8.6 其他降維技術208
8.7 練習題209

第9章非監督學習技術211
9.1 聚類212
9.2 高斯混合模型232
9.3 練習題245
```
#### 第二部分神經網絡與深度學習247
````
第10章Keras人工神經網絡簡介249
10.1 從生物神經元到人工神經元250
10.2 使用Keras實現MLP262
10.3 微調神經網絡超參數284
10.4 練習題290

第11章訓練深度神經網絡293
11.1 梯度消失與梯度爆炸問題293
11.2 重用預訓練層305
11.3 更快的優化器310
11.4 通過正則化避免過擬合321
11.5 總結和實用指南327
11.6 練習題329

第12章使用TensorFlow自定義模型和訓練330
12.1 TensorFlow快速瀏覽330
12.2 像NumPy一樣使用TensorFlow333
12.3 定製模型和訓練算法338
12.4 TensorFlow函數和圖356
12.5 練習題360

第13章使用TensorFlow加載和預處理數據362
13.1 數據API363
13.2 TFRecord格式372
13.3 預處理輸入特徵377
13.4 TF Transform385
13.5 TensorFlow數據集項目386
13.6 練習題388

第14章使用捲積神經網絡的深度計算機視覺390
14.1 視覺皮層的架構390
14.2 捲積層392
14.3 池化層399
14.4 CNN架構402
14.5 使用Keras實現ResNet-34 CNN416
14.6 使用Keras的預訓練模型417
14.7 遷移學習的預訓練模型418
14.8 分類和定位421
14.9 物體檢測422
14.10 語義分割428
14.11 練習題431

第15章使用RNN和CNN處理序列432
15.1 循環神經元和層432
15.2 訓練RNN436
15.3 預測時間序列437
15.4 處理長序列444
15.5 練習題453

第16章使用RNN和註意力機制進行自然語言處理455
16.1 使用字符RNN生成莎士比亞文本456
16.2 情感分析464
16.3 神經機器翻譯的編碼器-解碼器網絡470
16.4 註意力機制476
16.5 最近語言模型的創新486
16.6 練習題488

第17章使用自動編碼器和GAN的表徵學習和生成學習489
17.1 有效的數據表徵490
17.2 使用不完整的線性自動編碼器執行PCA491
17.3 堆疊式自動編碼器493
17.4 捲積自動編碼器499
17.5 循環自動編碼器500
17.6 去噪自動編碼器501
17.7 稀疏自動編碼器502
17.8變分自動編碼器505
17.9 生成式對抗網絡510
17.10 練習題522

第18章強化學習523
18.1 學習優化獎勵524
18.2 策略搜索525
18.3 OpenAI Gym介紹526
18.4 神經網絡策略529
18.5 評估動作：信用分配問題531
18.6 策略梯度532
18.7 馬爾可夫決策過程536
18.8 時序差分學習540
18.9 Q學習540
18.10 實現深度Q學習544
18.11 深度Q學習的變體547
18.12 TF-Agents庫550
18.13 一些流行的RL算法概述568
18.14 練習題569

第19章大規模訓練和部署TensorFlow模型571
19.1 為TensorFlow模型提供服務572
19.2 將模型部署到移動端或嵌入式設備586
19.3 使用GPU加速計算589
19.4 跨多個設備的訓練模型600
19.5 練習題613
19.6 致謝613

附錄A 課後練習題解答614
附錄B 機器學習項目清單642
附錄C SVM對偶問題647
附錄D 自動微分650
附錄E 其他流行的人工神經網絡架構656
附錄F 特殊數據結構663
附錄G TensorFlow圖669
```
