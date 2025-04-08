## I. 機器學習的基礎知識

1. 機器學習`前景`The Machine Learning `Landscape`
- 什麼是機器學習？
- 為什麼使用機器學習？
- 應用實例
- 機器學習系統的類型
  - 培訓監督Training Supervision
  - 批量學習與在線學習`Batch Learning` VS `Online Learning`
  - 基於實例的學習 與 基於模型的學習`Instance-Based` Learning VS `Model-Based` Learning
- 機器學習的主要挑戰(Main Challenges)
- 測試和驗證(Testing and Validating)
  - 超參數調整和模型選擇(Hyperparameter Tuning and Model Selection)
  - 資料不匹配(Data Mismatch)

2.(端到端End-to-End)機器學習專案(Machine Learning Project)
- 本章節以一個範例說明機器學習專案
- 範例:使用 California 人口普查資料構建該州的房價模型
- Working with Real Data
- Look at the Big Picture
  - 確定問題
  - 選擇`Performance Measure(績效度量)`
    - Root mean square error (RMSE) ==> regression
    - Mean absolute error (MAE) | average absolute deviation
  - 檢查假設 
- Get the Data
  - 可視化地理數據
  - 查找`相關性Correlations`
    - standard correlation coefficient | Pearson’s r  
  - 善用不同`屬性組合(Attribute Combinations)`來進行實驗 
- 資料檢視與資料視覺化(Explore and Visualize the Data to Gain Insights)
- 資料準備(Data Preparation) ==>Prepare the Data for Machine Learning Algorithms
- Select and Train a Model
- Fine-Tune Your Model
- Launch, Monitor, and Maintain Your System

3. 分類(Classification) ==> 監督學習 | 分類(Classification)
- MNIST資料集
- 二元分類(Training a Binary Classifier)
- 績效衡量標準(Performance Measures)
  - Measuring `Accuracy` Using Cross-Validation
  - Confusion Matrices
  - Precision and Recall
  - The Precision/Recall Trade-off
  - The ROC Curve 
- 多`類別`分類(Multi`class` Classification)
- 誤差分析(Error Analysis)
- 多`標籤`分類Multi`label` Classification
- 多`輸出`分類Multi`output` Classification

4. 訓練模型  ==> 監督學習 | 回歸(Regression)
- 線性回歸(Linear Regression)
- 梯度下降(Gradient Descent)
  - `Batch` Gradient Descent
  - `Stochastic` Gradient Descent
  - `Mini-Batch` Gradient Descent 
- 多項式回歸(Polynomial Regression)
- 學習曲線(Learning Curves)
- 正則化線性模型(Regularized Linear Models)
  - Ridge Regression|Tikhonov regularization ==> L2 norm(penality)
  - Lasso Regression ==> L1 norm(penality)
    - Least absolute shrinkage and selection operator regression 
  - Elastic Net Regression ==> r*(L1 norm) + (1-r)*(L2 norm)
    - r = 0 ==> Elastic Net Regression --> Ridge Regression
    - r = 1 ==> Elastic Net Regression --> Lasso Regression
  - Early Stopping
- Logistic 回歸(Logistic Regression)|logit regression ==> 二元分類
  - softmax regression | multinomial logistic regression. ==> 多類別分類

5. 支援向量機(Support Vector Machine)  ==>  可以用到 分類(Classification) 與 回歸(Regression)

6. 決策樹(Decision Trees) ==>  可以用到 分類(Classification) 與 回歸(Regression)

7. Ensemble學習和`隨機森林(Random Forests)` ==>  競賽常勝軍

8. 降維(Dimensionality Reduction)  ==>  非監督學習
- The Curse of Dimensionality
- 兩大類做法(Main Approaches for Dimensionality Reduction)
  - Projection
  - Manifold Learning
    - https://scikit-learn.org/stable/modules/manifold.html 
- PCA | Principal component analysis ==> Compression(壓縮)
  - Randomized PCA
  - Incremental PCA 
- Random Projection
  - `Gaussian`RandomProjection
  - `Sparse`RandomProjection 
- LLE | Locally linear embedding ==> nonlinear dimensionality reduction (NLDR)
- Other Dimensionality Reduction Techniques ==> Scikit-Learn支援其他夠多的Dimensionality Reduction
  - https://scikit-learn.org/stable/modules/decomposition.html
  - 2.5.1.4. Sparse principal components analysis
  - 2.5.2. Kernel Principal Component Analysis (kPCA)
  - ...
  - 2.5.8. Latent Dirichlet Allocation (LDA)

10. 非監督學習技術
- Clustering Algorithms: k-means and DBSCAN
  - https://scikit-learn.org/stable/modules/clustering.html#clustering
    - 參看 A comparison of the clustering algorithms in scikit-learn 
  - k-means
  - DBSCAN 
- Gaussian Mixtures Model
  - https://scikit-learn.org/stable/modules/mixture.html 
