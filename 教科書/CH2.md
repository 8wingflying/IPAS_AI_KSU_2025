## 2.(端到端End-to-End)機器學習專案(Machine Learning Project)
- 本章節以一個範例說明機器學習專案
- 範例:使用 California 人口普查資料構建該州的房價模型 ==> regression
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
- 模型微調Fine-Tune Your Model
- Launch, Monitor, and Maintain Your System

### Fine-Tune Your Model模型微調
- 參看 [sklearn.model_selection](https://scikit-learn.org/stable/api/sklearn.model_selection.html)
  - Splitters
  - Hyper-parameter optimizers
    - 👍網格搜尋GridSearchCV
    - HalvingGridSearchCV
    - HalvingRandomSearchCV
    - ParameterGrid
    - ParameterSampler
    - 👍隨機搜尋RandomizedSearchCV

 
