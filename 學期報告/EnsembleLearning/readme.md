# 集成學習 ==> 競賽常勝軍
- voting 投票表決法(多數決,Majority vote)
  - hard voting vs soft voting
  - VotingClassifier
- Bagging and Pasting ==> 訓練集進行抽樣
  - 👍Random Forest(隨機森林) == bag of tree(一堆樹木)
- Boosting ==> 逐步增強法
  - 👍AdaBoost == Adaptive Boosting (二元分類) VS
    - SAMME ==Stagewise Additive Modeling using a Multiclass Exponential loss function(多元分類) <==SCIKIT-LEARN實作
  - 👍Gradient Boosting ==> GradientBoostingClassifier
  - HistGradientBoostingClassifier 
- stacking

## scikit-learn 支援的協定
- [sklearn.ensemble](https://scikit-learn.org/stable/api/sklearn.ensemble.html)
- [使用者指南:1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking](https://scikit-learn.org/stable/modules/ensemble.html)
