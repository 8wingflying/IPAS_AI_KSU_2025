## Support Vector Machines
- 支援向量機 （SVM） 是一種功能強大且用途廣泛的機器學習模型，能夠執行線性或非線性分類、回歸，甚至新穎性檢測。
- SVM 適用於`中小型`非線性數據集（即數百到數千個實例），尤其是對於分類任務。
- 它們不能很好地擴展到非常大的數據集==> random forest

## [sklearn.svm](https://scikit-learn.org/stable/api/sklearn.svm.html)
- https://scikit-learn.org/stable/modules/svm.html
- Linear Support Vector Classification ==> LinearSVC | LinearSVR
- NonLinear Support Vector Classification
- SVM 對特徵尺度很敏感 ==> 特徵縮放 ==>  StandardScaler
- Hard Margin Classification ==> all instances must be off the street and on the correct side
  - Two problems:
    - only works if the data is linearly separable
    - 對異常值很敏感 see fig 5-3
- Soft Margin Classification ==> margin violations
- Regularization parameter正則化超參數
  - See  Fig 5-4
    - 小 c ==> 結果差
    - 大 c ==> 結果比較好 
  - SVM 模型 overfitting ==> 降低C 來正則化
## sklearn.svm
- The multiclass support is handled according to a `one-vs-one` scheme.
```
class sklearn.svm.SVC(
*, 
C=1.0, ==> Regularization parameter正則化超參數 see fig 5-4   ==> The penalty is a squared l2 penalty
kernel='rbf',  =========>  kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
degree=3, 
gamma='scale', 
coef0=0.0,
shrinking=True,
probability=False,
tol=0.001,
cache_size=200,
class_weight=None,
verbose=False,
max_iter=-1,
decision_function_shape='ovr',
break_ties=False,
random_state=None)
```

###  [sklearn.preprocessing](https://scikit-learn.org/stable/api/sklearn.preprocessing.html)

