# CH7.集成學習和隨機森林 ==> 競賽常勝軍
- voting 投票表決法(多數決,Majority vote)
  - hard voting vs soft voting
  - VotingClassifier
- Bagging and Pasting ==> 訓練集進行抽樣
  - Random Forest(隨機森林) == bag of tree(一堆樹木)
- Boosting ==> 逐步增強法
  - AdaBoost == Adaptive Boosting (二元分類) VS
    - SAMME ==Stagewise Additive Modeling using a Multiclass Exponential loss function(多元分類) <==SCIKIT-LEARN實作
  - Gradient Boosting ==> GradientBoostingClassifier
  - HistGradientBoostingClassifier 
- stacking


## sklearn.ensemble
## voting
- class sklearn.ensemble.VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)
```
sklearn.ensemble.VotingClassifier(
estimators,
*,
voting='hard', ==>  voting{‘hard 多數決’, ‘soft’}, default=’hard’
weights=None,
n_jobs=None,
flatten_transform=True,
verbose=False)
```
```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)

for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))

voting_clf.predict(X_test[:1])

[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]

voting_clf.score(X_test, y_test)

## 看最後總成績
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```
- soft voting
```python
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
```
## Bagging and Pasting
- bagging== bootstrap aggregating==>sampling with replacement
- pasting  ==>sampling without replacement
- https://www.geeksforgeeks.org/ml-bagging-classifier/
- BaggingClassifier
```
class sklearn.ensemble.BaggingClassifier(estimator=None,
n_estimators=10,
*,
max_samples=1.0,
max_features=1.0,
bootstrap=True(Bagging), ==>bootstrap=False(Pasting)
bootstrap_features=False, ==>
oob_score=False, ==> 
warm_start=False,
n_jobs=None,
random_state=None,
verbose=0)
```
- Out-of-Bag Evaluation
- Random Patches and Random Subspaces


## Random Forest  <== Decision Trees (一堆樹木成森林)
- Bagging(pasting) using Decision tree
- https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
```
class sklearn.ensemble.RandomForestClassifier(
n_estimators=100, *,
criterion='gini',
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.0,
max_features='sqrt',
max_leaf_nodes=None,
min_impurity_decrease=0.0,
bootstrap=True,
oob_score=False,
n_jobs=None,
random_state=None,
verbose=0,
warm_start=False,
class_weight=None,
ccp_alpha=0.0,
max_samples=None,
monotonic_cst=None)
```
```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```
#### 證實
- A Random Forest is equivalent to a bag of decision trees

### extra-tree
```python
class sklearn.ensemble.ExtraTreesClassifier(
n_estimators=100, *,
criterion='gini',
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.0,
max_features='sqrt',
max_leaf_nodes=None,
min_impurity_decrease=0.0,
bootstrap=False,
oob_score=False,
n_jobs=None,
random_state=None,
verbose=0,
warm_start=False,
class_weight=None,
ccp_alpha=0.0,
max_samples=None,
monotonic_cst=None)
```
### Stacking
```python
class sklearn.ensemble.StackingClassifier(
estimators, ==> 第一階段的學習器(個別的學習器)
final_estimator=None, ==> 最後的學習器(meta-learner)
*,
cv=None, ==> cross-validation generator
stack_method='auto', ==> 有四種
n_jobs=None,
passthrough=False,
verbose=0)
```
```python
clf = StackingClassifier(
      estimators=estimators, final_estimator=LogisticRegression()
      )
```
```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],

## meta-learner
    final_estimator=RandomForestClassifier(random_state=43),

    cv=5  # number of cross-validation folds
)
stacking_clf.fit(X_train, y_train)
```
