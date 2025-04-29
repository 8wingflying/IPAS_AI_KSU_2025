## Support Vector Machines
- 分類問題 ==> Support Vector發現
- hard vs soft
  - Soft Margin Classification
  - Soft Margin Classification 
- Linear vs Nonlinerar
  - Linear SVM Classification 
  - Nonlinear SVM Classification
    - Polynomial Kernel
    - Gaussian RBF Kernel
    - Sigmod Kernel => see Equation 5-7. Common kernels
- sklearn.svm模組說明 
  - LinearSVC vs SVC  ==> C
  - 主要區別在於預設使用的損失函數
  - multi-class classification ==>  不同程式的實作差異
    - see https://scikit-learn.org/stable/modules/svm.html#kernel-functions
    - SVC and NuSVC implement the “one-versus-one” approach for multi-class classification.
      - In total, n_classes * (n_classes - 1) / 2 classifiers are constructed and each one trains data from two classes
    - LinearSVC implements “one-vs-the-rest” multi-class strategy, thus training n_classes models. 
- 工作原理
  - Equation 5-1. Hard margin linear SVM classifier objective
  - Equation 5-2. Soft margin linear SVM classifier objective(引入 C)
  - Dual problem
  - training(兩種方式)
    - Using a QP solver
    - use gradient descent to minimize the hinge loss or the squared hinge loss  
- 進階主題
  - One-class SVM with non-linear kernel (RBF)
  - Plot classification boundaries with different SVM Kernels
  - Plot different SVM classifiers in the iris dataset
  - Plot the support vectors in LinearSVC
  - RBF SVM parameters
  - SVM Margins Example
  - SVM Tie Breaking Example
  - SVM with custom kernel
  - SVM-Anova: SVM with univariate feature selection
  - SVM: Maximum margin separating hyperplane
  - SVM: Separating hyperplane for unbalanced classes
  - SVM: Weighted samples
  - Scaling the regularization parameter for SVCs
  - Support Vector Regression (SVR) using linear and non-linear kernels 
- 作業
  - 第10題==>SVM 分類 ==>Train an SVM classifier on the wine dataset==> sklearn.datasets.load_wine()
  - 第11題==>SVM 回歸 ==>Train and fine-tune an SVM regressor on the California housing dataset
    - sklearn.datasets.fetch_california_housing() 
### 簡介
- 支援向量機 （SVM） 是一種功能強大且用途廣泛的機器學習模型，能夠執行線性或非線性分類(SVC)、回歸(SVR)，甚至新穎性檢測。
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

## sklearn.svm: SVC vs LinearSVC  
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
```
class sklearn.svm.LinearSVC(
penalty='l2',    ===> 
loss='squared_hinge',   ===> 
*,
dual='auto',
tol=0.0001,
C=1.0,
multi_class='ovr',
fit_intercept=True,
intercept_scaling=1,
class_weight=None,
verbose=0,
random_state=None,
max_iter=1000)
```

###  [sklearn.preprocessing](https://scikit-learn.org/stable/api/sklearn.preprocessing.html)
- Figure 5-2. Sensitivity to feature scales
- Figure 5-3. Hard margin sensitivity to outliers
- Figure 5-4. Large margin (left) versus fewer margin violations (right)
- The LinearSVC class is much faster than SVC(kernel="linear")
#### 非線性SVC 
- Figure 5-5. Adding features to make a dataset linearly separable

- Figure 5-7. SVM classifiers with a `polynomial` kernel 
- linear SCM
```python
from sklearn.svm import LinearSVC

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, dual=True, random_state=42))
```
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica

svm_clf = make_pipeline(StandardScaler(),
                        LinearSVC(C=1, dual=True, random_state=42))
svm_clf.fit(X, y)
```
- nonlinear SVM
```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, dual=True, random_state=42)
)
polynomial_svm_clf.fit(X, y)
```
- SEE Figure 5-6. Linear SVM classifier using `polynomial` features(特徵)
```python
# Polynomial Kernel
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(StandardScaler(),
                                    SVC(kernel="poly", degree=3, coef0=1, C=5))
poly_kernel_svm_clf.fit(X, y)
```
```python
rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X, y)
```
- Figure 5-9. SVM classifiers using an RBF kernel
#### Similarity Features
- Figure 5-8. Similarity features using the Gaussian RBF

#### 線性回歸|SVM Regression
```python
from sklearn.svm import LinearSVR

# extra code – these 3 lines generate a simple linear dataset
np.random.seed(42)
X = 2 * np.random.rand(50, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(50)

svm_reg = make_pipeline(StandardScaler(),
                        LinearSVR(epsilon=0.5, dual=True, random_state=42))
svm_reg.fit(X, y)
```
```python
from sklearn.svm import SVR

# extra code – these 3 lines generate a simple quadratic dataset
np.random.seed(42)
X = 2 * np.random.rand(50, 1) - 1
y = 0.2 + 0.1 * X[:, 0] + 0.5 * X[:, 0] ** 2 + np.random.randn(50) / 10

svm_poly_reg = make_pipeline(StandardScaler(),
                             SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1))
svm_poly_reg.fit(X, y)
```
