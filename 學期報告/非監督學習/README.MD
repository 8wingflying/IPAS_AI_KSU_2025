# 非監督學習主題
- 機器學習與非監督學習
  - [scikit-learn支援的非監督學習演算法](scikit-learn支援的非監督學習演算法.md)
- KERAS/Tensorflow 與非監督學習
- 深度學習之非監督學習

### 報告內容
- 非監督學習
- 非監督學習類型
- 非監督學習:降維
  - 維度的詛咒
  - 線性學習 ==> PCA
  - 非線性維度縮減| Manifold Learning
    - [LLE](LLE.md)
    - [t_SNE](t-SNE.md)
- 非監督學習:聚類
  - 四大類型_1_Centroid-based Methods: K-Means clustering
  - 四大類型_2_Distribution-based Methods ==>[Gaussian mixture models|高斯混合模型](高斯混合模型.md)
  - 四大類型_3_Connectivity based methods==> [Hierarchical clustering(階層式分群)](HierarchicalClustering.md)
  - 四大類型_4_Density Based methods==> [DBSCAN (Density-Based Spatial Clustering of Applications with Noise)](DBSCAN.md)
- 非監督學習評估指標
  - Silhouette score
  - Adjusted rand score 

### 教科書範例 ==> 降維 + 叢集法 + 高斯混合模型
- 第8章降維
  - 維度的詛咒
  - 降維的主要方法
  - [PCA](PCA.md) ==> 線性維度縮減 | 線性學習
    - 內核PCA ==> 非線性維度縮減
  - [LLE](LLE.md) ==> 非線性維度縮減| Manifold Learning
  - [t_SNE(推薦|補充)](t-SNE.md)
  - 其他降維技術
- 第9章非監督學習技術  [非監督學習之演算法類型](非監督學習之演算法類型.md)  [scikit-learn支援的非監督演算法](scikit-learn支援的非監督學習演算法.md)
  - 聚類
  - [高斯混合模型](高斯混合模型.md)

  
# 延伸學習 ==>非監督學習:書籍
- 👍[Hands-on Unsupervised Learning Using Python](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/) 
  - [GITHUB](https://github.com/aapatel09/handson-unsupervised-learning)
  - 繁體中譯本 [非監督式學習｜使用 Python](https://www.tenlong.com.tw/products/9789865024062?list_name=srh)
  - I. Fundamentals of Unsupervised Learning
    - `1`. Unsupervised Learning in the Machine Learning Ecosystem
    - `2`. End-to-End Machine Learning Project
  - II. Unsupervised Learning Using Scikit-Learn
    - `3`. Dimensionality Reduction(降維| 位度縮減)
    - `4`. Anomaly Detection(異常偵測)
    - `5`. Clustering(叢集法)
    - `6`. Group Segmentation(群組區隔) ==> 不同客群
  - III. Unsupervised Learning Using TensorFlow and Keras
    - 👍`7`. Autoencoders
    - `8`. Hands-On Autoencoder
    - `9`. Semisupervised Learning
  - IV. Deep Unsupervised Learning Using TensorFlow and Keras
    - 👍`10`. Recommender Systems Using Restricted Boltzmann Machines
    - `11`. Feature Detection Using Deep Belief Networks
    - `12`. Generative Adversarial Networks(另外書籍)
    - `13`. Time Series Clustering
- [Hands-On Unsupervised Learning with Python(2019)](https://www.packtpub.com/product/hands-on-unsupervised-learning-with-python/9781789348279)
  - 簡體中譯本 [Python 無監督學習](https://www.tenlong.com.tw/products/9787115540720?list_name=srh)
  - [GITHUB](https://github.com/PacktPublishing/Hands-on-Unsupervised-Learning-with-Python)
- [Applied Unsupervised Learning with Python](https://www.packtpub.com/product/applied-unsupervised-learning-with-python/9781789952292)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Topic Modeling
  - Market Basket Analysis
  - Hotspot Analysis 
- [The Unsupervised Learning Workshop(2020)](https://www.packtpub.com/product/the-unsupervised-learning-workshop/9781800200708) [GITHUB](https://github.com/PacktWorkshops/The-Unsupervised-Learning-Workshop)
- [Data Without Labels|Practical unsupervised machine learning](https://www.manning.com/books/data-without-labels)
  - https://github.com/vverdhan/DataWithoutLabels
