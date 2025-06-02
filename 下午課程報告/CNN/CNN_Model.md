## 使用內建MODEL
```
from tensorflow.keras.applications import Xception
xception = Xception()
xception.summary()
```
- [Keras 內建CNN MODEL| Keras Applications](https://keras.io/api/applications/)
## CNN Model
- 📝[A Survey of the Recent Architectures of Deep Convolutional Neural Networks(abs/1901.06032)](https://arxiv.org/abs/1901.06032)
- 📝[A Comprehensive Survey on Architectural Advances in Deep CNNs: Challenges, Applications, and Emerging Research Directions()](https://arxiv.org/abs/2503.16546)
- 🖊️LeNet(1989)
- 📅LeNet-5 (1998)
- 重要發展
  - GPU (2006)
  - NVIDIA (2007)
  - ImageNet(2010-2017) https://image-net.org/
- AlexNet(2012)
  - AlexNet was trained in parallel on two NVIDIA GTX 580 GPUs 
- ZFNet(2013)
- ⭐GoogLeNet/Inception(2014)
  -  Inception V2 ==> V3 ==> V4 
- VGGNet (2014)
- Residual Neural Network (ResNet)(2015) |
- DenseNet(2016)
  - [Densely Connected Convolutional Networks(arXiv:1608.06993)](https://arxiv.org/abs/1608.06993)
  - 相較於ResNet, DenseNet 可用更少的訓練參數取得更佳的準確率
  - CVPR 2017 最佳論文
  - Keras 提供數個DenseNet 版本供我們使用： DenseNet 121 、DenseNet 169 、DenseNet201
- SENet（Squeeze-and-Excitation Networks）(2017)
  - ImageNet 2017 競賽冠軍
  - [論文 Squeeze-and-Excitation Networks(arXiv:1709.01507)](https://arxiv.org/abs/1709.01507)
  - https://github.com/hujie-frank/SENet
- ShuffleNet(2017) 一種針對移動設備的極高效卷積神經網路
  - [論文ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices(arXiv:1707.01083)](https://arxiv.org/abs/1707.01083) 

## Vision Transformers (ViT)系列模型 2020-2021
- 應用Transformers 到電腦視覺的各種模型
- Vision Transformers (ViTs)(2020)
  - [論文An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale(arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)
  - Vision Transformer architecture comprises several key stages:
    - Image Patching and Embedding
    - Positional Encoding
    - Transformer Encoder
    - Classification Head (MLP Head) 
  - https://blog.csdn.net/abc13526222160/article/details/131228810
- DeiT (Data-efficient Image Transformer)
  - [[2012.12877]Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
  - https://blog.csdn.net/abc13526222160/article/details/132339050
- 微軟Swin Transformer
  - [[2103.14030] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows ](https://arxiv.org/abs/2103.14030)
  - https://www.geeksforgeeks.org/swin-transformer/
- 微軟 CvT (Convolutional Vision Transformer)
  - [[2103.15808] CvT: Introducing Convolutions to Vision Transformers ](https://arxiv.org/abs/2103.15808)
  - https://zhuanlan.zhihu.com/p/583450848
- T2T-ViT (Tokens-to-Token Vision Transformer)
  - [[2101.11986] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)
  - https://zhuanlan.zhihu.com/p/460938219  

## Vision Language Models ==> 邁向 MLLM(多模態大語言模型) 2021/2022
- CLIP (Contrastive Language-Image Pre-training)
  - 對比語言-圖像預訓練 （CLIP） 是一種使用對比目標訓練一對神經網路模型的技術，一個用於圖像理解，一個用於文本理解。
  - 這種方法在多個領域實現了廣泛的應用，包括跨模態檢索 、文本到圖像生成
  - https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training
  - [Contrastive Language-Image Pre-Training with Knowledge Graphs](https://arxiv.org/abs/2210.08901) 
- ALIGN (A Large-scale ImaGe and Noisy-text)
  - [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) 
- BLIP (Bootstrapping Language-Image Pre-training)
  - [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) 
