## CNN Model
- 📝[A Survey of the Recent Architectures of Deep Convolutional Neural Networks(abs/1901.06032)](https://arxiv.org/abs/1901.06032)
- 📝[A Comprehensive Survey on Architectural Advances in Deep CNNs: Challenges, Applications, and Emerging Research Directions()](https://arxiv.org/abs/2503.16546)
- 🖊️LeNet(1989)
- 📅LeNet-5 (1998)
- GPU (2006)
- NVIDIA (2007)
- ImageNet(2010)
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
- Vision Transformers (ViTs)(2020)
  - [論文An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale(arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)
  - Vision Transformer architecture comprises several key stages:
    - Image Patching and Embedding
    - Positional Encoding
    - Transformer Encoder
    - Classification Head (MLP Head) 
  - https://blog.csdn.net/abc13526222160/article/details/131228810
- 使用內建MODEL
```
from tensorflow.keras.applications import Xception
xception = Xception()
xception.summary()
```
