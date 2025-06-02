## GenAI
- VAE
- DCGAN
  - MNIST dataset ==> [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
  - Celeb-A Faces dataset ==>PyTorch [DCGAN Tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  - 
  - [Images of LEGO Bricks|40,000 images of 50 different LEGO bricks|Kaggle](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images)
    - [Generative Deep Learning, 2nd Edition](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174/)
      - CH `4`. Generative Adversarial Networks
      - DCGAN | WGAN-GP | CGAN 
    - [GAN for Lego Bricks](https://www.kaggle.com/code/gabrielcabas/gan-for-lego-bricks)
- [Diffusion model](DIFFUSION_2025.md)
  - 基本觀念 https://ithelp.ithome.com.tw/articles/10329715
  - 範例程式 
## GenAI
- VAE 2013
  - Stacked Autoencoders
    - Implementing a Stacked Autoencoder Using Keras
    - Unsupervised Pretraining Using Stacked Autoencoders
  - Training One Autoencoder at a Time
  - Convolutional Autoencoders
  - Denoising Autoencoders
  - Sparse Autoencoders
  - Variational Autoencoders 
- GAN(2014) Generative Adversarial Networks
  - The Difficulties of Training GANs 
  - Deep Convolutional GANs(DCGANs)
  - 2018 NVIDIA | Progressive Growing of GANs
  - 2018 NVIDIA |StyleGANs
  - WGAN
  - WGAN-GP
    - WGAN-GP相對於WGAN的改進很小，除了增加了一個正則項，其他部分都和WGAN一樣。
    - 這個正則項就是WGAN-GP中GP（gradient penalty），即梯度約束。
    - 這個約束的意思是：critic相對於原始輸入的梯度的L2範數要約束在 
  - 範例1 Fashion MNIST
  - 範例2
- Diffussion Model(2015)
  - 2015 
  - 2019
  - 2020 | DDPM|denoising diffusion probabilistic model (很慢)
  - 2021 | OpenAI | Denoising Diffusion Implicit Models (DDIM)
  - 2021 | LDM | latent diffusion models
  - 2022 | Stable Diffusion  ==> KerasCV
  - 推薦課程 ==> https://github.com/huggingface/diffusion-models-class
    - 中文說明 ==> [擴散模型從原理到實戰](https://www.tenlong.com.tw/products/9787115618870?list_name=srh) 
## GenAI 
## 程式範例
- Keras
  - VAE
    - [Variational AutoEncoder](https://keras.io/examples/generative/vae/)
  - GAN
    - [Conditional GAN](https://keras.io/examples/generative/conditional_gan/)
    - [Face image generation with StyleGAN](https://keras.io/examples/generative/stylegan/)
  - Diffusion Model
    - [Denoising Diffusion Probabilistic Model](https://keras.io/examples/generative/ddpm/)
    - [Denoising Diffusion Implicit Models](https://keras.io/examples/generative/ddim/)
    - [A walk through latent space with Stable Diffusion 3](https://keras.io/examples/generative/random_walks_with_stable_diffusion_3/)
    - [Fine-tuning Stable Diffusion](https://keras.io/examples/generative/finetune_stable_diffusion/)
- Tensorflow
  - [TensorFlow 教學課程](https://www.tensorflow.org/tutorials?hl=zh-tw)
    - [Intro to Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder)
    - [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae)
    - [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
    - [High-performance image generation using Stable Diffusion in KerasCV](https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion)
  - [API Documentation](https://www.tensorflow.org/api_docs)
- PyTorch
  - [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
    - [DCGAN Tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    - [Optimizing Vision Transformer Model for Deployment](https://docs.pytorch.org/tutorials/beginner/vt_tutorial.html) ==> Facebook Data-efficient Image Transformers
  - [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html)

## Awesome-VAE
- https://github.com/matthewvowels1/Awesome-VAEs
- Vector Quantized Variational Autoencoder (VQ-VAE)
  - https://github.com/rese1f/awesome-VQVAE 

## Awesome diffusion
- https://github.com/Zeqiang-Lai/awesome-diffusion
- https://github.com/diff-usion/Awesome-Diffusion-Models

## Awesome GAN
- https://github.com/nightrome/really-awesome-gan
- https://github.com/nashory/gans-awesome-applications
- https://github.com/Faldict/awesome-GAN
- https://github.com/dongb5/GAN-Timeline
- https://github.com/hindupuravinash/the-gan-zoo![image](https://github.com/user-attachments/assets/a9469ba2-a85a-4e12-8d16-be1bccb8fc3b)

## 延伸閱讀
- [Generative Deep Learning, 2nd Edition](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174/)
  - `3`. Variational Autoencoders
  - `4`. Generative Adversarial Networks
  - `5`. Autoregressive Models
    - 自我迴歸模型（英語：Autoregressive model，簡稱AR模型），是統計上一種處理時間序列的方法，用同一變數例如x的之前期資料來預測本期xt，並假設它們為一線性關係。
    - 因為這是從迴歸分析中的線性迴歸發展而來，只是不用x預測y,而是用x預測x（自己）；因此叫做自我迴歸。
    - 動迴歸模型是機器學習 (ML) 模型的一種類型，其透過測量先前依序的輸入來自動預測序列中的下一個組成。
    - 自動迴歸是時間序列分析中使用的統計技術，其假設時間序列的目前值是過去值的其中一個函數。自動迴歸模型使用類似的數學技術來確定序列中元素之間的概率相關性。其接著使用衍生的知識來猜測未知序列中的下一個元素
    - LSTM for time series
    - Stacked Recurrent Networks
    - Gated Recurrent Units
    - Bidirectional Cells
    - PixelCNN 
  - `6`. Normalizing Flow Models
  - `7`. Energy-Based Models 基於能量的模型
    - 事件的概率可以使用玻爾茲曼分佈來表示
    - 玻爾茲曼分佈是將實值能量函數標準化為 0 到 1 之間的特定函數
    - 這個分佈最初是由路德維希·玻爾茲曼 （Ludwig Boltzmann） 於 1868 年制定的，他用它來描述處於熱平衡狀態的氣體。
    - [Tutorial 8: Deep Energy-Based Generative Models](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)
  - `8`. Diffusion Models 


## old
- [GenAI_SECURITY202410](https://github.com/8wingflying/GenAI_SECURITY202410/) ==>LLM
- https://github.com/8wingflying/GenAI20240518
