### 資料來源
- WGAN ==> https://ithelp.ithome.com.tw/articles/10320794
- WGAN-ng  ==>  https://www.cnblogs.com/for-technology-lover/p/14854809.html
### WGAN
- https://ithelp.ithome.com.tw/articles/10319995
- https://ithelp.ithome.com.tw/articles/10320794
```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Activation, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import RMSprop #RMEprop優化器
import tensorflow.keras.backend as K #Keras後端
import matplotlib.pyplot as plt
import numpy as np
import os

class WGAN():
    def __init__(self, generator_lr, discriminator_lr, clip_value):
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.adversarial = self.build_adversarialmodel()
        self.clip_value = clip_value

        self.gloss = []
        self.dloss = []
        if not os.path.exists('./result/WGAN/imgs'):# 將訓練過程產生的圖片儲存起來
            os.makedirs('./result/WGAN/imgs')# 如果忘記新增資料夾可以用這個方式建立

    def load_data(self):
        (x_train, _), (_, _) = mnist.load_data()  # 底線是未被用到的資料，可忽略
        x_train = (x_train / 127.5)-1  # 正規化
        x_train = x_train.reshape((-1, 28, 28, 1))
        return x_train

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        input_ = Input(shape=(100, ))
        x = Dense(7*7*32)(input_)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
        x = Reshape((7, 7, 32))(x)
        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
        out = Conv2DTranspose(1, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

        model = Model(inputs=input_, outputs=out, name='Generator')
        model.summary()
        return model

    def build_discriminator(self):
        input_ = Input(shape = (28, 28, 1))
        x = Conv2D(256, kernel_size=4, strides=2, padding='same')(input_)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, kernel_size=4, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        out = Dense(1)(x)

        model = Model(inputs=input_, outputs=out, name='Discriminator')
        dis_optimizer = RMSprop(learning_rate=self.discriminator_lr)
        model.compile(loss=self.wasserstein_loss,
                      optimizer=dis_optimizer,
                      metrics=['accuracy'])
        model.summary()
        return model

    def build_adversarialmodel(self):
        noise_input = Input(shape=(100, ))
        generator_sample = self.generator(noise_input)
        self.discriminator.trainable = False
        out = self.discriminator(generator_sample)
        model = Model(inputs=noise_input, outputs=out)

        adv_optimizer = RMSprop(learning_rate=self.generator_lr)
        model.compile(loss=self.wasserstein_loss, optimizer=adv_optimizer)
        model.summary()
        return model

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 準備訓練資料
        x_train = self.load_data()
        # 準備訓練的標籤，分為真實標籤與假標籤，需注意標籤的內容！
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        for epoch in range(epochs):
            # 隨機取一批次的資料用來訓練
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            # 從常態分佈中採樣一段雜訊
            noise = np.random.normal(0, 1, (batch_size, 100))
            # 生成一批假圖片
            gen_imgs = self.generator.predict(noise)
            # 判別器訓練判斷真假圖片
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
			# 權重裁剪
            for l in self.discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                l.set_weights(weights)
            #儲存鑑別器損失變化 索引值0為損失 索引值1為準確率
            self.dloss.append(d_loss[0])
            # 訓練生成器的生成能力
            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.adversarial.train_on_batch(noise, valid)
            # 儲存生成器損失變化
            self.gloss.append(g_loss)
            # 將這一步的訓練資訊print出來
            print(f"Epoch:{epoch} [D loss: {d_loss[0]}, acc: {100 * d_loss[1]:.2f}] [G loss: {g_loss}]")
            # 在指定的訓練次數中，隨機生成圖片，將訓練過程的圖片儲存起來
            if epoch % sample_interval == 0:
                self.sample(epoch)
        self.save_data()

    def save_data(self):
        np.save(file='./result/WGAN/generator_loss.npy',arr=np.array(self.gloss))
        np.save(file='./result/WGAN/discriminator_loss.npy', arr=np.array(self.dloss))
        save_model(model=self.generator,filepath='./result/WGAN/Generator.h5')
        save_model(model=self.discriminator,filepath='./result/WGAN/Discriminator.h5')
        save_model(model=self.adversarial,filepath='./result/WGAN/Adversarial.h5')

    def sample(self, epoch=None, num_images=25, save=True):
        r = int(np.sqrt(num_images))
        noise = np.random.normal(0, 1, (num_images, 100))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = (gen_imgs+1)/2
        fig, axs = plt.subplots(r, r)
        count = 0
        for i in range(r):
            for j in range(r):
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        if save:
            fig.savefig(f"./result/WGAN/imgs/{epoch}epochs.png")
        else:
            plt.show()
        plt.close()

if __name__ == '__main__':
    gan = WGAN(generator_lr=0.00005,discriminator_lr=0.00005, clip_value=0.01)
    gan.train(epochs=10000, batch_size=128, sample_interval=200)
    gan.sample(save=False)
```

### WGAN-ng 
- https://www.cnblogs.com/for-technology-lover/p/14854809.html
```python

```
