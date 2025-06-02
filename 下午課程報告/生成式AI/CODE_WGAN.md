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
# wgan_and_wgan_gp.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import binary_accuracy

import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
print("Tensorflow", tf.__version__)

ds_train, ds_info = tfds.load('fashion_mnist', split='train',shuffle_files=True,with_info=True)
fig = tfds.show_examples(ds_train, ds_info)

batch_size = 64
image_shape = (32, 32, 1)

def preprocess(features):
    image = tf.image.resize(features['image'], image_shape[:2])    
    image = tf.cast(image, tf.float32)
    image = (image-127.5)/127.5
    return image

ds_train = ds_train.map(preprocess)
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size, drop_remainder=True).repeat()

train_num = ds_info.splits['train'].num_examples
train_steps_per_epoch = round(train_num/batch_size)
print(train_steps_per_epoch)

"""
WGAN
"""
class WGAN():
    def __init__(self, input_shape):

        self.z_dim = 128
        self.input_shape = input_shape
        
        # losses
        self.loss_critic_real = {}
        self.loss_critic_fake = {}
        self.loss_critic = {}
        self.loss_generator = {}
        
        # critic
        self.n_critic = 5
        self.critic = self.build_critic()
        self.critic.trainable = False

        self.optimizer_critic = RMSprop(5e-5)

        # build generator pipeline with frozen critic
        self.generator = self.build_generator()
        critic_output = self.critic(self.generator.output)
        self.model = Model(self.generator.input, critic_output)
        self.model.compile(loss = self.wasserstein_loss,
                           optimizer =  RMSprop(5e-5))
        self.critic.trainable = True

        
    def wasserstein_loss(self, y_true, y_pred):

        w_loss = -tf.reduce_mean(y_true*y_pred)

        return w_loss

    def build_generator(self):

        DIM = 128
        model = tf.keras.Sequential(name='Generator') 

        model.add(layers.Input(shape=[self.z_dim])) 

        model.add(layers.Dense(4*4*4*DIM))
        model.add(layers.BatchNormalization()) 
        model.add(layers.ReLU())
        model.add(layers.Reshape((4,4,4*DIM))) 

        model.add(layers.UpSampling2D((2,2), interpolation="bilinear"))
        model.add(layers.Conv2D(2*DIM, 5, padding='same')) 
        model.add(layers.BatchNormalization()) 
        model.add(layers.ReLU())

        model.add(layers.UpSampling2D((2,2), interpolation="bilinear"))
        model.add(layers.Conv2D(DIM, 5, padding='same')) 
        model.add(layers.BatchNormalization()) 
        model.add(layers.ReLU())

        model.add(layers.UpSampling2D((2,2), interpolation="bilinear"))       
        model.add(layers.Conv2D(image_shape[-1], 5, padding='same', activation='tanh')) 

        return model             
    
    def build_critic(self):

        DIM = 128
        model = tf.keras.Sequential(name='critics') 

        model.add(layers.Input(shape=self.input_shape)) 

        model.add(layers.Conv2D(1*DIM, 5, strides=2, padding='same'))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2D(2*DIM, 5, strides=2, padding='same'))
        model.add(layers.BatchNormalization()) 
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2D(4*DIM, 5, strides=2, padding='same'))
        model.add(layers.BatchNormalization()) 
        model.add(layers.LeakyReLU(0.2))


        model.add(layers.Flatten()) 
        model.add(layers.Dense(1)) 

        return model     
    
 
    def train_critic(self, real_images, batch_size):

        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)
                  
        g_input = tf.random.normal((batch_size, self.z_dim))
        fake_images = self.generator.predict(g_input)
        
        with tf.GradientTape() as total_tape:
            
            # forward pass
            pred_fake = self.critic(fake_images)
            pred_real = self.critic(real_images)
            
            # calculate losses
            loss_fake = self.wasserstein_loss(fake_labels, pred_fake)
            loss_real = self.wasserstein_loss(real_labels, pred_real)           

            # total loss
            total_loss = loss_fake + loss_real
            
            # apply gradients
            gradients = total_tape.gradient(total_loss, self.critic.trainable_variables)
            
            self.optimizer_critic.apply_gradients(zip(gradients, self.critic.trainable_variables))

        for layer in self.critic.layers: 
            weights = layer.get_weights() 
            weights = [tf.clip_by_value(w, -0.01, 0.01) for w in weights]
            layer.set_weights(weights) 

        return loss_fake, loss_real
                                                
    def train(self, data_generator, batch_size, steps, interval=200):

        val_g_input = tf.random.normal((batch_size, self.z_dim))
        real_labels = tf.ones(batch_size)

        for i in range(steps):
            for _ in range(self.n_critic):
                real_images = next(data_generator)
                loss_fake, loss_real = self.train_critic(real_images, batch_size)
                critic_loss = loss_fake + loss_real
                
            # train generator
            g_input = tf.random.normal((batch_size, self.z_dim))
            g_loss = self.model.train_on_batch(g_input, real_labels)
            
            self.loss_critic_real[i] = loss_real.numpy()
            self.loss_critic_fake[i] = loss_fake.numpy()
            self.loss_critic[i] = critic_loss.numpy()
            self.loss_generator[i] = g_loss

            if i%interval == 0:
                msg = "Step {}: g_loss {:.4f} critic_loss {:.4f} critic fake {:.4f}  critic_real {:.4f}"\
                .format(i, g_loss, critic_loss, loss_fake, loss_real)
                print(msg)

                fake_images = self.generator.predict(val_g_input)
                self.plot_images(fake_images)
                self.plot_losses()

    def plot_images(self, images):   
        grid_row = 1
        grid_col = 8
        f, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col*2.5, grid_row*2.5))
        for row in range(grid_row):
            for col in range(grid_col):
                if self.input_shape[-1]==1:
                    axarr[col].imshow(images[col,:,:,0]*0.5+0.5, cmap='gray')
                else:
                    axarr[col].imshow(images[col]*0.5+0.5)
                axarr[col].axis('off') 
        plt.show()

    def plot_losses(self):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.set_figwidth(10)
        fig.set_figheight(6)
        ax1.plot(list(self.loss_critic.values()), label='Critic loss', alpha=0.7)
        ax1.set_title("Critic loss")
        ax2.plot(list(self.loss_generator.values()), label='Generator loss', alpha=0.7)
        ax2.set_title("Generator loss")

        plt.xlabel('Steps')
        plt.show()

wgan = WGAN(image_shape)
wgan.generator.summary()

wgan.critic.summary()

wgan.train(iter(ds_train), batch_size, 2000, 100)

z = tf.random.normal((8, 128))
generated_images = wgan.generator.predict(z)
wgan.plot_images(generated_images)

wgan.generator.save_weights('./wgan_models/wgan_fashion_minist.weights')

"""
WGAN_GP
"""
class WGAN_GP():
    def __init__(self, input_shape):

        self.z_dim = 128
        self.input_shape = input_shape

        # critic
        self.n_critic = 5
        self.penalty_const = 10
        self.critic = self.build_critic()
        self.critic.trainable = False

        self.optimizer_critic = Adam(1e-4, 0.5, 0.9)

        # build generator pipeline with frozen critic
        self.generator = self.build_generator()
        critic_output = self.critic(self.generator.output)
        self.model = Model(self.generator.input, critic_output)
        self.model.compile(loss=self.wasserstein_loss, optimizer=Adam(1e-4, 0.5, 0.9))

    def wasserstein_loss(self, y_true, y_pred):

        w_loss = -tf.reduce_mean(y_true*y_pred)

        return w_loss
    
    def build_generator(self):

        DIM = 128
        model = Sequential([
            layers.Input(shape=[self.z_dim]),
            
            layers.Dense(4*4*4*DIM),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((4,4,4*DIM)),

            layers.UpSampling2D((2,2), interpolation='bilinear'),
            layers.Conv2D(2*DIM, 5, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.UpSampling2D((2,2), interpolation='bilinear'),
            layers.Conv2D(2*DIM, 5, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.UpSampling2D((2,2), interpolation='bilinear'),
            layers.Conv2D(image_shape[-1], 5, padding='same', activation='tanh')
        ],name='Generator')

        return model
    
    def build_critic(self):

        DIM = 128
        model = Sequential([
            layers.Input(shape=self.input_shape),

            layers.Conv2D(1*DIM, 5, strides=2, padding='same', use_bias=False),
            layers.LeakyReLU(0.2),

            layers.Conv2D(2*DIM, 5, strides=2, padding='same', use_bias=False),
            layers.LeakyReLU(0.2),

            layers.Conv2D(4*DIM, 5, strides=2, padding='same', use_bias=False),
            layers.LeakyReLU(0.2),

            layers.Flatten(),
            layers.Dense(1)
        ], name='critics')

        return model
    
    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=np.arange(1, len(loss.shape)))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        loss = self.penalty_const * loss
        return loss
    
    def train_critic(self, real_images, batch_size):
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        g_input = tf.random.normal((batch_size, self.z_dim))
        fake_images = self.generator.predict(g_input)

        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
            # forward pass
            pred_fake = self.critic(fake_images)
            pred_real = self.critic(real_images)

            # calculate losses
            loss_fake = self.wasserstein_loss(fake_labels, pred_fake)
            loss_real = self.wasserstein_loss(real_labels, pred_real)

            # gradient penalty
            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1-epsilon) * fake_images
            gradient_tape.watch(interpolates)

            critic_interpolates = self.critic(interpolates)
            gradients_interpolates = gradient_tape.gradient(critic_interpolates, [interpolates])
            gradient_penalty = self.gradient_loss(gradients_interpolates)

            # total loss
            total_loss = loss_fake + loss_real + gradient_penalty

            # apply gradients
            gradients = total_tape.gradient(total_loss, self.critic.variables)

            self.optimizer_critic.apply_gradients(zip(gradients, self.critic.variables))
        return loss_fake, loss_real, gradient_penalty
    
    def train(self, data_generator, batch_size, steps, interval=100):
        val_g_input = tf.random.normal((batch_size, self.z_dim))
        real_labels = tf.ones(batch_size)

        for i in range(steps):
            for _ in range(self.n_critic):
                real_images = next(data_generator)
                loss_fake, loss_real, gradient_penalty = self.train_critic(real_images, batch_size)
                critic_loss = loss_fake + loss_real + gradient_penalty
            # train generator
            g_input = tf.random.normal((batch_size, self.z_dim))
            g_loss = self.model.train_on_batch(g_input, real_labels)
            if i%interval == 0:
                msg = "Step {}: g_loss {:.4f} critic_loss {:.4f} critic fake {:.4f}  critic_real {:.4f} penalty {:.4f}".format(i, g_loss, critic_loss, loss_fake, loss_real, gradient_penalty)
                print(msg)

                fake_images = self.generator.predict(val_g_input)
                self.plot_images(fake_images)

    def plot_images(self, images):   
        grid_row = 1
        grid_col = 8
        f, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col*2.5, grid_row*2.5))
        for row in range(grid_row):
            for col in range(grid_col):
                if self.input_shape[-1]==1:
                    axarr[col].imshow(images[col,:,:,0]*0.5+0.5, cmap='gray')
                else:
                    axarr[col].imshow(images[col]*0.5+0.5)
                axarr[col].axis('off') 
        plt.show()

wgan = WGAN_GP(image_shape)
wgan.train(iter(ds_train), batch_size, 5000, 100)

wgan.model.summary()

wgan.critic.summary()

z = tf.random.normal((8, 128))
generated_images = wgan.generator.predict(z)
wgan.plot_images(generated_images)

```
