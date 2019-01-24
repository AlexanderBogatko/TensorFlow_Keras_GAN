from keras.layers import Dense, Input, Lambda, Flatten, concatenate, Reshape, RepeatVector
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import pickle as pkl
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from scipy.stats import norm
import tensorflow as tf

class CGAN:

  def __init__(self, digit_size, num_classes, latent_dim, sess_name=''):

    self.sess = tf.Session()
    self.digit_size = digit_size
    self.latent_dim = latent_dim

    self.a = tf.placeholder(tf.float32, shape=(None, self.digit_size, self.digit_size, 1))
    self.b = tf.placeholder(tf.float32, shape=(None, num_classes))
    self.c = tf.placeholder(tf.float32, shape=(None, latent_dim))

    self.img = Input(tensor=self.a)
    self.lbls = Input(tensor=self.b)
    self.z = Input(tensor=self.c)

    ############## Build Discriminator #################

    with tf.variable_scope('discriminator'):

      x = Conv2D(128, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(self.img)
      x = self.add_units_to_conv2d(x, self.lbls)

      x = MaxPooling2D((2, 2), padding='same')(x)

      x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
      x = MaxPooling2D((2, 2), padding='same')(x)

      x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x)

      h = Flatten()(x)
      disc_output = Dense(1, activation='sigmoid')(h)

    self.discriminator = Model([self.img, self.lbls], disc_output)

    ############# Build Generator #####################

    with tf.variable_scope('generator'):

        x = concatenate([self.z, self.lbls])

        x = Dense(7*7*128, activation='relu')(x)
        x = Reshape((7, 7, 128))(x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(x)

        x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)

        x = UpSampling2D(size=(2, 2))(x)

        gen_output = Conv2D(1, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)

    self.generator = Model([self.z, self.lbls], gen_output)

    ################ GAN ##################

    self.generated_img = self.generator([self.z, self.lbls])

    self.discr_real_img = self.discriminator([self.img, self.lbls])

    self.discr_fake_img = self.discriminator([self.generated_img, self.lbls])

    #self.cgan = Model([self.img, self.lbls], discr_fake_img)

    ############## Define Losses #################

    log_d_img   = tf.reduce_mean(-tf.log(self.discr_real_img + 1e-10))
    log_d_gen_img = tf.reduce_mean(-tf.log(1. - self.discr_fake_img + 1e-10))

    self.GenLoss = -log_d_gen_img
    self.DiscrLoss = 0.5 * (log_d_gen_img + log_d_img)

    ############## Optimizers #############
    GenOpt = tf.train.RMSPropOptimizer(0.0003)
    DiscrOpt = tf.train.RMSPropOptimizer(0.0001)

    Gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    Discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

    self.gen_step = GenOpt.minimize(self.GenLoss, var_list=Gen_vars)
    self.discr_step = DiscrOpt.minimize(self.DiscrLoss, var_list=Discr_vars)

    self.saver = tf.train.Saver()

    self.sess.run(tf.global_variables_initializer())

    if(sess_name):
      self.saver.restore(self.sess, './' + sess_name)

  def add_units_to_conv2d(self, conv2, units):
    dim1 = int(conv2.shape[1])
    dim2 = int(conv2.shape[2])
    dimc = int(units.shape[1])
    repeat_n = dim1*dim2
    units_repeat = RepeatVector(repeat_n)(self.lbls)
    units_repeat = Reshape((dim1, dim2, dimc))(units_repeat)
    return concatenate([conv2, units_repeat])

  def gen_train_step(self, data_batch, lbls_batch, z):
    loss, _ = self.sess.run([self.DiscrLoss, self.gen_step], 
                                feed_dict={self.img: data_batch,
                                          self.lbls: lbls_batch,
                                          self.z: z,
                                          K.learning_phase(): 1})
    return loss

  def discr_train_step(self, data_batch, lbls_batch, z):
    loss, _ = self.sess.run([self.DiscrLoss, self.discr_step], 
                            feed_dict={self.img: data_batch,
                                      self.lbls: lbls_batch,
                                      self.z: z,
                                      K.learning_phase(): 1})
    return loss

  def train(self, Images, Labels, batch_size, epochs, k_steps):

    saving_period = 50 # frequency of saving model

    for i in range(epochs):
        # Select a random batch of images
        idx = np.random.randint(0, Images.shape[0], batch_size)
        imgs = Images[idx]
        lbls = Labels[idx]
        zp = np.random.randn(batch_size, latent_dim)

        ########## Train Discriminator ##########
        discr_loss = 0
        counter = 1
        for j in range(k_steps):
          loss = self.discr_train_step(imgs, lbls, zp)
          # next minibatch
          idx = np.random.randint(0, Images.shape[0], batch_size)
          imgs = Images[idx]
          lbls = Labels[idx]
          zp = np.random.randn(batch_size, latent_dim)

          discr_loss += loss

          if loss < 1.0:
            break

          counter += 1


        discr_loss /= counter

        ########## Train Generator ##########
        gen_loss = 0
        counter = 1
        for j in range(k_steps):
          loss = self.gen_train_step(imgs, lbls, zp)
          gen_loss += loss
          if loss > 0.4:
            break
          # next minibatch
          idx = np.random.randint(0, Images.shape[0], batch_size)
          imgs = Images[idx]
          lbls = Labels[idx]
          zp = np.random.randn(batch_size, latent_dim)
          counter += 1

        gen_loss /= counter

        print ("%d [D loss: %f] [G loss: %f]" % (i, discr_loss, gen_loss))
        
        if(not i % saving_period):
          self.saver.save(self.sess, './checkpoints/my-model')

  def generate(self, z, lbl):
    return self.sess.run(self.generator([self.z, self.lbls]), 
                        feed_dict={self.z: z,
                                  self.lbls: lbl,
                                  K.learning_phase(): 0})

  def draw_manifold(self, lbl):
        n = 15
        # Draw samples from manifold
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        figure = np.zeros((self.digit_size * n, self.digit_size * n))
        input_lbl = np.zeros((1, 10))
        input_lbl[0, lbl] = 1
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.zeros((1, self.latent_dim))
                z_sample[:, :2] = np.array([[xi, yi]])

                x_decoded = self.sess.run(self.generator([self.z, self.lbls]), 
                                                          feed_dict={self.z: z_sample,
                                                                    self.lbls: input_lbl,
                                                                    K.learning_phase(): 0})
                digit = x_decoded[0].squeeze()
                figure[i * self.digit_size: (i + 1) * self.digit_size,
                       j * self.digit_size: (j + 1) * self.digit_size] = digit

        # Visualization
        plt.figure(figsize=(10, 10), num='Manifold')
        plt.imshow(figure, cmap='Greys_r')
        plt.grid(False)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
        return figure


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))
y_train_cat = to_categorical(y_train).astype(np.float32)
y_test_cat  = to_categorical(y_test).astype(np.float32)


sess_name = 'checkpoints/my-model'

# network parameters 
batch_size = 128
epochs = 2000
latent_dim = 10
digit_size = 28
num_classes = 10

cgan = CGAN(digit_size, num_classes, latent_dim, sess_name=sess_name)

cgan.train(x_train, y_train_cat, batch_size, epochs, 5)

cgan.draw_manifold(0)
cgan.draw_manifold(1)
cgan.draw_manifold(2)
cgan.draw_manifold(3)
cgan.draw_manifold(4)

