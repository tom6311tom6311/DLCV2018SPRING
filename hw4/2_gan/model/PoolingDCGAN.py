from keras.layers import Input, Conv2D, Conv2DTranspose, AveragePooling2D, UpSampling2D, Dense, Flatten, BatchNormalization, LeakyReLU, Reshape, Activation, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, sgd
import numpy as np
import sys
import util

class PoolingDCGAN:
  def __init__(self, w_in, discrim_dims, latent_dim):
    self.latent_dim = latent_dim
    input_img = Input(shape=(w_in,w_in,3), name='dcgan_discrim_in')
    discrim = input_img

    for i, (filters, kernel_size, strides) in enumerate(discrim_dims):
      discrim = Conv2D(filters, kernel_size, padding='same', name='dcgan_discrim_conv_'+str(i))(discrim)
      discrim = AveragePooling2D(strides, name='dcgan_discrim_avg_pool'+str(i))(discrim)
      # discrim = BatchNormalization()(discrim)
      discrim = LeakyReLU(0.2)(discrim)

    last_conv_dim = K.int_shape(discrim)
    discrim = Conv2D(1, 5, strides=1, padding='same', name='dcgan_discrim_conv_last')(discrim)
    discrim = AveragePooling2D(last_conv_dim[1], name='dcgan_discrim_avg_pool_last')(discrim)
    discrim = Flatten()(discrim)
    discrim = Activation('sigmoid')(discrim)

    gen_input = Input(shape=(self.latent_dim,), name='dcgan_gen_in')
    gen = gen_input

    gen = Dense(last_conv_dim[1] * last_conv_dim[2] * last_conv_dim[3])(gen)
    gen = BatchNormalization()(gen)
    gen = Reshape((last_conv_dim[1], last_conv_dim[2], last_conv_dim[3]))(gen)
    for j, (filters, kernel_size, strides) in reversed(list(enumerate(discrim_dims))):
      gen = Conv2DTranspose(filters, kernel_size, padding='same', name='dcgan_gen_conv_'+str(j))(gen)
      gen = BatchNormalization()(gen)
      gen = LeakyReLU(0.2)(gen)
      gen = Dropout(0.3)(gen)
      gen = UpSampling2D(strides)(gen)

    gen = Conv2D(3, 5, strides=1, padding='same', activation='tanh')(gen)

    self.discriminator = Model(inputs=input_img, outputs=discrim)
    self.generator = Model(inputs=gen_input, outputs=gen)
    d_o_g = self.discriminator(self.generator(gen_input))
    self.dis_of_gen = Model(inputs=gen_input, outputs=d_o_g)

    discrim_optim = sgd(lr=0.01, momentum=0.9, nesterov=True)
    generator_optim = Adam(lr=1e-3)
    dis_of_gen_optim = Adam(lr=1e-3)

    self.discriminator.compile(optimizer=discrim_optim, loss='binary_crossentropy')
    self.generator.compile(optimizer=generator_optim, loss='binary_crossentropy')
    self.dis_of_gen.compile(optimizer=dis_of_gen_optim, loss='binary_crossentropy')

    print('\ndiscriminator: ')
    self.discriminator.summary()
    print('\ngenerator: ')
    self.generator.summary()

  def summary(self):
    self.dis_of_gen.summary()

  def train(self, x_train, epochs, batch_size, chk_point_interval=5, out_model_prefix='../out/pool_gan_', out_img_dir='../img_pool_gan/'):
    print("Number of batches", int(x_train.shape[0]/batch_size))
    for epoch in range(epochs):
      print("\nEpoch #" + str(epoch))
      d_loss = 0
      g_loss = 0
      for index in range(int(x_train.shape[0]/batch_size)):
        if (d_loss != 0):
          can_lock_d = True
        if (index <= 2 or d_loss >= g_loss):
          noise = np.random.uniform(-1, 1, size=(batch_size, self.latent_dim))
          image_batch = x_train[index*batch_size:(index+1)*batch_size]
          generated_images = self.generator.predict(noise, verbose=0)
          images = np.concatenate((image_batch, generated_images))
          y = [1] * batch_size + [0] * batch_size
          d_loss = self.discriminator.train_on_batch(images, y)
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        self.discriminator.trainable = False
        g_loss = self.dis_of_gen.train_on_batch(noise, [1] * batch_size)
        self.discriminator.trainable = True
        sys.stdout.write("batch %d d_loss : %f, g_loss : %f\r" % (index, d_loss, g_loss))
        sys.stdout.flush()
      if epoch % chk_point_interval == chk_point_interval-1:
        util.save_image(generated_images[0], out_img_dir + str(epoch) + '.png', isFlattened=False, val_range=(-1,1))
        self.generator.save_weights(out_model_prefix + str(epoch) + '_gen', True)
        self.discriminator.save_weights(out_model_prefix + str(epoch) + '_discrim', True)
