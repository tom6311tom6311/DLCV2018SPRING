from keras.layers import Input, Conv2D, Conv2DTranspose, AveragePooling2D, Dense, Flatten, BatchNormalization, LeakyReLU, Reshape, Activation, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import backend as K
import numpy as np
import sys
import util
import math

def make_trainable(net, val):
  net.trainable = val
  for l in net.layers:
    l.trainable = val

class DCGAN:
  def __init__(self, w_in, discrim_dims, gen_dims, latent_dim):
    self.latent_dim = latent_dim
    input_img = Input(shape=(w_in,w_in,3), name='dcgan_discrim_in')
    discrim = input_img

    for i, (filters, kernel_size, strides) in enumerate(discrim_dims):
      discrim = Conv2D(filters, kernel_size, strides=strides, padding='same', name='dcgan_discrim_conv_'+str(i))(discrim)
      # discrim = BatchNormalization(momentum=0.8)(discrim)
      discrim = LeakyReLU(0.2)(discrim)
      discrim = Dropout(0.3)(discrim)

    # last_conv_dim = K.int_shape(discrim)
    # discrim = Conv2D(1, 2, strides=1, padding='same', name='dcgan_discrim_conv_last')(discrim)
    # discrim = AveragePooling2D(last_conv_dim[1], name='dcgan_discrim_avg_pool')(discrim)
    discrim = Flatten()(discrim)
    discrim = Dense(256)(discrim)
    discrim = LeakyReLU(0.2)(discrim)
    discrim = Dropout(0.3)(discrim)
    discrim = Dense(1)(discrim)
    discrim = Activation('sigmoid')(discrim)

    gen_input = Input(shape=(self.latent_dim,), name='dcgan_gen_in')
    gen = gen_input

    # gen = Dense(last_conv_dim[1] * last_conv_dim[2] * last_conv_dim[3], kernel_initializer='glorot_normal')(gen)
    w_gen_first = int(w_in / math.pow(2,len(gen_dims)))
    gen = Dense(w_gen_first * w_gen_first * 128, kernel_initializer='glorot_normal')(gen)
    gen = BatchNormalization(momentum=0.8)(gen)
    gen = Activation('relu')(gen)
    gen = Reshape((w_gen_first, w_gen_first, 128))(gen)
    # for j, (filters, kernel_size, strides) in reversed(list(enumerate(discrim_dims))):
    for j, (filters, kernel_size, strides) in enumerate(gen_dims):
      gen = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', name='dcgan_gen_conv_'+str(j))(gen)
      gen = BatchNormalization()(gen)
      gen = Activation('relu')(gen)
      # gen = Dropout(0.2)(gen)

    gen = Conv2D(3, 1, strides=1, padding='same', activation='tanh')(gen)

    self.generator = Model(inputs=gen_input, outputs=gen)
    self.discriminator = Model(inputs=input_img, outputs=discrim)
    self.discriminator.compile(optimizer=SGD(0.002), loss='binary_crossentropy' ,metrics=['acc'])

    make_trainable(self.discriminator, False)

    gan_input = Input(shape=(self.latent_dim,))
    gan = self.generator(gan_input)
    gan = self.discriminator(gan)
    self.gan = Model(inputs=gan_input, outputs=gan)
    self.gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy',  metrics=['acc'])

    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # d_o_g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    print('\ndiscriminator: ')
    self.discriminator.summary()
    print('\ngenerator: ')
    self.generator.summary()

  def summary(self):
    self.gan.summary()

  def train(self, x_train, epochs, batch_size, chk_point_interval=5, out_model_prefix='../out/gan_', out_img_dir='../img_gan/', update_ratio=2):
    print("Number of batches", int(x_train.shape[0]/batch_size))
    for epoch in range(epochs):
      print("\nEpoch #" + str(epoch))
      d_loss = 0
      g_loss = 0
      for index in range(int(x_train.shape[0]/batch_size)):
        noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
        if (index % update_ratio == 0): #(index <= 2 or d_loss >= g_loss):
          image_batch = x_train[index*batch_size:(index+1)*batch_size]
          generated_images = self.generator.predict(noise)
          images = np.concatenate((image_batch, generated_images))
          y = [1] * batch_size + [0] * batch_size
          d_loss, d_acc = self.discriminator.train_on_batch(images, y)
        
        g_loss, g_acc = self.gan.train_on_batch(noise, [1] * batch_size)
        sys.stdout.write("batch %d d_loss : %f, g_loss : %f d_acc : %f, g_acc : %f\r" % (index, d_loss, g_loss, d_acc, g_acc))
        sys.stdout.flush()
        if epoch % chk_point_interval == chk_point_interval-1 and index % 100 == 0:
          util.save_image(generated_images[0], out_img_dir + str(epoch) + '_' + str(index) + '.png', isFlattened=False, val_range=(-1,1))
      if epoch % chk_point_interval == chk_point_interval-1:
        self.generator.save_weights(out_model_prefix + str(epoch) + '_gen', True)
        self.discriminator.save_weights(out_model_prefix + str(epoch) + '_discrim', True)
