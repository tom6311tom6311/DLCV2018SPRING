from model.AEBase import AEBase
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras.losses import mse
from keras import backend as K

class VariationalAE(AEBase):
  def __init__(self, w_in, encoding_conv_dims, latent_dim, kl_lambda=1e-5):
    input_img = Input(shape=(w_in,w_in,3), name='vae_in')

    encoded = input_img

    for i, (filters, kernel_size) in enumerate(encoding_conv_dims):
      encoded = Conv2D(filters, kernel_size, padding='same', activation='relu', name='vae_enc_conv_'+str(i))(encoded)
      encoded = MaxPooling2D((2, 2), padding='same', name= 'vae_enc_pool_'+str(i))(encoded)

    enc_last_shape = K.int_shape(encoded)

    encoded = Flatten()(encoded)
    z_mean = Dense(latent_dim, name='vae_z_mean')(encoded)
    z_log_var = Dense(latent_dim, name='vae_z_log_var')(encoded)
    z = Lambda(self.sampling, output_shape=(latent_dim,), name='vae_z')([z_mean, z_log_var])

    decoder_input = Input(shape=(latent_dim,), name='vae_decoder_in')
    decoder = decoder_input


    decoder = Dense(enc_last_shape[1] * enc_last_shape[2] * enc_last_shape[3], activation='relu')(decoder)
    decoder = Reshape((enc_last_shape[1], enc_last_shape[2], enc_last_shape[3]))(decoder)
    for j, (filters, kernel_size) in reversed(list(enumerate(encoding_conv_dims))):
      decoder = Conv2D(filters, kernel_size, padding='same', activation='relu', name='vae_dec_conv_'+str(j))(decoder)
      decoder = UpSampling2D((2, 2), name='vae_dec_upsample_'+str(j))(decoder)

    decoder = Conv2D(3, encoding_conv_dims[0][1], padding='same', activation='sigmoid', name='vae_decoded')(decoder)

    self.encoder = Model(input=input_img, output=[z_mean, z_log_var, z])
    self.decoder = Model(input=decoder_input, output=decoder)

    vae_out = self.decoder(self.encoder(input_img)[2])
    self.autoencoder = Model(input=input_img, output=vae_out)

    reconstruction_loss = mse(K.flatten(input_img), K.flatten(vae_out))
    # reconstruction_loss *= w_in * w_in * 3
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss +  kl_lambda * kl_loss)
    self.autoencoder.add_loss(vae_loss)
    self.autoencoder.compile(optimizer='adadelta', loss='mse')

  def sampling(self, args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon