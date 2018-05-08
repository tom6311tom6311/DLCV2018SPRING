from model.AEBase import AEBase
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

class ConvAE(AEBase):
  def __init__(self, w_in, encoding_dims):
    input_img = Input(shape=(w_in,w_in,3), name='conv_ae_in')

    encoded = input_img

    for i, (filters, kernel_size) in enumerate(encoding_dims):
      encoded = Conv2D(filters, kernel_size, padding='same', activation='relu', name='conv_ae_enc_conv_'+str(i))(encoded)
      encoded = MaxPooling2D((2, 2), padding='same', name= 'conv_ae_enc_pool_'+str(i))(encoded)

    w_encoded = w_in / (2 ** len(encoding_dims))
    decoder_input = Input(shape=(w_encoded, w_encoded, encoding_dims[-1][0]), name='conv_ae_decoder_in')

    decoded = encoded
    decoder = decoder_input
    for j, (filters, kernel_size) in reversed(list(enumerate(encoding_dims))):
      conv_layer = Conv2D(filters, kernel_size, padding='same', activation='relu', name='conv_ae_dec_conv_'+str(j))
      decoded = conv_layer(decoded)
      decoder = conv_layer(decoder)

      upsample_layer = UpSampling2D((2, 2), name='conv_ae_dec_upsample_'+str(j))
      decoded = upsample_layer(decoded)
      decoder = upsample_layer(decoder)

    conv_layer = Conv2D(3, encoding_dims[0][1], padding='same', activation='sigmoid', name='conv_ae_decoded')
    decoded = conv_layer(decoded)
    decoder = conv_layer(decoder)

    self.autoencoder = Model(input=input_img, output=decoded)
    self.encoder = Model(input=input_img, output=encoded)
    self.decoder = Model(input=decoder_input, output=decoder)

    self.autoencoder.compile(optimizer='adadelta', loss='mse')