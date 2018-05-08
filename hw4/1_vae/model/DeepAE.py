from model.AEBase import AEBase
from keras.layers import Input, Dense
from keras.models import Model

class DeepAE(AEBase):
  def __init__(self, dim_in, encoding_dims):
    input_img = Input(shape=(dim_in,), name='deep_ae_in')

    encoded = Dense(encoding_dims[0], activation='relu', name='deep_ae_enc_0')(input_img)
    for i, dim in enumerate(encoding_dims[1:]):
      encoded = Dense(dim, activation='relu', name='deep_ae_enc_'+str(i+1))(encoded)

    decoder_input = Input(shape=(encoding_dims[-1],), name='deep_ae_decoder_in')

    decoded = encoded
    decoder = decoder_input
    for j, dim in reversed(list(enumerate(encoding_dims[:-1]))):
      layer = Dense(dim, activation='relu', name='deep_ae_dec_'+str(j))
      decoded = layer(decoded)
      decoder = layer(decoder)

    layer = Dense(dim_in, activation='sigmoid', name='deep_ae_decoded')
    decoded = layer(decoded)
    decoder = layer(decoder)

    self.autoencoder = Model(input=input_img, output=decoded)
    self.encoder = Model(input=input_img, output=encoded)
    self.decoder = Model(input=decoder_input, output=decoder)

    self.autoencoder.compile(optimizer='adadelta', loss='mse')