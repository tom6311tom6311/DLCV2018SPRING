from model.AEBase import AEBase
from keras.layers import Input, Dense
from keras.models import Model

class SimpleAE(AEBase):
	def __init__(self, dim_in, encoding_dim):
		input_img = Input(shape=(dim_in,), name='simple_ae_in')

		encoded = Dense(encoding_dim, activation='relu', name='simple_ae_encoded')(input_img)		

		decoded = Dense(dim_in, activation='sigmoid', name='simple_ae_decoded')(encoded)

		self.autoencoder = Model(input=input_img, output=decoded)

		self.encoder = Model(input=input_img, output=encoded)

		encoded_input = Input(shape=(encoding_dim,), name='simple_ae_decoder_in')
		decoder_layer = self.autoencoder.layers[-1]
		self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')