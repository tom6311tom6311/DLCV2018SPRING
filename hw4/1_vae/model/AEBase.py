from keras import backend
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

class AEBase(object):
  def train(self, x_train, x_test, epochs, batch_size, log_dir='../tmp/ae', stop_early=True, save_check_points=True, out_model_prefix='../out/ae_'):
    callbacks = []
    if backend._BACKEND == 'tensorflow':
      callbacks.append(TensorBoard(log_dir=log_dir))

    if stop_early:
      callbacks.append(EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto'))

    if save_check_points:
      callbacks.append(ModelCheckpoint(out_model_prefix + '{epoch:02d}.hdf5', period=10, monitor='accuracy'))

    self.autoencoder.fit(x_train, x_train,
      nb_epoch=epochs,
      batch_size=batch_size,
      shuffle=True,
      validation_data=(x_test, x_test),
      callbacks=callbacks)

  def encode(self, x):
    return self.encoder.predict(x)

  def decode(self, x):
    return self.decoder.predict(x)
    
  def summary(self):
    self.autoencoder.summary()