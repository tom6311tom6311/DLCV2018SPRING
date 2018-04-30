from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Activation
from keras.models import Model

def build_model():
  img_input = Input(shape=(512, 512, 3))
  x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
  x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
  x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)
  x = Conv2D(4096, (7,7), activation='relu', padding='same', name='fcn_conv1')(x)
  x = Dropout(0.5)(x)
  x = Conv2D(4096, (1,1), activation='relu', padding='same', name='fcn_conv2')(x)
  x = Dropout(0.5)(x)
  x = Conv2D(8, (1,1), kernel_initializer='he_normal', name='fcn_conv3')(x)
  x = Conv2DTranspose(8, kernel_size=(64,64), strides=(32,32) , padding='same', use_bias=False, name='fcn_conv_t1')(x)
  x = Activation('softmax')(x)

  model = Model(img_input, x)
  return model