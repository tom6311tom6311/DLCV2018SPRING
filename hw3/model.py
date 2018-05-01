from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Activation, Add, Cropping2D, BatchNormalization
from keras.models import Model

# crop o1 wrt o2
def crop(o1, o2, i):
  o_shape2 = Model(i, o2).output_shape
  outputHeight2 = o_shape2[2]
  outputWidth2 = o_shape2[3]

  o_shape1 = Model(i, o1).output_shape
  outputHeight1 = o_shape1[2]
  outputWidth1 = o_shape1[3]

  cx = abs(outputWidth1 - outputWidth2)
  cy = abs(outputHeight2 - outputHeight1)

  if outputWidth1 > outputWidth2:
    o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
  else:
    o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

  if outputHeight1 > outputHeight2 :
    o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
  else:
    o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

  return o1, o2

def fcn32():
  img_input = Input(shape=(512, 512, 3))
  x = Conv2D(64, (3,3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3,3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
  x = Conv2D(128, (3,3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, (3,3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
  x = Conv2D(256, (3,3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3,3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3,3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)
  x = Conv2D(512, (3,3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
  x = Conv2D(512, (3,3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)
  x = Conv2D(4096, (7,7), padding='same', name='fcn_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  x = Conv2D(4096, (1,1), padding='same', name='fcn_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  x = Conv2D(8, (1,1), kernel_initializer='he_normal', name='fcn_conv3')(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(8, kernel_size=(64,64), strides=(32,32) , padding='same', use_bias=False, name='fcn_conv_t1')(x)
  x = Activation('softmax')(x)

  model = Model(img_input, x)
  return model

def fcn8():
  img_input = Input(shape=(512, 512, 3))
  x = Conv2D(64, (3,3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3,3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
  x = Conv2D(128, (3,3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, (3,3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
  x = Conv2D(256, (3,3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3,3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3,3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)
  f3 = x
  x = Conv2D(512, (3,3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
  f4 = x
  x = Conv2D(512, (3,3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3,3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)

  o = x
  o = Conv2D(4096, (7,7), padding='same', name='fcn_conv1')(o)
  o = BatchNormalization()(o)
  o = Activation('relu')(o)
  o = Dropout(0.5)(o)
  o = Conv2D(4096, (1,1), padding='same', name='fcn_conv2')(o)
  o = BatchNormalization()(o)
  o = Activation('relu')(o)
  o = Dropout(0.5)(o)
  o = Conv2D(8, (1,1), kernel_initializer='he_normal', name='fcn_conv3')(o)
  o = BatchNormalization()(o)
  o = Conv2DTranspose(8, kernel_size=(4,4), strides=(2,2), padding='same', use_bias=False, name='fcn_conv_t1')(o)

  o2 = f4
  o2 = Conv2D(8, (1,1), kernel_initializer='he_normal', name='fcn_conv4')(o2)
  o2 = BatchNormalization()(o2)

  o, o2 = crop(o, o2, img_input)
  o = Add()([o, o2])
  o = Activation('relu')(o)

  o = Conv2DTranspose(8, kernel_size=(4,4), strides=(2,2), padding='same', use_bias=False, name='fcn_conv_t2')(o)
  o2 = f3
  o2 = (Conv2D(8, (1,1), kernel_initializer='he_normal', name='fcn_conv5'))(o2)
  o2 = BatchNormalization()(o2)
  o2, o = crop(o2, o, img_input)
  o  = Add()([o2, o])
  o = Activation('relu')(o)
  o = Conv2DTranspose(8, kernel_size=(16,16), strides=(8,8), padding='same', use_bias=False, name='fcn_conv_t3')(o)

  o = Activation('softmax')(o)
  model = Model(img_input, o)
  return model

def build_model(use_baseline_model):
  if use_baseline_model == 'True':
    return fcn32()
  else:
    return fcn8()
