from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.layers import Flatten
import numpy as np

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
output = model.get_layer('avg_pool').output
output = Flatten()(output)
model = Model(model.input, output)

model.summary()

img_path = 'data/FullLengthVideos/videos/train/OP01-R01-PastaSalad/00001.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img).reshape((1,224,224,3))
print(x.shape)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

preds = model.predict(x)
print(preds)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]