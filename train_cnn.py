from cnn import define_model
from cnn import train_coefficients_array
from load_data import get_train_labels
from tensorflow.keras.utils import to_categorical
import numpy as np


model = define_model(train_coefficients_array(0, 1)[0])
class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
train_images = np.load('data/train_images.npy')
train_images = np.expand_dims(train_images, axis=-1)
train_labels = get_train_labels()
train_labels = to_categorical(train_labels, num_classes=5)
history = model.fit(train_images, train_labels, epochs=10, class_weight=class_weight)
model.save('data/cnn_model_weight2.h5')
