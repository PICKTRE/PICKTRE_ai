import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
#.h5파일을 json으로 변환
model = tf.keras.models.load_model('model/ResNet50V2_fine_tuned.h5')
tfjs.converters.save_keras_model(model, 'model/ResNet50V2_fine_tuned.json')
