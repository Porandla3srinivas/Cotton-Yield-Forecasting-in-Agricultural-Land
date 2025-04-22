import tensorflow as tf
from keras.applications import EfficientNetB0
import numpy as np
from Evaluation_nrml import evaluation

def model(input_shape=(224, 224, 3)):
  model = EfficientNetB0(input_shape=input_shape, include_top=False)
  return model

def Model_EFFICIENTNET(train_data, train_target, test_data, test_target):

    IMG_SIZE = 224
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    Model = model()
    Model.fit(train_data, train_target, epochs=1, batch_size=150)
    pred = Model.predict(test_data).ravel()

    Eval = evaluation(pred, test_target)
    return Eval
