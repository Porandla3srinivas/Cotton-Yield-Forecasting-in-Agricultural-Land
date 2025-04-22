from keras.applications import EfficientNetB0
import numpy as np
from keras.layers import *
from keras import backend as K
from Evaluation_nrml import evaluation

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        super(attention,self).build(input_shape)


def call(self, x):
    e = K.tanh(K.dot(x,self.W)+self.b)
    a = K.softmax(e, axis=1)
    output = x*a
    if self.return_sequences:

        return output
    return K.sum(output, axis=1)

def model(sol,input_shape=(224, 224, 3)):
    model = EfficientNetB0(input_shape=input_shape, include_top=False,filters_in=sol[0])
    return model

def Model_AM_EFFICIENTNET(train_data, train_target, test_data, test_target,sol=None):
    if sol is None:
        sol = [32,1,150]
    IMG_SIZE = 224
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    Eval = []
    for a in range(3):  # Multiscale
        for i in range(test_data.shape[0]):
            temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
            Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
        Model = model(sol)
        Model.fit(train_data, train_target, epochs=sol[1], batch_size=sol[2])
        pred = Model.predict(test_data).ravel()
        Eval.append(evaluation(pred, test_target))
    Evals = (Eval[0] + Eval[1] + Eval[2])/3
    return Evals
