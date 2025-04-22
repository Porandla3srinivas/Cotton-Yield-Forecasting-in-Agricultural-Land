from keras.layers import *
import keras
from keras.optimizers import Adam
import random as rn
import sys
import warnings
import matplotlib
import tflearn
from tflearn.layers.estimator import regression
import numpy as np
matplotlib.use('agg')
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import layers
import tensorflow as tf
import cv2 as cv


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

def ResUnetp(input_size=(128, 128,128,3)):
    inputs = Input(input_size)
    embed_dim = 32
    num_heads = 3
    ff_dim = 100
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x=transformer_block(inputs)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2,2))(conv1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2, 2))(conv3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2, 2))(drop4)

    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2,2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2,2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2,2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)
    outputs = Conv3D(1, (1,1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def Train_3d_trans_resunetp(train_data, train_target, test_data):
    if test_data is None:
        test_data = train_data
    # Set some parameters
    IMG_SIZE = 128

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    rn.seed = seed
    np.random.seed = seed

    X_train = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE,IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((train_target.shape[0], IMG_SIZE, IMG_SIZE,IMG_SIZE, 3), dtype=np.uint8)
    X_test = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Y_test = np.zeros((test_target.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_train[i, :, :, :] = np.resize(Temp,(IMG_SIZE, IMG_SIZE, 3)).astype('uint8')

    for i in range(train_target.shape[0]):
        Temp = cv.resize(train_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = np.resize(Temp,(IMG_SIZE, IMG_SIZE, 3)).astype('uint8')
        for j in range(Temp.shape[0]):
            for k in range(Temp.shape[1]):
                for l in range(Temp.shape[2]):
                    if Temp[j, k,l] < 0.5:
                        Temp[j, k,l] = 0
                    else:
                        Temp[j, k,l] = 1
        Y_train[i, :, :, :] = Temp

    for i in range(test_data.shape[0]):
        Temp = cv.resize(test_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_test[i, :, :, :] = np.resize(Temp,(IMG_SIZE, IMG_SIZE, 3)).astype('uint8')

    '''for i in range(test_target.shape[0]):
        Temp = cv.resize(test_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = Temp.reshape((IMG_SIZE, IMG_SIZE, 1))
        for j in range(Temp.shape[0]):
            for k in range(Temp.shape[1]):
                if Temp[j, k] < 0.5:
                    Temp[j, k] = 0
                elif Temp[j, k] >= 0.5:
                    Temp[j, k] = 1
        Y_test[i, :, :, :] = Temp'''
    sys.stdout.flush()
    model = ResUnetp()
    X_test = np.resize(X_test,[4,128,128,128,3]).astype('uint8')
    pred_img = model.predict(X_test)
    ret_img = pred_img
    return ret_img


class automaticmaplabelling():
    def __init__(self, modelPath, full_chq, X_test, width, height, channels, model):
        self.modelPath = modelPath
        self.full_chq = full_chq
        self.X_test = X_test
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.IMG_CHANNELS = channels
        self.model = model

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def prediction(self):
        X_test = self.X_test
        Y_Pred = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        preds_test = self.model.predict(X_test, verbose=1)
        preds_test = (preds_test > 0.5).astype(np.uint8)
        for i in range(preds_test.shape[0]):
            mask = preds_test[i]
            for j in range(mask.shape[0]):
                for k in range(mask.shape[1]):
                    if mask[j][k] >= 1:
                        mask[j][k] = 255
                    else:
                        mask[j][k] = 0
            Y_Pred[i] = mask
        return Y_Pred




