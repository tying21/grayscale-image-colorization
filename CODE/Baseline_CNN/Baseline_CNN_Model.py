from scipy.io import loadmat
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, RepeatVector, concatenate, UpSampling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, Sequence
import tensorflow as tf
import numpy as np

def conv_block(x, num_filters, kernel_size, num_strides, activation, kernel_initializer, addBN, addDR, dRate, name):
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=num_strides, kernel_initializer=kernel_initializer,
               padding='same', name=name)(x)
    if addBN == True:
        x = BatchNormalization(name='bn_' + name)(x)
    if addDR == True:
        x = Dropout(dRate, name='dr_' + name)(x)
    x = Activation(activation, name='act_' + name)(x)
    return x

def convT_block(x, num_filters, kernel_size, num_strides, activation, kernel_initializer, kernel_regularizer, addBN, addDR, dRate, name):
    x = Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=num_strides,
                        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                        padding='same', name=name)(x)
    if addBN == True:
        x = BatchNormalization(name='bn_' + name)(x)
    if addDR == True:
        x = Dropout(dRate, name='dr_' + name)(x)
    x = Activation(activation, name='act_' + name)(x)
    return x

def CNN(input_shape, num_filters, kernel_size, num_strides, num_layers, activation, kernel_initializer,
        kernel_regularizer, addBN, optimizer, loss, model_name):
    in_layer = Input(shape=input_shape)
    x = in_layer
    # multiply starting filters with [1, 2, 4, 8, 16] if num_layers = 4
    for i in [2 ** j for j in range(num_layers)]:
        x = Conv2D(filters=num_filters * i, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, padding='same', strides=num_strides,
                   name='conv_' + str(num_filters * i))(x)
        x = Activation(activation, name='conv_act_' + str(num_filters * i))(x)
        if addBN == True:
            x = BatchNormalization(name='conv_BN_' + str(num_filters * i))(x)
        x = Conv2D(filters=num_filters * i, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, padding='same', strides=2,
                   name='stridePool_' + str(num_filters * i))(x)
        x = Activation(activation, name='stridePool_act_' + str(num_filters * i))(x)
        if addBN == True:
            x = BatchNormalization(name='stridePool_BN_' + str(num_filters * i))(x)

    # multiply starting filters with [16, 8, 4, 2, 1] if num_layers = 5
    for i in [2 ** j for j in range(num_layers - 1, -1, -1)]:
        x = Conv2DTranspose(filters=num_filters * i, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer, padding='same', strides=2,
                            name='convT_' + str(num_filters * i))(x)
        x = Activation(activation, name='convT_act_' + str(num_filters * i))(x)
        if addBN == True:
            x = BatchNormalization(name='convT_BN_' + str(num_filters * i))(x)

    out_layer = Conv2D(filters=2, kernel_size=kernel_size, activation='tanh', padding='same', strides=1)(x)
    model = Model(in_layer, out_layer, name=model_name + '_CNN')
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae']
        )
    return model