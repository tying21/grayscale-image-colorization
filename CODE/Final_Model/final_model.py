
'''
Builds final Inception-ResNet-v2 Transfer Learning Model
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, RepeatVector, concatenate, UpSampling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def conv_block(x, num_filters, kernel_size, num_strides, activation, kernel_initializer, kernel_regularizer, addBN, addDR, dRate, name):
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=num_strides, 
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, 
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



def buildModel(input_shape, input_shape_inres, num_filters, kernel_size, num_layers, activation, kernel_initializer, kernel_regularizer, addBN, addDR, dRate, optimizer, loss, model_name):
    inp = Input(shape=input_shape, name='input')
    x = inp
    
    # encoder
    for i in range(num_layers):
        x = conv_block(x, num_filters=num_filters, kernel_size=kernel_size, num_strides=2,
                       activation=activation, kernel_initializer=kernel_initializer, 
                       kernel_regularizer=kernel_regularizer, addBN=addBN, addDR=addDR, dRate=dRate, 
                       name='en_stridePool_' + str(num_filters))
        num_filters = num_filters * 2
        x = conv_block(x, num_filters=num_filters, kernel_size=kernel_size, num_strides=1,
                       activation=activation, kernel_initializer=kernel_initializer, 
                       kernel_regularizer=kernel_regularizer, addBN=addBN, addDR=addDR, dRate=dRate, 
                       name='en_' + str(num_filters))
    encoder_shape = K.int_shape(x)

    # inres
    inres_inp = Input(shape=input_shape_inres, name='inres_input')
    inres_out = Dense(units=int(input_shape_inres[-1] / 3), activation=activation, 
                      kernel_initializer=kernel_initializer, name='inres_dense')(inres_inp)
    inres_out = RepeatVector(encoder_shape[1] * encoder_shape[2], name='inres_repeat')(inres_out)
    inres_out = Reshape((encoder_shape[1], encoder_shape[2], int(input_shape_inres[-1] / 3)), name='inres_reshape')(inres_out)
    inres_out = concatenate([x, inres_out], axis=3, name='inres_concat')
    num_filters = int(num_filters / 2)
    x = conv_block(inres_out, num_filters=num_filters, kernel_size=1, num_strides=1,
                   activation=activation, kernel_initializer=kernel_initializer, 
                   kernel_regularizer=kernel_regularizer, addBN=addBN, addDR=addDR, dRate=dRate, 
                   name='inres_conv' + str(num_filters))
    
    # decoder
    for i in range(num_layers):
        num_filters = int(num_filters / 2)
        x = conv_block(x, num_filters=num_filters, kernel_size=kernel_size, num_strides=1,
                       activation=activation, kernel_initializer=kernel_initializer, 
                       kernel_regularizer=kernel_regularizer, addBN=addBN, addDR=addDR, dRate=dRate, 
                        name='de_' + str(num_filters))
        x = UpSampling2D(name='de_upsample' + str(num_filters))(x)
    
    out = conv_block(x, num_filters=2, kernel_size=kernel_size, num_strides=1,
                     activation='tanh', kernel_initializer=kernel_initializer, 
                     kernel_regularizer=kernel_regularizer, addBN=False, addDR=False, dRate=0.0, 
                     name='output')
    
    model = Model([inp, inres_inp], out, name=model_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    model.summary()
    return model

