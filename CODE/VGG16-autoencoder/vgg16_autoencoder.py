#%tensorflow_version 1.x
import os
import numpy as np
import random
import gc
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
print('Tensorflow Version:', tf.__version__)

import requests
from io import BytesIO
import sys

import keras
from keras.preprocessing import image
from keras.engine import Layer
from keras.layers import Conv2D, Conv3D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image, ImageFile
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model, Sequence

def get10KImageArray(FOLDER_PATH, target_size, isColor,isSplit,val_split=0.1, test_split=0.1):
  image_files_lst = os.listdir(FOLDER_PATH)
  image_files_lst.sort()
  image_files_lst=image_files_lst[-20000:]
  num_images = len(image_files_lst)
  x = []
  if isColor == True:
    color_mode = 'rgb'
  else:
    color_mode = 'grayscale'
  
  for img in image_files_lst:
    if '.png' in img or '.jpg' in img:
      img_arr = img_to_array(load_img(path=FOLDER_PATH + '/' + img, target_size=target_size, color_mode=color_mode)) * 1.0/255
      x.append(img_arr)
  x = np.array(x)
  print('Finished converting', str(num_images), 'images as numpy arrays!')
  print('Image pixels are in range', x.min(), 'to', x.max())
  if isSplit==True:
    train, test = train_test_split(x, test_size=test_split)
    train, val = train_test_split(train, test_size=val_split)
    print('Train Shape:', train.shape)
    print('Val   Shape:', val.shape)
    print('Test  Shape:', test.shape)
    return train, test, val
  return x

def getImageArray(FOLDER_PATH, target_size, isColor,isSplit,val_split=0.1, test_split=0.1):
  image_files_lst = os.listdir(FOLDER_PATH)
  image_files_lst.sort()
  num_images = len(image_files_lst)
  x = []
  if isColor == True:
    color_mode = 'rgb'
  else:
    color_mode = 'grayscale'
  for img in image_files_lst:
    if '.png' in img or '.jpg' in img:
      img_arr = img_to_array(load_img(path=FOLDER_PATH + '/' + img, target_size=target_size, color_mode=color_mode)) * 1.0/255
      x.append(img_arr)
  x = np.array(x)
  print('Finished converting', str(num_images), 'images as numpy arrays!')
  print('Image pixels are in range', x.min(), 'to', x.max())
  if isSplit==True:
    train, test = train_test_split(x, test_size=test_split)
    train, val = train_test_split(train, test_size=val_split)
    print('Train Shape:', train.shape)
    print('Val   Shape:', val.shape)
    print('Test  Shape:', test.shape)
    return train, test, val
  return x

def getVggBatch(batch):
  lab_batch = rgb2lab(batch)
  X_batch = lab_batch[:,:,:,0]
  X_batch=X_batch.reshape(X_batch.shape+(1,))
  vggfeatures = []
  for i, sample in enumerate(X_batch):
    sample=gray2rgb(sample)
    sample=sample.reshape((1,224,224,3))
    prediction = newmodel.predict(sample)
    prediction=prediction.reshape((7,7,512))
    vggfeatures.append(prediction)
  vggfeatures=np.array(vggfeatures)
  print('The vggfeatures shape is:', vggfeatures.shape)
  Y_batch = lab_batch[:,:,:,1:] / 128
  return (vggfeatures, Y_batch)

class DataGenerator(Sequence):
    def __init__(self, datagen, isInres=False):
        self.datagen = datagen
        #self.isInres = isInres
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.datagen)
    
    def on_epoch_end(self):
        self.datagen.on_epoch_end()

    def __getitem__(self, idx):
        'Generates one batch of data'
        batch = self.datagen[idx]
        lab_batch = rgb2lab(batch)
        X = lab_batch[:, :, :, 0] # scale to [0, 1]
        X = np.expand_dims(X, axis=3)
        X = gray2rgb(X)
        X = X.reshape((len(batch), 224, 224, 3))
        prediction = newmodel.predict(X)
        prediction = prediction.reshape((len(batch), 7,7,512))
        Y = lab_batch[:, :, :, 1:]/128

        return prediction, Y


def predictTestArrays(arrays,target_size):
  cur_pred=[]
  test_color=[]
  
  for array in arrays:
    #test=img_to_array(load_img(testpath+file))
    
    test=resize(array,target_size,anti_aliasing=True)
    lab=rgb2lab(test)
    l=lab[:,:,0]
    #reprocess to feed vgg16.encoder
    L=gray2rgb(l)
    L=L.reshape((1,224,224,3))
    vggpred=newmodel.predict(L)
    ab=model.predict(vggpred)
    ab=ab*128
    cur=np.zeros((224,224,3))
    cur[:,:,0]=l
    cur[:,:,1:]=ab
    cur_pred.append(cur)
    test_color.append(test)
  return cur_pred,test_color 

#plot_model(newmodel,'vgg16_encoder.png',show_shapes=True, show_layer_names=True)
def vggAEmodel():
  #Encoder
  encoder_input = Input(shape=(7, 7, 512,))
  #Decoder
  decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_input)
  decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
  decoder_output = UpSampling2D((2, 2))(decoder_output)
  model = Model(inputs=encoder_input, outputs=decoder_output)
  print(model.summary())
  
  model.compile(optimizer='Adam', loss='mse' , metrics=['mae'])
  return model

def saveImage(arr, FOLDER_PATH):
    counter = 0
    for img in arr:
        imsave(FOLDER_PATH + '/' + str(counter) + '.png', lab2rgb(img))
        counter += 1
    print('Finished saving', counter, 'images!')

if __name__ == '__main__':
    if len(sys.argv)<2:
        print("Function usaage: python vgg16_autoencoder.py ../data/test-color/vg ../data/test-pred/vg")
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print('input file is ', input_file +'\n')
    print('output file is ', output_file + '\n')
    
    
    BATCH_SIZE = 64
    TARGET_SIZE = (224, 224)
    #load vgg model
    vggmodel = keras.applications.vgg16.VGG16()
    newmodel = Sequential() 
    num = 0
    for i, layer in enumerate(vggmodel.layers):
    #if i not in [3,6]:
      if i<19:
        newmodel.add(layer)
    #newmodel.summary()
    for layer in newmodel.layers:
        layer.trainable=False
    newmodel._make_predict_function() 
    #get data
    train,test,val = getImageArray(input_file, TARGET_SIZE, isColor=True,isSplit=True,val_split=0.2, test_split=0.2)
   
    

    # specify image data generators
    train_datagen = ImageDataGenerator(data_format='channels_last', validation_split=0.2, 
                                   rotation_range=40, shear_range=0.2, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, 
                                   horizontal_flip=True, vertical_flip=True)

    train_gen = DataGenerator(train_datagen.flow(train, subset='training',  batch_size=BATCH_SIZE))
    val_gen = DataGenerator(train_datagen.flow(test, subset='validation', batch_size=BATCH_SIZE))
     #load model
    model=vggAEmodel()
    model._make_predict_function() 
    
    start = time.time()
    history=model.fit_generator(train_gen,epochs=1,steps_per_epoch=len(train_gen),verbose=1,validation_data=val_gen,validation_steps=len(val_gen))
    end=time.time()-start
    print('Training time is:',end)

    # load weights
    model.load_weights('../weights/vgg16_weights.h5')
    
    cur_pred,test_color=predictTestArrays(test,TARGET_SIZE)
    
    saveImage(arr=cur_pred, FOLDER_PATH=output_file)
