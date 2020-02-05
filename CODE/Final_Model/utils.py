
'''
Preprocess data and other utils for Models
'''

import os
import numpy as np
import random
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import warnings
warnings.filterwarnings("ignore")


def getImageArray(FOLDER_PATH, target_size=(224, 224), isColor=True):
    img_files_lst = os.listdir(FOLDER_PATH)
    img_files_lst.sort()
    num_images = len(img_files_lst)
    x = []
    if isColor == True:
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'
    for img in img_files_lst:
        if '.png' in img or '.jpg' in img:
            img_arr = img_to_array(load_img(path=FOLDER_PATH + '/' + img, target_size=target_size, color_mode=color_mode)) * 1.0/255
            x.append(img_arr)
    x = np.array(x)
    print('Finished converting', len(x), 'images as numpy arrays!')
    print('Image pixels are in range', x.min(), 'to', x.max())
    print('Shape of image array:', x.shape)
    return x

def getProcessedDataFinal(arr, inres_model):
    arr_lab = rgb2lab(arr)
    X = arr_lab[:, :, :, 0] / 100
    X = np.expand_dims(X, axis=3)
    Y = arr_lab[:, :, :, 1:] / 128
    X_inres = np.concatenate([X, X, X], axis=3)
    X_inres = inres_model.predict(X_inres)
    print('Shape of X:', X.shape, 'with pixels in range', X.min(), 'to', X.max())
    print('Shape of X_inres:', X.shape, 'with pixels in range', X_inres.min(), 'to', X_inres.max())
    print('Shape of Y:', Y.shape, 'with pixels in range', Y.min(), 'to', Y.max())
    return X, X_inres, Y

def saveImagePred(arr, FOLDER_PATH):
    counter = 0
    for img in arr:
        imsave(FOLDER_PATH + str(counter) + '.png', lab2rgb(img))
        counter += 1
    print('Finished saving', counter, ' prediction images!')

