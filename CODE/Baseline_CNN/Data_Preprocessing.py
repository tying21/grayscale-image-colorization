import os
import numpy as np
import tensorflow as tf
print('Tensorflow Version:', tf.__version__)
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb, deltaE_ciede2000, deltaE_cie76
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split
from skimage.io import imsave

BATCH_SIZE = 64
TARGET_SIZE = (224,224)

# Data Processing Functions
def getImageArray(FOLDER_PATH, target_size=TARGET_SIZE, isColor=True, isSplit=False, test_split=0.1):
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
    print('Finished converting', str(num_images), 'images as numpy arrays!')
    print('Image pixels are in range', x.min(), 'to', x.max())
    if isSplit == True:
        train, test = train_test_split(x, test_size=test_split)
        print('Train Shape:', train.shape)
        print('Test  Shape:', test.shape)
        return train, test
    return x

def concatLandAB(color_me,output):
    result_list=[]
    for i in range(len(output)):
        result = np.zeros((224, 224, 3))
        result[:,:,0] = color_me[i][:,:,0]
        result[:,:,1:] = output[i]
        #imsave("result"+i+".png", lab2rgb(result))
        result_list.append(result)
    return result_list

def getProcessedData(arr):
    arr_lab = rgb2lab(arr)
    X = 2 * arr_lab[:, :, :, 0] / 100 - 1 # scale to [-1, 1]
    X = np.expand_dims(X, axis=3)
    Y = arr_lab[:, :, :, 1:] / 128 # scale to [-1, 1]
    print('Shape of X:', X.shape, 'with pixels in range', X.min(), 'to', X.max())
    print('Shape of Y:', Y.shape, 'with pixels in range', Y.min(), 'to', Y.max())
    return X,  Y

def saveImage(arr, FOLDER_PATH):
    counter = 0
    for img in arr:
        imsave(FOLDER_PATH + str(counter) + '.png', lab2rgb(img))
        counter += 1
    print('Finished saving', counter, 'images!')