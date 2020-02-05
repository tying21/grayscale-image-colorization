from Data_Preprocessing import getImageArray, getProcessedData, saveImage
from Baseline_CNN_Model import CNN
import numpy as np

def predict(model, X, Y):
    eval_results = model.evaluate(X, Y, verbose=0)
    print('Loss:', np.round(eval_results[0], 4), '\tMSE:', np.round(eval_results[1], 4))
    pred = model.predict(X)
    canvas = np.zeros(X.shape[:-1]+(3,))
    canvas[:, :, :, 0] = (X[:, :, :, 0] + 1) * 50
    canvas[:, :, :, 1:] = pred * 128
    return canvas


if __name__ == '__main__':
    TARGET_SIZE = (224, 224)
    VAL_SPLIT = 0.131
    BATCH_SIZE = 32
    TARGET_SIZE = (224, 224)
    # get image array
    data = getImageArray(FOLDER_PATH='../data/test-color/flower', target_size=TARGET_SIZE, isColor=True)

    # process image array
    X, Y = getProcessedData(data)

    # build model
    model = CNN(input_shape=TARGET_SIZE+(1,), num_filters=64, kernel_size=(3,3), num_strides=1,num_layers=4, activation='relu',
                   kernel_initializer='he_uniform', kernel_regularizer=None,
                     addBN=True, optimizer='adam', loss='mse', model_name='base_l2')

    # load weights
    model.load_weights('../weights/Base_CNN_weights.hdf5')

    # predict output
    pred = predict(model=model, X=X, Y=Y)

    # save predictions and gray version of data
    saveImage(arr=pred, FOLDER_PATH='../data/test-pred/flower/')

