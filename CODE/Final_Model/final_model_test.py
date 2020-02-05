
'''
Trains final Inception-ResNet-v2 Transfer Learning Model
'''

import numpy as np
from final_model import buildModel
from utils import getImageArray, getProcessedDataFinal, saveImagePred
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import warnings
warnings.filterwarnings("ignore")


def predict(model, X, X_inres, Y):
    eval_results = model.evaluate([X, X_inres], Y, verbose=0)
    print('Test  L2 Loss:', np.round(eval_results[0], 4), '\tMAE:', np.round(eval_results[1], 4))
    pred = model.predict([X, X_inres])
    print('Finished predicting output with Shape:', pred.shape)
    canvas = np.zeros(X.shape[:-1]+(3,))
    canvas[:, :, :, 0] = X[:, :, :, 0] * 100
    canvas[:, :, :, 1:] = pred * 128
    print('Final prediction Shape:', canvas.shape)
    return canvas


if __name__ == '__main__':
	TARGET_SIZE = (256, 256)
	
	# build inres model
	print('\nBuilding Inception-ResNet-v2 Model')
	inres_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=TARGET_SIZE+(3,), pooling='avg')
	print('INRES MODEL INPUT  SHAPE:', inres_model.input.shape)
	print('INRES MODEL OUTPUT SHAPE:', inres_model.output.shape)

	# get image array
	print('\nGetting VG images as numpy array')
	data_vg = getImageArray(FOLDER_PATH='../data/test-color/vg', target_size=TARGET_SIZE, isColor=True)
	print('\nGetting Flower images as numpy array')
	data_flower = getImageArray(FOLDER_PATH='../data/test-color/flower', target_size=TARGET_SIZE, isColor=True)

	# process image array
	print('\nProcessing VG images array')
	X_vg, X_inres_vg, Y_vg = getProcessedDataFinal(arr=data_vg, inres_model=inres_model)
	print('\nProcessing Flower images array')
	X_flower, X_inres_flower, Y_flower = getProcessedDataFinal(arr=data_flower, inres_model=inres_model)

	# build model
	print('\nBuilding Model')
	model = buildModel(input_shape=TARGET_SIZE+(1,), input_shape_inres=(inres_model.output.shape[-1],), 
		num_filters=64, kernel_size=3, num_layers=3, activation='relu', 
		kernel_initializer='he_normal', kernel_regularizer=None,
		addBN=True, addDR=False, dRate=0.0, optimizer='adam', loss='mse', model_name='final_model')

	# load weights
	print('\nLoading VG Weights')
	model.load_weights('../weights/final-model-vg_weights.hdf5')

	# predict output
	print('Making Predictions')
	pred_vg = predict(model=model, X=X_vg, X_inres=X_inres_vg, Y=Y_vg)

	# save predictions as images
	print('Saving predictions as images')
	saveImagePred(arr=pred_vg, FOLDER_PATH='../data/test-pred/vg/')

	# load weights
	print('\nLoading Flower Weights')
	model.load_weights('../weights/final-model-flower_weights.hdf5')

	# predict output
	print('Making Predictions')
	pred_flower = predict(model=model, X=X_flower, X_inres=X_inres_flower, Y=Y_flower)

	# save predictions as images
	print('Saving predictions as images')
	saveImagePred(arr=pred_flower, FOLDER_PATH='../data/test-pred/flower/')


