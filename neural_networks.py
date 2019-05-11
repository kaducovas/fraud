import numpy			as np
import pandas			as pd

import dirs
import defines			  as defs
import keras
from keras.models import Sequential
# from keras.regularizers import l1, l2
# from keras.layers.core import Dense, Activation, Dropout
# from keras.optimizers import Adam, SGD
# import keras.callbacks as callbacks
# from keras.utils import np_utils
# from keras.models import load_model
# from keras import backend as K

def multi_layer_perceptron(x_train, y_train, x_test, y_test, hidden_neurons=10,
						 hidden_activation='tanh', output_activation='tanh', lossFunction='mean_squared_error',
						 optmizer='Adam', metrics=['mae','mape','acc','categorical_accuracy']):
	'''
		Neural Networks classifier.

		x_train, x_test: DataFrames of shape data x features.
	'''

	model = Sequential()
	model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation=hidden_activation))
	model.add(Dense(1, activation=output_activation))
	model.compile(loss=lossFunction, optimizer=optmizer, metrics=metrics)

	return model
