import numpy                as np
import pandas               as pd
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.preprocessing  import MinMaxScaler

import dirs
import defines              as defs

def least_squares(x_train, y_train, x_test, y_test, thres=0.5):
    '''
        Train Least Squares classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''

    lmq = LinearRegression(fit_intercept=True, n_jobs=1)

    lmq.fit(x_train, y_train)#, weights) # TODO: Add class weights

    predScores = lmq.predict(x_test)

    # Normalize predictions to [0, 1] range
    scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
    scaler.fit_transform(predScores.reshape(-1,1))

    # Classify test data according to threshold
    predictions = np.where(predScores >= thres, defs.posCode, defs.negCode)

    return predictions

def ridge_least_squares(x_train, y_train, x_test, y_test, thres=0.5, regVal=1.0):
    '''
        Train Least Squares classifier with L2 regularization on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''

    rls = Ridge(alpha=regVal, fit_intercept=True)

    rls.fit(x_train, y_train) # TODO: Add class weights
    predScores = rls.predict(x_test)

    # Normalize predictions to [0, 1] range
    scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
    scaler.fit_transform(predScores.reshape(-1,1))

    # Classify test data according to threshold
    predictions = np.where(predScores >= thres, defs.posCode, defs.negCode)

    return predictions
