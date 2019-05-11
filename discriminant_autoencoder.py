from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
import numpy as np
import os
import re
import seaborn as sb
import cPickle
import pickle
import TuningTools
import pandas as pd
import numpy
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 12,4.8
import math
from sklearn.preprocessing import MinMaxScaler
#15, 6
import matplotlib.pyplot as plt
import time
#import statsmodels
#from statsmodels.tsa.stattools import adfuller
#import statsmodels.api as sm
import scipy
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import numpy as np
np.random.seed(43)
from keras.models import Model
from keras.layers import Dense, GaussianNoise, GaussianDropout, Input, Conv2D, Flatten, Conv2DTranspose, Dropout
from keras.layers.normalization import BatchNormalization
from keras.activations import relu


def move_to_means(train_data, train_classes_data, a=0.2):
    import copy
    train_classes = copy.deepcopy(train_classes_data)

    n_classes = 2 #np.max(train_classes)+1
    print '------------------- n_classes : ', n_classes, ' a : ', a, ' -----------------------'
    centers = np.zeros((n_classes, train_data.shape[1]))
    new_train_data = np.zeros(train_data.shape)

    train_classes[train_classes==1] = 0
    train_classes[train_classes==-1] = 1


    for i in range(n_classes):
        indices = np.squeeze(np.where(train_classes==i)[0])
        centers[i,...] = np.mean(train_data[indices,...], axis=0)
    for i in range(train_data.shape[0]):
        new_train_data[i,...] = (1-a)*train_data[i,...] + a*centers[int(train_classes[i]),...]
    return new_train_data


def move_away_from_means(train_data, train_classes_data, a=0.1, k=0):

    import copy
    train_classes = copy.deepcopy(train_classes_data)

    train_classes[train_classes==1] = 0
    train_classes[train_classes==-1] = 1

    n_classes = 2 #np.max(train_classes)+1
    centers = np.zeros((n_classes, train_data.shape[1]))
    for j in range(n_classes):
        indices = np.squeeze(np.where(train_classes==j)[0])
        centers[j, ...] = np.mean(train_data[indices, ...], axis=0)
    new_train_data = np.zeros(train_data.shape)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
    #print centers
    nbrs.fit(centers)
    n_ind = nbrs.kneighbors(train_data, n_neighbors=k + 1, return_distance=False)
    for i in range(train_data.shape[0]):
        new_train_data[i,...] = (1-a)*train_data[i,...] - a*np.squeeze(centers[np.abs(n_ind[i]-1)])
    return new_train_data


def move_to_neighbors(train_data, train_classes_data, k=3, a=0.2):

    import copy
    train_classes = copy.deepcopy(train_classes_data)
    train_classes[train_classes==1] = 0
    train_classes[train_classes==-1] = 1

    n_classes = 2 #np.max(train_classes) + 1
    new_train_data = np.zeros(train_data.shape)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    for i in range(n_classes):
        #find indices of the given class
        ind = np.squeeze(np.where(train_classes == i)[0])
        #train_data[ind,...].shape = (n_samples of class i, features) da classe
        nbrs.fit(train_data[ind, ...])
        #n_dis and n_ind are the distance and indices of the k closest points, includind the point itself
        #n_dis, n_ind shape = (n_samples of class, k)
        n_dis, n_ind = nbrs.kneighbors(train_data[ind, ...], k, return_distance=True)
        #para cada amostra
        for j in range(train_data[ind,...].shape[0]):
            #neight.shape = (k-1, features), sendo as variaveis dos vizinhos mais proximos excluindo a amosrta original
            neigh = train_data[ind,...][n_ind[j,...], ...][1:, ...]
            #anda na direcao dos k-1 vizinhos mais proximos da mesma classe
            new_train_data[ind[j],...] = (1-a)*train_data[ind[j],...] + a*np.mean(neigh, axis=0)
    return new_train_data


def move_to_safer_neighbors(train_data, train_classes_data, k=3, a=0.2):

    import copy
    train_classes = copy.deepcopy(train_classes_data)
    train_classes[train_classes==1] = 0
    train_classes[train_classes==-1] = 1

    n_classes = 2 #np.max(train_classes) + 1
    centers = np.zeros((n_classes, train_data.shape[1]))
    for j in range(n_classes):
        indices = np.squeeze(np.where(train_classes == j)[0])
        centers[j, ...] = np.mean(train_data[indices, ...], axis=0)
    new_train_data = np.zeros(train_data.shape)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    for i in range(n_classes):
        ind = np.squeeze(np.where(train_classes == i)[0])
        nbrs.fit(train_data[ind, ...])
        n_ind = nbrs.kneighbors(train_data[ind, ...], k, return_distance=False)
        for j in range(train_data[ind,...].shape[0]):
            neigh = train_data[ind,...][n_ind[j,...], ...]
            s_neigh = []
            s = 0
            for n in range(k):
                if euclidean(neigh[n,...], centers[train_classes[j,...],...]) <= euclidean(train_data[j,...], centers[train_classes[j,...],...]):
                    s_neigh.append(neigh[n,...])
                    s = s+1
            if s>0:
                new_train_data[ind[j],...] = (1-a)*train_data[ind[j],...] + a*np.mean(np.squeeze(np.asarray(s_neigh)), axis=0)
            else:
                new_train_data[ind[j], ...] = train_data[ind[j], ...]
    return new_train_data


def move_away_from_rivals(train_data, train_classes_data, k=3, a=0.2):

    import copy
    train_classes = copy.deepcopy(train_classes_data)

    train_classes[train_classes==1] = 0
    train_classes[train_classes==-1] = 1

    n_classes = 2 #np.max(train_classes) + 1
    new_train_data = train_data
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    # nbrs = LSHForest(random_state=43, n_neighbors=k, n_estimators=2)
    for i in range(n_classes):
        ind = np.squeeze(np.where(train_classes != i)[0])
        sc_ind = np.squeeze(np.where(train_classes == i)[0])
        nbrs.fit(train_data[ind,...])
        if sc_ind.size >1:
            #retorna o indice dos k vizinhos mais proximos da amostra, pertencentes a outra classe
            n_ind = nbrs.kneighbors(train_data[sc_ind,...], k, return_distance=False)
            for j in range(train_data[sc_ind,...].shape[0]):
                if sc_ind.size>1:
                    #neight.shape = (k, features), sendo as variaveis dos vizinhos mais proximos pertencentes a outra classe
                    neigh = train_data[ind,...][n_ind[j,...],...]
                    new_train_data[sc_ind[j],...] = (1+a)*train_data[sc_ind[j],...] - a*np.squeeze(np.mean(neigh, axis=0))
    return new_train_data


import time
import datetime

def transform_data(x_values,train_target,nIteration=9):
    for i in range(nIteration):
        print "Iteration: "+str(i+1)

        print "Starting Move to Means ..."
        start_run= time.time()
        new_x_values  = move_to_means(x_values, train_target)
        end_run = time.time()
        print "Move to Means took: " + str(datetime.timedelta(seconds=(end_run - start_run)))
        #np.savez_compressed(dirout+'new_train_x_iter_'+str(i+1)+'_move_to_means.npz',new_x_values)

        print "Starting Move Away from Means ..."
        start_run= time.time()
        new_x_values = move_away_from_means(new_x_values, train_target)
        end_run = time.time()
        print "Move Away from Means tool: " + str(datetime.timedelta(seconds=(end_run - start_run)))
        #np.savez_compressed(dirout+'new_train_x_iter_'+str(i+1)+'move_away_from_means.npz',new_x_values)

        x_values = new_x_values
    return x_values
