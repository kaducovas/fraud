# -*- coding: utf-8 -*-
'''
   Author: Vinícius dos Santos Mello viniciusdsmello at poli.ufrj.br
   Class created to implement a Stacked Autoencoder for Classification.
'''
import os
import pickle
import numpy as np
import time

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics

#import sys
#sys.path.remove('/scratch/22061a/common-cern/pyenv/versions/2.7.9/lib/python2.7/site-packages')
#import tensorflow
import keras
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
#from keras import losses
from keras import regularizers
from TuningTools import SAE_TrainParameters as trnparams
from TuningTools.MetricsLosses import kullback_leibler_divergence, contractive_loss
from keras.losses import kullback_leibler_divergence
import keras.losses
#keras.losses.custom_loss = contractive_loss
import keras.backend as K
#import multiprocessing

#num_process = multiprocessing.cpu_count()

class StackedAutoEncoders:
    def __init__(self, params = None, development_flag = False, n_folds = 1, save_path='', prefix_str='RawData',aetype='vanilla',dataEncoded='all',nIteration=9,layerNumber=None, CVO=None,
                 noveltyDetection=False, inovelty = 0):
        self.trn_params       = params
        self.development_flag = development_flag
        self.n_folds          = n_folds
        self.save_path        = save_path
        self.noveltyDetection = noveltyDetection

        self.n_inits          = self.trn_params.params['n_inits']
        self.params_str       = self.trn_params.get_params_str()
        self.analysis_str     = 'StackedAutoEncoder'
        self._aetype = aetype
        self._dataEncoded = dataEncoded
        self._layerNumber = layerNumber
        self._nIteration = nIteration
        # Distinguish between a SAE for Novelty Detection and SAE for 'simple' Classification
        if noveltyDetection:
            self.CVO          = CVO[inovelty]
            self.prefix_str   = prefix_str+'_%i_novelty'%(inovelty)
        else:
            self.CVO          = CVO
            if dataEncoded == 'all':
              self.prefix_str   = prefix_str+'-'+aetype
            else:
              self.prefix_str   = prefix_str+'-'+aetype+'-'+dataEncoded
            if aetype == 'discriminant':
              self.prefix_str   = self.prefix_str+'-'+str(nIteration)+'Iter'

        # Choose optmizer algorithm
        if self.trn_params.params['optmizerAlgorithm'] == 'SGD':
            self.optmizer = SGD(lr=self.trn_params.params['learning_rate'],
                                    # momentum=self.trn_params.params['momentum'],
                                    # decay=self.trn_params.params['decay'],
                                    nesterov=self.trn_params.params['nesterov'])

        elif self.trn_params.params['optmizerAlgorithm'] == 'Adam':
            self.optmizer = Adam(lr=self.trn_params.params['learning_rate'],
                                    beta_1=self.trn_params.params['beta_1'],
                                    beta_2=self.trn_params.params['beta_2'],
                                    epsilon=self.trn_params.params['epsilon'])
        else:
            self.optmizer = self.trn_params.params['optmizerAlgorithm']

        # Choose loss functions
        if self.trn_params.params['loss'] == 'kullback_leibler_divergence':
            self.lossFunction = kullback_leibler_divergence
        else:
            self.lossFunction = self.trn_params.params['loss']
        #from keras import losses
        #losses.custom_loss = self.lossFunction
    '''
        Method that creates a string in the format: (InputDimension)x(1º Layer Dimension)x...x(Nº Layer Dimension)
    '''
    def getNeuronsString(self, data, hidden_neurons=[]):
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons:
            neurons_str = neurons_str + 'x' + str(ineuron)
        return neurons_str
    '''
        Method that preprocess data normalizing it according to 'norm' parameter.
    '''
    def normalizeData(self, data, ifold):
        #normalize data based in train set
        train_id, test_id = self.CVO[ifold]
        if self.trn_params.params['norm'] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(data[train_id,:])
        elif self.trn_params.params['norm'] == 'mapstd_rob':
           scaler = preprocessing.RobustScaler().fit(data[train_id,:])
        elif self.trn_params.params['norm'] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
        norm_data = scaler.transform(data)

        return norm_data
    '''
        Method that returns the output of an intermediate layer.
    '''
    def getDataProjection(self, data, trgt,transformed_data=None, hidden_neurons=[80], layer=1, ifold=0,sort=999,etBinIdx=999, etaBinIdx=999):
        if layer > len(hidden_neurons):
            print "[-] Error: The parameter layer must be less or equal to the size of list hidden_neurons"
            return 1
        proj_all_data = data #self.normalizeData(data=data, ifold=ifold)
        #print sort,etBinIdx,etaBinIdx
        #print 'LOSSSS'
        #print self.trn_params.params['loss']
        #print self.lossFunction


        if layer == 1:
            neurons_str = self.getNeuronsString(data, hidden_neurons[:layer])
            previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path,
                                                                    self.analysis_str,
                                                                    self.prefix_str,
                                                                    self.n_folds,
                                                                    self.params_str,
                                                                    neurons_str)
            if not self.development_flag:
                file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model.h5'%(previous_model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
            else:
                file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model_dev.h5'%(previous_model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)

            # Check if previous layer model was trained
            if not os.path.exists(file_name):
                self.trainLayer(data=data, trgt=trgt,transformed_data=transformed_data, ifold=ifold, hidden_neurons = hidden_neurons[:layer], layer=layer, folds_sweep=True)

            custom_obj={}
            if self._aetype == 'contractive':
              from TuningTools.MetricsLosses import contractive_loss
              custom_obj['contractive_loss']=contractive_loss(hidden_neurons[layer-1],data.shape[1],self.trn_params.params['hidden_activation'],self.trn_params.params['output_activation'])
            else:
              custom_obj[self.trn_params.params['loss']]= self.lossFunction

            layer_model = load_model(file_name, custom_objects=custom_obj)
            #layer_model = load_model(file_name, custom_objects={'loss': usedloss}) ###self.lossFunction})
            #layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
            print "Loading Model: "+file_name
            get_layer_output = K.function([layer_model.layers[0].input],
                                          [layer_model.layers[1].output])
            # Projection of layer
            proj_all_data = get_layer_output([proj_all_data])[0]
        elif layer > 1:
            for ilayer in range(1,layer+1):
                neurons_str = self.getNeuronsString(data, hidden_neurons[:ilayer])
                previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path,
                                                                        self.analysis_str,
                                                                        self.prefix_str,
                                                                        self.n_folds,
                                                                        self.params_str,
                                                                        neurons_str)
                if not self.development_flag:
                    file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model.h5'%(previous_model_str,sort,etBinIdx, etaBinIdx,self._layerNumber)
                else:
                    file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model_dev.h5'%(previous_model_str,sort,etBinIdx, etaBinIdx,self._layerNumber)

                # Check if previous layer model was trained
                if not os.path.exists(file_name):
                    self.trainLayer(data=data, trgt=trgt,transformed_data=transformed_data, ifold=ifold, hidden_neurons = hidden_neurons[:ilayer], layer=ilayer, folds_sweep=True)

                custom_obj={}
                if self._aetype == 'contractive':
                  from TuningTools.MetricsLosses import contractive_loss
                  custom_obj['contractive_loss']=contractive_loss(hidden_neurons[layer-1],data.shape[1],self.trn_params.params['hidden_activation'],self.trn_params.params['output_activation'])
                else:
                  custom_obj[self.trn_params.params['loss']]= self.lossFunction
                print "Loading Model: "+file_name
                layer_model = load_model(file_name, custom_objects=custom_obj)
                #layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                print "Model Loaded"
                get_layer_output = K.function([layer_model.layers[0].input],
                                              [layer_model.layers[1].output])
                # Projection of layer
                proj_all_data = get_layer_output([proj_all_data])[0]
        return proj_all_data

    '''
        Method used to perform the layerwise algorithm to train the SAE
    '''
    def trainLayer(self, data=None, trgt=None,transformed_data=None, ifold=0, hidden_neurons = [80], layer=1, folds_sweep=False,
                   regularizer=None, regularizer_param=None,sort=999,etBinIdx=999, etaBinIdx=999, tuning_folder=None):
        # Change elements equal to zero to one
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        if self.trn_params.params['verbose']:
            print '[+] Using %s as optmizer algorithm'%self.trn_params.params['optmizerAlgorithm']

        neurons_str = self.getNeuronsString(data,hidden_neurons[:layer])

        if regularizer != None and len(regularizer) != 0:
            model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path, self.analysis_str,
                                                                              self.prefix_str, self.n_folds,
                                                                              self.params_str, neurons_str)
        else:
            model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path, self.analysis_str,
                                                           self.prefix_str, self.n_folds,
                                                           self.params_str, neurons_str)
        if not self.development_flag:
            file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model.h5'%(model_str,sort,etBinIdx, etaBinIdx,self._layerNumber)
            #file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            #print file_name
            if os.path.exists(file_name):
                #print 'LOSSSS'
                #print self.trn_params.params['loss']
                #print self.lossFunction
                custom_obj={}
                if self._aetype == 'contractive':
                  # def contractive_loss(y_pred, y_true):
                      # lam = 1e-4
                      # mse = K.mean(K.square(y_true - y_pred), axis=1)

                      # W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
                      # W = K.transpose(W)  # N_hidden x N
                      # h = model.get_layer('encoded').output
                      # dh = h * (1 - h)  # N_batch x N_hidden

                      # # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
                      # contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

                      # return mse + contractive
                  # #usedloss=contractive_loss
                  # #self.lossFunction=
                  from TuningTools.MetricsLosses import contractive_loss
                  custom_obj['contractive_loss']=contractive_loss(hidden_neurons[layer-1],data.shape[1],self.trn_params.params['hidden_activation'],self.trn_params.params['output_activation'])
                else:
                  custom_obj[self.trn_params.params['loss']]= self.lossFunction
                  #usedloss=self.lossFunction
                #print usedloss


                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model.h5'%(model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
                #file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
                #classifier = load_model(file_name, custom_objects={'loss': usedloss}) ###self.lossFunction})
                classifier = load_model(file_name, custom_objects=custom_obj)
                #classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_trn_desc.jbl'%(model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
                trn_desc   = joblib.load(file_name)

                file_name_prefix = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i'%(model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
                with open(self.save_path+tuning_folder,'a+') as t_file:
                    t_file.write(file_name_prefix+ "\n")
                t_file.close()
                return ifold, classifier, trn_desc
        else:
            file_name = '%s_sort_%i_etbin_%i_etabin_%i_model_dev.h5'%(model_str,sort,etBinIdx, etaBinIdx)
            #file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                file_name = '%s_sort_%i_etbin_%i_etabin_%i_model_dev.h5'%(model_str,sort,etBinIdx, etaBinIdx)
                #file_name  = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_sort_%i_etbin_%i_etabin_%i_trn_desc_dev.jbl'%(model_str,sort,etBinIdx, etaBinIdx)
                trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc

        #train_id, test_id = self.CVO[ifold]

        #norm_data = self.normalizeData(data, ifold)
        #norm_data = data

        best_init = 0
        best_loss = 9999999

        classifier = []
        trn_desc = {}

        print 'Number of SAE training inits: '+str(self.n_inits)
        for i_init in range(self.n_inits):
            #print 'Number of SAE training inits: '+ str(self.n_inits)
            print 'Layer: %i - Neuron: %i - Fold %i of %i Folds -  Init %i of %i Inits'%(self._layerNumber,
                                                                                         hidden_neurons[layer-1],
                                                                                         ifold+1,
                                                                                         self.n_folds,
                                                                                         i_init+1,
                                                                                         self.n_inits)
            model = Sequential()
            #proj_all_data = norm_data
            proj_all_data = data
            if layer == 1:
                print 'LAYER 1'
                print hidden_neurons[layer-1], data.shape[1]
                if regularizer == 'l1':
                  model.add(Dense(hidden_neurons[layer-1], input_dim=data.shape[1], activity_regularizer=regularizers.l1(regularizer_param), name='encoded'))
                else:
                  model.add(Dense(hidden_neurons[layer-1], input_dim=data.shape[1], name='encoded'))
                model.add(Activation(self.trn_params.params['hidden_activation']))
                if regularizer == "dropout":
                    model.add(Dropout(regularizer_param))
                #elif regularizer == "l1":
                #    model.regularizers = [l1(regularizer_param)]
                #elif regularizer == "l2":
                #    model.regularizers = [l2(regularizer_param)]
                model.add(Dense(data.shape[1]))
                model.add(Activation(self.trn_params.params['output_activation']))
            elif layer > 1:
                for ilayer in range(1,layer):
                    print hidden_neurons[:ilayer]
                    neurons_str = self.getNeuronsString(data, hidden_neurons[:ilayer])
                    previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path,
                                                                            self.analysis_str,
                                                                            self.prefix_str,
                                                                            self.n_folds,
                                                                            self.params_str,
                                                                            neurons_str)
                    if not self.development_flag:
                        file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model.h5'%(previous_model_str,sort,etBinIdx, etaBinIdx,self._layerNumber)
                        #file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                    else:
                        file_name = '%s_sort_%i_etbin_%i_etabin_%i_model_dev.h5'%(previous_model_str,sort,etBinIdx, etaBinIdx)
                        #file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                    # Check if previous layer model was trained
                    if not os.path.exists(file_name):
                        self.trainLayer(data=data, trgt=trgt,transformed_data=transformed_data, ifold=ifold, hidden_neurons = hidden_neurons[:ilayer], layer=ilayer, folds_sweep=True)

                    custom_obj={}
                    if self._aetype == 'contractive':
                      from TuningTools.MetricsLosses import contractive_loss
                      custom_obj['contractive_loss']=contractive_loss(hidden_neurons[layer-1],data.shape[1],self.trn_params.params['hidden_activation'],self.trn_params.params['output_activation'])
                    else:
                      custom_obj[self.trn_params.params['loss']]= self.lossFunction
                    #print "Loading Model: "+file_name
                    layer_model = load_model(file_name, custom_objects=custom_obj)
                    #layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})

                    get_layer_output = K.function([layer_model.layers[0].input],
                                                  [layer_model.layers[1].output])
                    # Projection of layer
                    proj_all_data = get_layer_output([proj_all_data])[0]

                model.add(Dense(hidden_neurons[layer-1], input_dim=proj_all_data.shape[1], name='encoded'))
                model.add(Activation(self.trn_params.params['hidden_activation']))
                if regularizer == "dropout":
                    model.add(Dropout(regularizer_param))
                elif regularizer == "l1":
                    model.regularizers = [l1(regularizer_param)]
                elif regularizer == "l2":
                    model.regularizers = [l2(regularizer_param)]
                model.add(Dense(proj_all_data.shape[1]))
                model.add(Activation(self.trn_params.params['output_activation']))
                #norm_data = proj_all_data
                data = proj_all_data
            # end of elif layer > 1:

            if self._aetype == 'contractive':
              def contractive_loss(y_pred, y_true):
                  lam = 1e-4
                  mse = K.mean(K.square(y_true - y_pred), axis=1)

                  W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
                  W = K.transpose(W)  # N_hidden x N
                  h = model.get_layer('encoded').output
                  dh = h * (1 - h)  # N_batch x N_hidden

                  # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
                  contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

                  return mse + contractive
              usedloss=contractive_loss
            else:
              usedloss=self.lossFunction
            model.compile(loss=usedloss, #self.lossFunction,
                          optimizer=self.optmizer,
                          metrics=self.trn_params.params['metrics'])

            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience= 5, #self.trn_params.params['patience'],
                                                    verbose=self.trn_params.params['train_verbose'],
                                                    mode='auto')




            ae_encoding_name  = '%s_%i_folds_%s_%s_neurons'%(self.prefix_str, self.n_folds,self.params_str, neurons_str)
            ae_encoding = ae_encoding_name.split('_')
            ae_encoding_string = ae_encoding[0]+'_'+ae_encoding[24]+'_sort_%i_et_%i_eta_%i_layer_%i'%(sort,etBinIdx, etaBinIdx, self._layerNumber)
            tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/caducovas/tensorboard/EncodingError/'+ae_encoding_string, histogram_freq=1, write_graph=True, write_images=True,write_grads=True,update_freq='batch')

            import time
            import datetime
            start_run = time.time()
            if self._aetype == 'discriminant':
              init_trn_desc = model.fit(data, transformed_data,
                                        nb_epoch=300, #self.trn_params.params['n_epochs'],
                                        batch_size= 1024, #self.trn_params.params['batch_size'],
                                        callbacks=[earlyStopping], #, tbCallBack],
                                        verbose=1)

            else:
              init_trn_desc = model.fit(data, data,
                                        nb_epoch=self.trn_params.params['n_epochs'],
                                        batch_size= 1024, #self.trn_params.params['batch_size'],
                                        callbacks=[earlyStopping], #, tbCallBack],
                                        verbose=1, #self.trn_params.params['verbose'],
                                        validation_data=(trgt,
                                                         trgt))

            end_run = time.time()
            print 'Model took '+ str(datetime.timedelta(seconds=(end_run - start_run))) +' to finish.'

            if self._aetype == 'discriminant':
                classifier = model
                trn_desc['epochs'] = init_trn_desc.epoch
                trn_desc['loss'] = init_trn_desc.history['loss']
            else:
                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = i_init
                    best_loss = np.min(init_trn_desc.history['val_loss'])
                    classifier = model
                    trn_desc['epochs'] = init_trn_desc.epoch

                    #for imetric in range(len(self.trn_params.params['metrics'])):
                    #    if self.trn_params.params['metrics'][imetric] == 'kullback_leibler_divergence':
                    #        metric = kullback_leibler_divergence
                    #    else:
                    #        metric = self.trn_params.params['metrics'][imetric]
                    #    trn_desc[metric] = init_trn_desc.history[metric]
                    #    trn_desc['val_'+metric] = init_trn_desc.history['val_'+metric]

                    trn_desc['loss'] = init_trn_desc.history['loss']
                    trn_desc['val_losS'] = init_trn_desc.history['val_loss']
                    trn_desc['kullback_leibler_divergence'] = init_trn_desc.history['kullback_leibler_divergence']
                    trn_desc['val_kullback_leibler_divergence'] = init_trn_desc.history['val_kullback_leibler_divergence']

        model.summary()
        # save model
        if not self.development_flag:
            file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_model.h5'%(model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
            #file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i_trn_desc.jbl'%(model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
            #file_name = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)

            file_name_prefix = '%s_sort_%i_etbin_%i_etabin_%i_layer_%i'%(model_str,sort,etBinIdx, etaBinIdx, self._layerNumber)
            with open(self.save_path+tuning_folder,'a+') as t_file:
                t_file.write(file_name_prefix+ "\n")
            t_file.close()

        else:
            file_name = '%s_sort_%i_etbin_%i_etabin_%i_model_dev.h5'%(model_str,sort,etBinIdx, etaBinIdx)
            #file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_sort_%i_etbin_%i_etabin_%i_trn_desc_dev.jbl'%(model_str,sort,etBinIdx, etaBinIdx)
            #file_name = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        return ifold, classifier, trn_desc

    '''
        Method that return the classifier according to topology parsed
    '''
    def loadClassifier(self, data=None, trgt=None, hidden_neurons=[1], layer=1, ifold=0):
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        # Turn trgt to one-hot encoding
        trgt_sparse = np_utils.to_categorical(trgt.astype(int))

        # load model
        neurons_str = self.getNeuronsString(data,hidden_neurons[:layer]) + 'x' + str(trgt_sparse.shape[1])
        model_str = '%s/%s/Classification_(%s)_%s_%i_folds_%s'%(self.save_path,
                                                                self.analysis_str,
                                                                neurons_str,
                                                                self.prefix_str,
                                                                self.n_folds,
                                                                self.params_str)

        classifier = {}
        if not self.development_flag:
            file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
            try:
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
            except:
                print '[-] Error: File or Directory not found'
        else:
            file_name  = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            try:
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
            except:
                print '[-] Error: File or Directory not found'
        return classifier

    '''
        Function used to do a Fine Tuning in Stacked Auto Encoder for Classification of the data
        hidden_neurons contains the number of neurons in the sequence: [FirstLayer, SecondLayer, ... ]
    '''
    def trainClassifier(self, data=None, trgt=None, ifold=0, hidden_neurons=[1], layer=1):
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        # Turn trgt to one-hot encoding
        trgt_sparse = np_utils.to_categorical(trgt.astype(int))

        # load or create cross validation ids
        CVO = trnparams.ClassificationFolds(folder=self.save_path,
                                            n_folds=self.n_folds,
                                            trgt=trgt,
                                            dev=self.development_flag)

        neurons_str = self.getNeuronsString(data,hidden_neurons[:layer]) + 'x' + str(trgt_sparse.shape[1])
        model_str = '%s/%s/Classification_(%s)_%s_%i_folds_%s'%(self.save_path,
                                                                self.analysis_str,
                                                                neurons_str,
                                                                self.prefix_str,
                                                                self.n_folds,
                                                                self.params_str)
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
                trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                classifier = {}
                trn_desc = {}
                if not self.development_flag:
                    file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
                    classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                    file_name  = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
                    trn_desc   = joblib.load(file_name)
                else:
                    file_name  = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
                    classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                    file_name  = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
                    trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc
        train_id, test_id = CVO[ifold]

        norm_data = self.normalizeData(data, ifold)

        best_init = 0
        best_loss = 999

        classifier = []
        trn_desc = {}

        for i_init in range(self.n_inits):
            print 'Layer: %i - Fold: %i of %i Folds -  Init: %i of %i Inits'%(layer,
                                                                           ifold+1,
                                                                           self.n_folds,
                                                                           i_init+1,
                                                                           self.n_inits)
            # Start the model
            model = Sequential()
            # Add layers
            for ilayer in range(1,layer+1):
                 # Get the weights of ilayer
                neurons_str = self.getNeuronsString(data,hidden_neurons[:ilayer])
                previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path,
                                                                        self.analysis_str,
                                                                        self.prefix_str,
                                                                        self.n_folds,
                                                                        self.params_str,
                                                                        neurons_str)

                if not self.development_flag:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                # Check if the layer was trained
                if not os.path.exists(file_name):
                    self.trainLayer(data=data,
                                    trgt=data,
                                    ifold=ifold,
                                    layer=ilayer,
                                    hidden_neurons = hidden_neurons[:ilayer])


                layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                layer_weights = layer_model.layers[0].get_weights()
                if ilayer == 1:
                    model.add(Dense(hidden_neurons[0], input_dim=norm_data.shape[1], weights=layer_weights,
                                    trainable=True))
                else:
                    model.add(Dense(hidden_neurons[ilayer-1], weights=layer_weights, trainable=True))

                model.add(Activation(self.trn_params.params['hidden_activation']))

            # Add Output Layer
            model.add(Dense(trgt_sparse.shape[1]))
            model.add(Activation('softmax'))

            model.compile(loss=self.trn_params.params['loss'],
                          optimizer=self.optmizer,
                          metrics=self.trn_params.params['metrics'])
            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.trn_params.params['patience'],
                                                    verbose=self.trn_params.params['train_verbose'],
                                                    mode='auto')

            init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id],
                                      epochs=self.trn_params.params['n_epochs'],
                                      batch_size=self.trn_params.params['batch_size'],
                                      callbacks=[earlyStopping],
                                      verbose= self.trn_params.params['verbose'],
                                      validation_data=(norm_data[test_id],
                                                       trgt_sparse[test_id]),
                                      shuffle=True)
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifier = model
                trn_desc['epochs'] = init_trn_desc.epoch
                trn_desc['acc'] = init_trn_desc.history['acc']
                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']
                trn_desc['val_acc'] = init_trn_desc.history['val_acc']

        # save model
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        return ifold, classifier, trn_desc
