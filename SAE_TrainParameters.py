"""
    This file contents some classification analysis functions
"""

import os
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
#from TuningTools.MetricsLosses import kullback_leibler_divergence
from keras.losses import kullback_leibler_divergence
from TuningTools.MetricsLosses import contractive_loss

class TrnParams(object):
    """
        Basic class
    """
    def __init__(self, analysis="None"):
        self.analysis = analysis
        self.params = None

    def save(self, name="None"):
        joblib.dump([self.params],name,compress=9)

    def load(self, name="None"):
        [self.params] = joblib.load(name)

    def printParams(self):
        for iparameter in self.params:
            print iparameter + ': ' + str(self.params[iparameter])


# classification

def ClassificationFolds(folder, n_folds=1, trgt=None, dev=False, verbose=True):

    #if n_folds < 2:
        #print 'Invalid number of folds'
        #return -1

    if not dev:
        file_name = '%s/%i_folds_cross_validation.jbl'%(folder,n_folds)
    else:
        file_name = '%s/%i_folds_cross_validation_dev.jbl'%(folder,n_folds)

    if not os.path.exists(file_name):
        if verbose:
            print "Creating %s"%(file_name)

        if trgt is None:
            print 'Invalid trgt'
            return -1

        CVO = model_selection.StratifiedKFold(trgt, n_folds)
        CVO = list(CVO)
        joblib.dump([CVO],file_name,compress=9)
    else:
        if verbose:
            print "File %s exists"%(file_name)
        [CVO] = joblib.load(file_name)

    return CVO

class NeuralClassificationTrnParams(TrnParams):
    """
        Neural Classification TrnParams
    """

    def __init__(self,
                 n_inits=1,
                 norm='mapstd',
                 verbose=True,
                 train_verbose=True,
                 n_epochs=500,
                 learning_rate=0.001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-08,
                 learning_decay=1e-6,
                 momentum=0.3,
                 nesterov=True,
                 patience=5,
                 batch_size=4,
                 hidden_activation='tanh',
                 output_activation='linear',
                 metrics=['kullback_leibler_divergence'],
                 loss='mean_squared_error',
                 optmizerAlgorithm='Adam'
                ):
        self.params = {}

        self.params['n_inits'] = n_inits
        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['train_verbose'] = train_verbose

        # train params
        self.params['n_epochs'] = n_epochs
        self.params['learning_rate'] = learning_rate
        self.params['beta_1'] = beta_1
        self.params['beta_2'] = beta_2
        self.params['epsilon'] = epsilon
        self.params['learning_decay'] = learning_decay
        self.params['momentum'] = momentum
        self.params['nesterov'] = nesterov
        self.params['patience'] = patience
        self.params['batch_size'] = batch_size
        self.params['hidden_activation'] = hidden_activation
        self.params['output_activation'] = output_activation
        self.params['metrics'] = metrics
        self.params['loss'] = loss
        self.params['optmizerAlgorithm'] = optmizerAlgorithm

    def get_params_str(self):
        param_str = ('%i_inits_%i_epochs_%i_batch_size_%s_hidden_activation_%s_output_activation_%s_metric_%s_loss'%
                     (self.params['n_inits'],self.params['n_epochs'],self.params['batch_size'],
                      self.params['hidden_activation'],self.params['output_activation'],
                      self.params['metrics'][0],
                      self.params['loss']))
        return param_str

# novelty detection
