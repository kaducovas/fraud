# Train Process
#from Functions import LogFunctions as log

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam
import keras.callbacks as callbacks
from keras.utils import np_utils
#from keras.layers import Merge

from sklearn.externals import joblib

#from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import numpy.linalg as la
from keras.models import load_model

#m_time = time.time()

class NonLinPCA:

    def __init__(self, save_path=''):
        self.development_flag = False
        self.save_path        = save_path
        self.analysis_str     = 'NLPCA'

    def trainNLPCA(self, data=None, trgt=None,n_inits = 1, n_nlpcas=30, n_neurons_mapping=50, learning_rate=0.01,learning_decay=0.00001, momentum=0.3, nesterov=True, train_verbose=True, n_epochs=5000, batch_size=200,sort=999,etBinIdx=999, etaBinIdx=999, tuning_folder=None):

        # Create a train information file
        #n_folds = n_folds
        n_inits = n_inits
        n_nlpcas = n_nlpcas
        print "qq eh isso"
        train_info = {}
        #train_info['n_folds'] = n_folds
        train_info['n_inits'] = n_inits
        train_info['n_nlpcas'] = n_nlpcas
        train_info['n_neurons_mapping'] = n_neurons_mapping

        # train info
        train_info['learning_rate'] = learning_rate
        train_info['learning_decay'] = learning_decay
        train_info['momentum'] = momentum
        train_info['nesterov'] = nesterov
        train_info['train_verbose'] = train_verbose
        train_info['n_epochs'] = n_epochs
        train_info['batch_size'] = batch_size

        file_name = 'inits_%i_bottleneck_%i_mapping_%i_epochs_%i_sort_%i_etbin_%i_etabin_%i'%(n_inits, n_nlpcas, n_neurons_mapping,n_epochs,sort,etBinIdx, etaBinIdx)

        train_info_name = self.save_path+'/train_info_files'+'/'+file_name+'_train_info.jbl'
        classifiers_name = self.save_path+'/classifiers_files'+'/'+file_name+'_classifiers'
        nlpcas_file_name = self.save_path+'/output_files'+'/'+file_name


        # CVO = cross_validation.StratifiedKFold(all_trgt, train_info['n_folds'])
        # CVO = list(CVO)
        # train_info['CVO'] = CVO

        joblib.dump([train_info],train_info_name,compress=9)

        # train classifiers
        nlpca_extractor = {}
        trn_desc = {}
        nlpca = {}

        nlpca_extractor = {}
        trn_desc = {}

        #for inlpca in [n_nlpcas]: #range(1,train_info['n_nlpcas']+1):
        best_init = 0
        best_loss = 999

        for i_init in range(train_info['n_inits']):
            print ('Fold: %i - NLPCA: %i of %i - Init: %i of %i'
                   %(sort+1,
                     n_nlpcas, train_info['n_nlpcas'],
                     i_init+1,train_info['n_inits']))
            model = Sequential()
            # model.add(Dense(all_data.shape[1],
                            # input_dim=all_data.shape[1],
                            # init='identity',trainable=False))
            # model.add(Activation('linear'))
            model.add(Dense(n_neurons_mapping, input_dim=data.shape[1], init='uniform'))
            model.add(Activation('tanh'))
            model.add(Dense(n_nlpcas,init='uniform'))
            model.add(Activation('tanh'))
            model.add(Dense(n_neurons_mapping,init='uniform'))
            model.add(Activation('tanh'))
            model.add(Dense(data.shape[1],init='uniform'))
            model.add(Activation('linear'))

            sgd = SGD(lr=train_info['learning_rate'],
                      decay=train_info['learning_decay'],
                      momentum=train_info['momentum'],
                      nesterov=train_info['nesterov'])

            adamOpt = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
            model.compile(loss='mean_squared_error',
                          optimizer=adamOpt,
                          metrics=['accuracy'])

            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    verbose=0,
                                                    mode='auto')
            # Train model
            import keras
            nlcpca_encoding_name  = 'nlpca_adam_bottleneck_%i_mapping_%i_epochs_sort_%i_etbin_%i_etabin_%i'%(n_nlpcas, n_neurons_mapping,sort,etBinIdx, etaBinIdx)
            tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/caducovas/tensorboard/graphs/'+nlcpca_encoding_name, histogram_freq=10, write_graph=True, write_images=True,write_grads=True,update_freq='epoch')

            import time
            import datetime
            start_run = time.time()

            init_trn_desc = model.fit(data, data,
                                      nb_epoch=train_info['n_epochs'],
                                      batch_size= 1024, #train_info['batch_size'], #1024
                                      callbacks=[earlyStopping,tbCallBack],
                                      verbose=train_info['train_verbose'],
                                      validation_data=(trgt,
                                                       trgt),
                                      shuffle=True)

            end_run = time.time()
            print 'Model NLPCA took '+ str(datetime.timedelta(seconds=(end_run - start_run))) +' to finish.'

            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                nlpca_extractor[n_nlpcas] = model
                trn_desc[n_nlpcas] = {}
                trn_desc[n_nlpcas]['epochs'] = init_trn_desc.epoch
                trn_desc[n_nlpcas]['perf'] = init_trn_desc.history['loss']
                trn_desc[n_nlpcas]['vperf'] = init_trn_desc.history['val_loss']

        (nlpca_extractor[n_nlpcas].save(
                '%s.h5'%(nlpcas_file_name)))


        joblib.dump([trn_desc],'%s_train_desc.jbl'%(nlpcas_file_name),compress=9)

        file_name_prefix  = 'inits_%i_bottleneck_%i_mapping_%i_epochs_%i_sort_%i_etbin_%i_etabin_%i'%(n_inits, n_nlpcas, n_neurons_mapping,n_epochs,sort,etBinIdx, etaBinIdx)
        with open(self.save_path+tuning_folder,'a+') as t_file:
            t_file.write(file_name_prefix+ "\n")
        t_file.close()
        return


    def getDataProjection(self, data, trgt, n_inits = 1, n_nlpcas=30, n_neurons_mapping=50,n_epochs=5000,sort=999,etBinIdx=999, etaBinIdx=999):

        #choose_date = '2016_09_21_20_56_00'
        proj_all_data = data
        file_name = 'inits_%i_bottleneck_%i_mapping_%i_epochs_%i_sort_%i_etbin_%i_etabin_%i'%(n_inits, n_nlpcas, n_neurons_mapping,n_epochs,sort,etBinIdx, etaBinIdx)
        # load train info
        train_info_name = self.save_path+'/train_info_files'+'/'+file_name+'_train_info.jbl'
        train_info = joblib.load(train_info_name)
        train_info = train_info[0]
        nlpcas_file_name = self.save_path+'/output_files'+'/'+file_name

        classifiers = {}
        results = {}

        # get the output of an intermediate layer
        from keras import backend as K

        #for ifold in range(train_info['n_folds']):
            #if ifold > 1:
            #    continue

            # classifiers = {}
            # results = {}

            # nlpcas_file_name = self.save_path+'/output_files'+'/'+choose_date+'_nlpcas'

        #for inlpca in range(train_info['n_nlpcas']):

        nlpca_model = load_model('%s.h5'%(nlpcas_file_name))
        print "Loading Model: "+file_name
            # best_init = 0
            # best_loss = 999

            # with a Sequential model
        get_layer_output = K.function([nlpca_model.layers[0].input],
                              [nlpca_model.layers[3].output])
        data_proj_nlpca = get_layer_output([proj_all_data])[0]
        return data_proj_nlpca

            # ###Daqui pra baixo eh o classificador. Acho que nao precisa
            # for i_init in range(train_info['n_inits']):
                # print ("Processing Fold: %i of %i - NLPCA %i of %i - Init %i of %i"%
                       # (ifold+1,train_info['n_folds'],
                        # inlpca+1,train_info['n_nlpcas'],
                        # i_init+1,train_info['n_inits']))

                # model = Sequential()
                # model.add(Dense(data_proj_nlpca.shape[1],
                                # input_dim=data_proj_nlpca.shape[1],
                                # init='identity',trainable=False))
                # model.add(Activation('linear'))
                # model.add(Dense(50, input_dim=data_proj_nlpca.shape[1], init='uniform'))
                # model.add(Activation('tanh'))
                # model.add(Dense(trgt_sparse.shape[1], init='uniform'))
                # model.add(Activation('tanh'))

                # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                # model.compile(loss='mean_squared_error',
                              # optimizer=sgd, metrics=['accuracy'])

                # earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=25,
                                                # verbose=0, mode='auto')
                # # Train model
                # init_trn_desc = model.fit(data_proj_nlpca[train_id], trgt_sparse[train_id],
                                          # nb_epoch=50,
                                          # batch_size=8,
                                          # callbacks=[earlyStopping],
                                          # verbose=0,
                                          # validation_data=(data_proj_nlpca[test_id],
                                                           # trgt_sparse[test_id]),
                                          # shuffle=True)
                # if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    # best_init = i_init
                    # best_loss = np.min(init_trn_desc.history['val_loss'])
                    # classifiers[ifold][inlpca] = model

            # classifiers[ifold][inlpca].save('%s_classifiers_fold_%i_inlpca_%i.h5'%(nlpcas_file_name,ifold,inlpca))
