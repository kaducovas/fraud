__all__ = [ 'DataTrainEvolution', 'Layer', 'Neural', 'NeuralCollection'
          , 'Neural']

import numpy as np
from RingerCore import LimitedTypeList, checkForUnusedVars, Logger, NotSet, RawDictStreamable
from TuningTools.TuningJob import ReferenceBenchmark
from TuningTools.coreDef import npCurrent

class DataTrainEvolution:

  """
    Class TrainDataEvolution is a sub class. This hold the train evolution into a
    list. Basically this is like a c++ struct.
  """
  def __init__(self, train=None, full_data=False):
    #Slim data
    self.mse_trn        = list()  
    self.mse_val        = list()  
    self.mse_tst        = list()  
    
    self.bestsp_point_sp_val     = list()  
    self.bestsp_point_det_val    = list()  
    self.bestsp_point_fa_val     = list()  
    self.bestsp_point_sp_tst     = list()  
    self.bestsp_point_det_tst    = list()  
    self.bestsp_point_fa_tst     = list()  
    self.det_point_sp_val        = list()  
    self.det_point_det_val       = list()  
    self.det_point_fa_val        = list()  
    self.det_point_sp_tst        = list()  
    self.det_point_det_tst       = list()  
    self.det_point_fa_tst        = list()  
    self.fa_point_sp_val         = list()  
    self.fa_point_det_val        = list()  
    self.fa_point_fa_val         = list()  
    self.fa_point_sp_tst         = list()  
    self.fa_point_det_tst        = list()  
    self.fa_point_fa_tst         = list()  

    #Train evolution information
    is_best_mse    = list() 
    is_best_sp     = list() 
    is_best_det    = list() 
    is_best_fa     = list()  
 
    if full_data:
      self.num_fails_mse  = list() 
      self.num_fails_sp   = list() 
      self.num_fails_det  = list() 
      self.num_fails_fa   = list() 
      self.stop_mse       = list() 
      self.stop_sp        = list()    
      self.stop_det       = list() 
      self.stop_fa        = list() 

    #Get train evolution information from TrainDatapyWrapper
    if train is not None:
      from coreDef import coreConf, TuningToolCores
      if coreConf() is TuningToolCores.FastNet:
        self.maxEpoch = len(train)
        for i in range(len(train)):
          
          self.mse_trn.append              ( train[i].mseTrn               )
          self.mse_val.append              ( train[i].mseVal               )
          self.mse_tst.append              ( train[i].mseTst               )

          self.bestsp_point_sp_val.append  ( train[i].bestsp_point_sp_val  )
          self.bestsp_point_det_val.append ( train[i].bestsp_point_det_val )
          self.bestsp_point_fa_val.append  ( train[i].bestsp_point_fa_val  )
          self.bestsp_point_sp_tst.append  ( train[i].bestsp_point_sp_tst  )
          self.bestsp_point_det_tst.append ( train[i].bestsp_point_det_tst )
          self.bestsp_point_fa_tst.append  ( train[i].bestsp_point_fa_tst  )
   
          self.det_point_sp_val.append     ( train[i].det_point_sp_val     )
          self.det_point_det_val.append    ( train[i].det_point_det_val    )
          self.det_point_fa_val.append     ( train[i].det_point_fa_val     )
          self.det_point_sp_tst.append     ( train[i].det_point_sp_tst     )
          self.det_point_det_tst.append    ( train[i].det_point_det_tst    )
          self.det_point_fa_tst.append     ( train[i].det_point_fa_tst     )
   
          self.fa_point_sp_val.append      ( train[i].fa_point_sp_val      )
          self.fa_point_det_val.append     ( train[i].fa_point_det_val     )
          self.fa_point_fa_val.append      ( train[i].fa_point_fa_val      )
          self.fa_point_sp_tst.append      ( train[i].fa_point_sp_tst      )
          self.fa_point_det_tst.append     ( train[i].fa_point_det_tst     )
          self.fa_point_fa_tst.append      ( train[i].fa_point_fa_tst      )
          
          is_best_mse.append               ( train[i].isBestMse            )
          is_best_sp.append                ( train[i].isBestSP             )
          is_best_det.append               ( train[i].isBestDet            )
          is_best_fa.append                ( train[i].isBestFa             )

          if full_data:
            self.num_fails_mse.append ( train[i].numFailsMse )
            self.num_fails_sp.append  ( train[i].numFailsSP  )
            self.num_fails_det.append ( train[i].numFailsDet )
            self.num_fails_fa.append  ( train[i].numFailsFa  )
            self.stop_mse.append      ( train[i].stopMse     )
            self.stop_sp.append       ( train[i].stopSP      )
            self.stop_det.append      ( train[i].stopDet     )
            self.stop_fa.append       ( train[i].stopFa      )

        self.epoch_best_mse = self.__lastIndex(is_best_mse,  True)
        self.epoch_best_sp  = self.__lastIndex(is_best_sp ,  True)
        self.epoch_best_det = self.__lastIndex(is_best_det,  True)
        self.epoch_best_fa  = self.__lastIndex(is_best_fa ,  True)
      elif coreConf() is TuningToolCores.keras:
        self.maxEpoch = len(train.epoch)
        self.mse_trn = train.history['loss']
        self.mse_val = train.history['val_loss']
        self.acc_trn = train.history['acc']
        self.acc_val = train.history['val_acc']
        keys = train.history.keys()
        sp_keys = [key for key in keys if '_SP_' in key]
        pd_keys = [key for key in keys if '_Pd_' in key]
        pf_keys = [key for key in keys if '_Pf_' in key]
        if sp_keys:
          self.bestsp_point_sp_val  = train.history[[sp_key for sp_key in sp_keys if sp_key.endswith('sp_value')][0]]
          self.bestsp_point_det_val = train.history[[sp_key for sp_key in sp_keys if sp_key.endswith('pd_value')][0]]
          self.bestsp_point_fa_val  = train.history[[sp_key for sp_key in sp_keys if sp_key.endswith('pf_value')][0]]

        if pd_keys:
          self.det_point_sp_val     = train.history[[pd_key for pd_key in pd_keys if pd_key.endswith('sp_value')][0]]
          self.det_point_det_val    = train.history[[pd_key for pd_key in pd_keys if pd_key.endswith('pd_value')][0]]
          self.det_point_fa_val     = train.history[[pd_key for pd_key in pd_keys if pd_key.endswith('pf_value')][0]]
   
        if pf_keys:
          self.fa_point_sp_val      = train.history[[pf_key for pf_key in pf_keys if pf_key.endswith('sp_value')][0]]
          self.fa_point_det_val     = train.history[[pf_key for pf_key in pf_keys if pf_key.endswith('pd_value')][0]]
          self.fa_point_fa_val      = train.history[[pf_key for pf_key in pf_keys if pf_key.endswith('pf_value')][0]]

  def toRawObj(self):
    "Return a raw dict object from itself"
    from copy import copy # Every complicated object shall be changed to a rawCopyObj
    raw = copy(self.__dict__)
    return raw

  def __lastIndex(self,  l, value ):
    try:
      l.reverse()
      return len(l) - 1 - l.index(value)
    except ValueError:
      return len(l) - 1 


class Layer:
  def __init__(self, w, b, **kw):
    self.layer = kw.pop('Layer',0)
    self.func  = kw.pop('Func' ,'tansig')
    del kw
    self.W = np.matrix(w)
    self.b = np.transpose( np.matrix(b) )

  def __call_func(self, Y):
    if self.func == 'tansig':  return self.__tansig(Y)
    if self.func == 'sigmoid': return self.__sigmoid(Y)
 
  def __sigmoid(self, x):
    return (1 / (1 + np.exp(-x)))

  def __tansig(self, x):
    return (2 / (1 + np.exp(-2*x)))-1

  def __call__(self, X):
    B = self.b * np.ones((1, X.shape[1]))
    Y = np.dot(self.W,X)+B
    return self.__call_func(Y)
 
  def get_w_array(self):
    return np.array(np.reshape(self.W, (1,self.W.shape[0]*self.W.shape[1])))[0]
 
  def get_b_array(self):
    return np.array(np.reshape(self.b, (1,self.b.shape[0]*self.b.shape[1])))[0]

  def __str__(self):
    return ('Layer: %d , function: %s, neurons: %d and inputs: %d')%\
           (self.layer,self.func,self.W.shape[0],self.W.shape[1])


class Neural:
  """
    Class Neural will hold the weights and bias information that came
    from tuningtool core format
  """
  def __init__(self, name):
    self._name    = name
    self._nodes   = list()        
    self._layers  = list()
    self._nLayers = 0


  def __call__(self, input):
    '''
      This method can be used like this:
        outputVector = net( inputVector )
      where net is a Neural object intance and outputVector
      is a list with the same length of the input
    '''
    Y = []
    for l in range(len(self._nodes) - 1): 
      if l == 0: Y = self._layers[l](input)
      else: Y = self._layers[l](Y)
    return Y


  def rawDiscrDict(self):
    return {
             'nodes' : self._nodes,
             'weights' : self.get_w_array(),
             'bias' : self.get_b_array(),
           }

  def show(self):
    print  'The Neural configuration:'
    print ('input  layer: %d') % (self._nodes[0])
    print ('hidden layer: %d') % (self._nodes[1])
    print ('output layer: %d') % (self._nodes[2])
    print 'The layers configuration:'
    for layer in range(self._nLayers):
      print self._layers[layer]

  def get_w_array(self):
    w = np.array([])
    for l in range(len(self._nodes) - 1):
      w = np.concatenate((w,self._layers[l].get_w_array()),axis=0)
    return w

  def get_b_array(self):
    b = np.array([])
    for l in range(len(self._nodes) - 1):
      b = np.concatenate((b,self._layers[l].get_b_array()),axis=0)
    return b

  def nodes(self):
    return self._nodes

  def layers(self):
    self._layers

  def set_from_dict(self, rawDict):
    #Retrieve nodes information
    self._nodes = rawDict['nodes']
    self._nLayers = len(self._nodes)-1
    w = rawDict['weights'].tolist()
    b = rawDict['bias'].tolist()
    self._layers=[]
    #Loop over layers
    for layer in range(self._nLayers):
      W=[]; B=[]
      for count in range(self._nodes[layer]*self._nodes[layer+1]):
         W.append(w.pop(0))
      W=np.array(W).reshape(self._nodes[layer+1], self._nodes[layer]).tolist()
      for count in range(self._nodes[layer+1]):
        B.append(b.pop(0))
      self._layers.append( Layer(W,B,Layer=layer) )

  def set_from_fastnet(self, fastnetObj):
    w = []; b = [];
    #Reset layers
    self._layers    = []

    #Retrieve nodes information
    self._nLayers = fastnetObj.getNumLayes()
    for layer in range(self._nLayers): 
      self._nodes.append( fastnetObj.getNumNodes(layer) )
    #Alloc zeros
    for layer in range(len(self._nodes) - 1):
      #func.append( net.getTrfFuncName(layer) )
      w.append( [ [0]*self.nNodes[layer] for d in range( self.nNodes[layer+1])]  )
      b.append( [0]*self.nNodes[layer+1] )
    #Populate our matrix from DiscriminatorpyWrapper
    for layer in range( len(self._nodes) - 1 ):
      for n in range( self._nodes[layer+1] ):
        for k in range( self._nodes[layer] ):
          w[layer][n][k] = fastnetObj.getWeight(layer,n,k)
        b[layer][n] = fastnetObj.getBias(l,n)
      self._layers.append( Layer( w[layer], b[layer], Layer=layer) )

#class Layer:
#  def __init__(self, w, b, **kw):
#    pass
## Just for backward compatibility
#class Neural:
#  def __init__(self, name):
#    pass

NeuralCollection = LimitedTypeList('NeuralCollection',(),{'_acceptedTypes':(Neural,)})

class OldLayer(Logger):
  def __init__(self, w, b, **kw):
    pass

class OldNeural( Logger ):
  def __init__(self, net, **kw):
    Logger.__init__( self, kw )

class PerformancePoint(object):

  def __init__(self, name, sp, pd, pf, thres):
    setattr( self, name + '_sp_value',    sp    )
    setattr( self, name + '_pd_value',    pd    )
    setattr( self, name + '_pf_value',    pf    )
    setattr( self, name + '_thres_value', thres )
    self.name = name

  def __getattr__(self, attr):
    return self.__dict__[self.name + '_' + attr]

class Roc(object):
  """
    Create ROC information holder
  """
  __metaclass__ = RawDictStreamable

  def __init__( self ): 
    pass

  def __call__( self, y_score, y_true = NotSet ):
    """
     output -> The output space generated by the classifier.
     target -> The targets which should be returned by the classifier.
    """
    if y_true is NotSet:
      self.sps        = npCurrent.fp_array( y_score[0] )
      self.pds        = npCurrent.fp_array( y_score[1] )
      self.pfs        = npCurrent.fp_array( y_score[2] )
      self.thresholds = npCurrent.fp_array( y_score[3] )
    else:
      # We have to determine what is signal and noise from the datasets using
      # the targets:
      try:
        from sklearn.metrics import roc_curve
      except ImportError:
        # FIXME Can use previous function that we used here as an alternative
        raise ImportError("sklearn is not available, please install it.")
      self.pfs, self.pds, self.thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=True)
      pds = self.pds
      bps = 1. - self.pfs
      self.sps = np.sqrt( ( pds  + bps )*.5 * np.sqrt( pds * bps ) )

  def retrieve( self, benchmark, extraLabel='' ):
    """
    Retrieve nearest ROC operation to benchmark
    """
    if benchmark.reference is ReferenceBenchmark.SP:
      idx = np.argmax( self.sps )
    else:
      # Get reference for operation:
      if benchmark.reference is ReferenceBenchmark.Pd:
        ref = self.pds
      elif benchmark.reference is ReferenceBenchmark.Pf:
        ref = self.pfs
      delta = ref - benchmark.refVal
      idx   = np.argmin( np.abs( delta ) )
    return PerformancePoint( name=extraLabel + benchmark.name
                           , sp=self.sps[ idx ]
                           , pd=self.pds[ idx ]
                           , pf=self.pfs[idx]
                           , thres=self.thresholds[idx]
                           )
