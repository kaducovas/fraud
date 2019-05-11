__all__ = ['PreProcArchieve', 'PrepObj', 'Projection',  'RemoveMean', 'RingerRp',
           'UndoPreProcError', 'UnitaryRMS', 'FirstNthPatterns', 'KernelPCA',
           'MapStd', 'MapStd_MassInvariant', 'NoPreProc', 'Norm1', 'PCA',
           'PreProcChain', 'PreProcCollection', 'RingerEtaMu', 'RingerFilterMu',
           'StatReductionFactor','StackedAutoEncoder','LSTMAutoEncoder','NLPCA','SAE_FineTuning']

from RingerCore import ( Logger, LoggerStreamable, checkForUnusedVars
                       , save, load, LimitedTypeList, LoggingLevel, LoggerRawDictStreamer
                       , LimitedTypeStreamableList, RawDictStreamer, RawDictCnv )
#from RingerCore import LimitedTypeListRDC, LoggerLimitedTypeListRDS, \
#                       LimitedTypeListRDS
from TuningTools.coreDef import npCurrent
import numpy as np


import os
import sys
import pickle
import time

import sklearn.metrics
from sklearn.externals import joblib

#print sys.path
#sys.path.remove('/scratch/22061a/common-cern/pyenv/versions/2.7.9/lib/python2.7/site-packages')
#import tensorflow
import keras
from keras.utils import np_utils
from keras.models import load_model

print 'PreProc '+keras.__version__
#import sklearn.metrics
#from sklearn.externals import joblib

from TuningTools import SAE_TrainParameters as trnparams
from TuningTools.StackedAutoEncoders import StackedAutoEncoders
from TuningTools.nlpca import NonLinPCA

import multiprocessing


class PreProcArchieve( Logger ):
  """
  Context manager for Pre-Processing archives

  Version 3: - saves raw version of pp collection
  Version 2: - saved a pp collection for each eta/et bin and each sort.
  Version 1: - saved a pre-processing chain.
  """

  _type = 'PreProcFile'
  _version = 3

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with PreProcArchieve("/path/to/file") as data:
      BLOCK

    PreProcArchieve( "file/path", ppCol = Norm1() )
    """
    Logger.__init__(self, kw)
    self._filePath = filePath
    self._ppCol = kw.pop( 'ppCol', None )
    checkForUnusedVars( kw, self._warning )

  @property
  def filePath( self ):
    return self._filePath

  def filePath( self, val ):
    self._filePath = val

  @property
  def ppCol( self ):
    return self._ppCol

  def getData( self ):
    if not self._ppCol:
       self._fatal("Attempted to retrieve empty data from PreProcArchieve.")
    return {'type' : self._type,
            'version' : self._version,
            'ppCol' : self._ppCol.toRawObj() }

  def save(self, compress = True):
    return save( self.getData(), self._filePath, compress = compress )

  def __enter__(self):
    from cPickle import PickleError
    try:
      ppColInfo = load( self._filePath )
    except PickleError:
      # It failed without renaming the module, retry renaming old module
      # structure to new one.
      import sys
      sys.modules['FastNetTool.PreProc'] = sys.modules[__name__]
      ppColInfo = load( self._filePath )
    try:
      if ppColInfo['type'] != self._type:
        self._fatal(("Input crossValid file is not from PreProcFile "
            "type."))
      if ppColInfo['version'] == 3:
        ppCol = PreProcCollection.fromRawObj( ppColInfo['ppCol'] )
      elif ppColInfo['version'] == 2:
        ppCol = ppColInfo['ppCol']
      elif ppColInfo['version'] == 1:
        ppCol = PreProcCollection( ppColInfo['ppCol'] )
      else:
        self._fatal("Unknown job configuration version.")
    except RuntimeError, e:
      self._fatal(("Couldn't read PreProcArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    return ppCol

  def __exit__(self, exc_type, exc_value, traceback):
    # Remove bound
    self.ppChain = None

class UndoPreProcError(RuntimeError):
  """
    Raised when it is not possible to undo pre-processing.
  """
  def __init__( self, *args ):
    RuntimeError.__init__( self, *args )

# TODO List:
#
# - Add remove_constant_rows
# - Check for Inf, NaNs and so on

class PrepObj( LoggerStreamable ):
  """
    This is the base class of all pre-processing objects.
  """

  def __init__(self, d = {}, **kw):
    d.update( kw )
    LoggerStreamable.__init__(self, d)

  def __call__(self, data, revert = False):
    """
      The revert should be used to undo the pre-processing.
    """
    if revert:
      try:
        self._debug('Reverting %s...', self.__class__.__name__)
        data = self._undo(data)
      except AttributeError:
        self._fatal("It is impossible to revert PreProc ")#%s" % \
        #    self.__class__.___name__)
    else:
      self._debug('Applying %s...', self.__class__.__name__)
      data = self._apply(data)
    return data

  def takeParams(self, data):
    """
      Calculate pre-processing parameters.
    """
    self._debug("No need to retrieve any parameters from data.")
    return self._apply(data)

  def release(self):
    """
      Release calculated pre-proessing parameters.
    """
    self._debug(("No parameters were taken from data, therefore none was "
        "also empty."))

  # TODO: Do something smart here, this is needed at the moment.
  def concatenate(self, data, extra):
    """
      Concatenate extra patterns if needed
    """
    return data

  def __str__(self):
    """
      Overload this method to return the string representation
      of the pre-processing.
    """
    pass

  def shortName(self):
    """
      Overload this method to return the short string representation
      of the pre-processing.
    """
    pass

  def isRevertible(self):
    """
      Check whether the PreProc is revertible
    """
    import inspect
    return any([a[0] == '_undo' for a in inspect.getmembers(self) ])

  def _apply(self, data):
    """
      Overload this method to apply the pre-processing
    """
    return data



class NoPreProc(PrepObj):
  """
    Do not apply any pre-processing to data.
  """

  def __init__( self, **kw ):
    PrepObj.__init__( self, kw )
    checkForUnusedVars(kw, self._warning )
    del kw

  def __str__(self):
    """
      String representation of the object.
    """
    return "NoPreProc"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "NoPP"

  def _undo(self, data):
    return data


class Projection(PrepObj):
  """
    Project data into new base
  """

  # FIXME: This will probably give problematic results if data is saved
  # with one numpy type and executed with other type

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mat'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mat'})

  def __init__( self, **kw ):
    PrepObj.__init__( self, kw )
    self._mat = kw.pop( 'matrix', npCurrent.fp_array([[]]) )
    checkForUnusedVars(kw, self._warning )
    del kw

  def __str__(self):
    """
      String representation of the object.
    """
    return "Proj"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "P"

  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cData in data:
        if npCurrent.useFortran:
          ret.append( np.dot( self._mat, cData ) )
        else:
          ret.append( np.dot( cData, self._mat ) )
    else:
      if npCurrent.useFortran:
        ret = np.dot( self._mat , data )
      else:
        ret = np.dot( data , self._mat )
    return ret

  def takeParams(self, trnData, **kw):
    return self._apply(trnData)

class RemoveMean( PrepObj ):
  """
    Remove data mean
  """
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mean'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mean'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._mean = np.array( [], dtype=npCurrent.dtype )

  @property
  def mean(self):
    return self._mean

  @property
  def params(self):
    return self.mean()

  def takeParams(self, trnData):
    """
      Calculate mean for transformation.
    """
    import copy
    data = copy.deepcopy(trnData)
    if isinstance(trnData, (tuple, list,)):
      data = np.concatenate( trnData, axis=npCurrent.odim )
    self._mean = np.mean( data, axis=npCurrent.odim, dtype=data.dtype ).reshape(
            npCurrent.access( pidx=data.shape[npCurrent.pdim],
                              oidx=1 ) )
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "rm_mean"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "no_mu"

  def _apply(self, data):
    if not self._mean.size:
      self._fatal("Attempted to apply RemoveMean before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( cdata - self._mean )
    else:
      ret = data - self._mean
    return ret

  def _undo(self, data):
    if not self._mean.size:
      self._fatal("Attempted to undo RemoveMean before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata + self._mean )
    else:
      ret = data + self._mean
    return ret

class UnitaryRMS( PrepObj ):
  """
    Set unitary RMS.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_invRMS'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_invRMS'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._invRMS  = np.array( [], dtype=npCurrent.dtype )

  @property
  def rms(self):
    return 1 / self._invRMS

  @property
  def params(self):
    return self.rms()

  def takeParams(self, trnData):
    """
      Calculate rms for transformation.
    """
    # Put all classes information into only one representation
    # TODO Make transformation invariant to each class mass.
    import copy
    data = copy.deepcopy(trnData)
    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    tmpArray = np.sqrt( np.mean( np.square( data ), axis=npCurrent.odim ) ).reshape(
                npCurrent.access( pidx=data.shape[npCurrent.pdim],
                                  oidx=1 ) )
    tmpArray[tmpArray==0] = 1
    self._invRMS = 1 / tmpArray
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "UnitRMS"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "1rms"

  def _apply(self, data):
    if not self._invRMS.size:
      self._fatal("Attempted to apply UnitaryRMS before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( cdata * self._invRMS )
    else:
      ret = ( data * self._invRMS )
    return ret

  def _undo(self, data):
    if not self._invRMS.size:
      self._fatal("Attempted to undo UnitaryRMS before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata / self._invRMS )
    else:
      ret = ( data / self._invRMS )
    return ret

class TrackSimpleNorm( PrepObj ):
  """
    Specific normalization for track parameters in mc14, 13TeV.
    Six variables, normalized through simple multiplications.
  """
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_factors'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_factors'})
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__(self, d)
    checkForUnusedVars(d, self._warning)
    self._factors = [0.05,  # deltaeta1
                     1.0,   # deltaPoverP
                     0.05,  # deltaPhiReescaled
                     6.0,   # d0significance
                     0.2,   # d0pvunbiased
                     1.0  ] # TRT_PID (from eProbabilityHT)
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "Tracking data simple normalization."

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "TrackSimple"

  def _apply(self, data):
    # NOTE: is it a problem that I don't deepcopy the data?
    if isinstance(data, (list,tuple)):
      if len(data) == 0: return data
      import numpy as np
      import copy
      ret = []
      for conj in data:
        tmp = copy.deepcopy(conj)
        for i in range(len(self._factors)):
          tmp[:,i] = tmp[:,i]/self._factors[i]
        ret.append(tmp)
      return ret
    else:
      self._fatal("Data is not in the right format, must be list or tuple")

  def takeParams(self, trnData):
    return self._apply(trnData)


class Norm1(PrepObj):
  """
    Applies norm-1 to data
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    #self._normslist = ''
    #self._beforenorm = ''
    #self._afternorm = ''
    del d

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    # if isinstance(data, (tuple, list,)):
    #   norms = []
    #   for cdata in data:
    #     cnorm = cdata.sum(axis=npCurrent.pdim).reshape(
    #         npCurrent.access( pidx=1,
    #                           oidx=cdata.shape[npCurrent.odim] ) )
    #     cnorm[cnorm==0] = 1
    #     norms.append( cnorm )
    # else:
    #   norms = data.sum(axis=npCurrent.pdim).reshape(
    #         npCurrent.access( pidx=1,
    #                           oidx=data.shape[npCurrent.odim] ) )
    #   norms[norms==0] = 1
    # #self._norms = norms
    # return norms

    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = np.abs( cdata.sum(axis=npCurrent.pdim).reshape(
            npCurrent.access( pidx=1,
                              oidx=cdata.shape[npCurrent.odim] ) ) )
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = np.abs( data.sum(axis=npCurrent.pdim).reshape(
            npCurrent.access( pidx=1,
                              oidx=data.shape[npCurrent.odim] ) ) )
      norms[norms==0] = 1
    #self._norms = norms
    return norms

  def __str__(self):
    """
      String representation of the object.
    """
    return "Norm1"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "N1"

  def _apply(self, data):
    norms = self.__retrieveNorm(data)
    #self._info("norma1 list")
    #self._info(self.__retrieveNorm(data))
    self._beforenorm = data
    self._normslist = self.__retrieveNorm(data)
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata / norms[i] )
    else:
      ret = data / norms
    self._afternorm=ret #.append(ret)
    return ret

class ExpertNetworksSimpleNorm(PrepObj):
  """
    Specific normalization for calorimeter and tracking parameters
    to be used in the Expert Neural Networks training.
    Usage of Norm1 normalization for calorimeter data and
    TrackSimpleNorm for tracking data.
  """
  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_factors'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_factors'})
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__(self, d)
    checkForUnusedVars(d, self._warning)
    self._factors = [0.05,  # deltaeta1
                     1.0,   # deltaPoverP
                     0.05,  # deltaPhiReescaled
                     6.0,   # d0significance
                     0.2,   # d0pvunbiased
                     1.0  ] # eProbabilityHT
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "Expert Neural Networks simple normalization."

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "Exp1"

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters of Norm1 normalization.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = cdata.sum(axis=npCurrent.pdim).reshape(
            npCurrent.access( pidx=1,
                              oidx=cdata.shape[npCurrent.odim] ) )
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = data.sum(axis=npCurrent.pdim).reshape(
            npCurrent.access( pidx=1,
                              oidx=data.shape[npCurrent.odim] ) )
      norms[norms==0] = 1
    return norms

  def _apply(self, data):
    # Take care of different number of samples:
    for i in xrange(len(data[0])):
      if data[1][i].shape[ npCurrent.odim ] != data[0][i].shape[ npCurrent.odim ]:
        self._fatal("Data dimensions are not the same! Make sure to extract using createData from the same ntuples!")
    data_calo = data[0]

    norms = self.__retrieveNorm(data_calo)
    if isinstance(data_calo, (tuple, list,)):
      ret_calo = []
      for i, cdata in enumerate(data_calo):
        ret_calo.append( cdata / norms[i] )
    else:
      ret_calo = data_calo / norms

    data_track = data[1]
    if not isinstance(data_track, (list,tuple)):
      self._fatal("Data is not in the right format, must be list or tuple")
    else:
      if len(data_track) == 0:
        ret_track = data_track
      else:
        import numpy as np
        import copy
        ret_track = []
        for conj in data_track:
          tmp = copy.deepcopy(conj)
          for i in range(len(self._factors)):
            tmp[:,i] = tmp[:,i]/self._factors[i]
          ret_track.append(tmp)
    return [ret_calo,ret_track]

class _ExpertCaloNetworksNormRDS( LoggerRawDictStreamer ):
  def treatDict(self, obj, raw):
    """
    Add efficiency value to be readable in matlab
    """
    raw['_mean'] = obj._mapStd.mean
    raw['_invRMS'] = obj._mapStd.invRMS
    return RawDictStreamer.treatDict( self, obj, raw )

class _ExpertCaloNetworksNormRDC( RawDictCnv ):
  def treatObj( self, obj, d ):
    obj._mapStd._mean = d['_mean']
    obj._mapStd._invRMS = d['_invRMS']
    return obj

class ExpertNorm1Std(PrepObj):
  """
    Applies norm1 to first dataset and MapStd to second one.
  """
  _streamerObj = _ExpertCaloNetworksNormRDS()
  _cnvObj = _ExpertCaloNetworksNormRDC()

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__(self, d)
    checkForUnusedVars(d, self._warning)
    del d
    # Rings normalization
    self._norm1 = Norm1()
    # Standard quantities normalization
    self._mapStd = MapStd()

  def __str__(self):
    """
      String representation of the object.
    """
    return "ExpertCaloNorm"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "ExpStdN1"

  def takeParams(self, data):
    """
      Calculate pre-processing parameters.
    """
    # Take care of different number of samples:
    for i in xrange(len(data[0])):
      if data[1][i].shape[ npCurrent.odim ] != data[0][i].shape[ npCurrent.odim ]:
        self._fatal("Data dimensions are not the same! Make sure to extract using createData from the same ntuples!")
    self._debug("No need to retrieve any parameters from data.")
    self._mapStd.takeParams( data[1] )
    self._norm1.takeParams( data[0] )
    return self._apply(data)

  def _apply(self, data):
    retNorm1 = self._norm1._apply( data[0] )
    retMapStd = self._mapStd._apply( data[1] )
    return [retNorm1,retMapStd]

class FirstNthPatterns(PrepObj):
  """
    Get first nth patterns from data
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_n'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_n'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self._n = d.pop('nth',0)
    checkForUnusedVars(d, self._warning )
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "First_%dPat" % self._n

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "F%dP" % self._n

  def _apply(self, data):
    try:
      if isinstance(data, (tuple, list,)):
        ret = []
        for cdata in data:
          ret.append( cdata[npCurrent.access( pidx=slice(0,self._n), oidx=':'  ) ] )
      else:
        ret = data[ npCurrent.access( pidx=slice(0,self._n), oidx=':'  ) ]
    except IndexError, e:
      self._fatal("Data has not enought patterns!\n%s", str(e), IndexError)
    return ret

class RingerRp( Norm1 ):
  """
    Apply ringer-rp reprocessing to data.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_alpha', '_beta','_rVec'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_alpha','_beta','_rVec'})

  def __init__(self, alpha = 1., beta = 1., d = {}, **kw):
    d.update( kw ); del kw
    Norm1.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._alpha = alpha
    self._beta = beta
    #Layers resolution
    #PS      = 0.025 * np.arange(1,8)
    #EM1     = 0.003125 * np.arange(1,64)
    #EM2     = 0.025 * np.arange(1,8)
    #EM3     = 0.05 * np.arange(1,8)
    #HAD1    = 0.1 * np.arange(1,4)
    #HAD2    = 0.1 * np.arange(1,4)
    #HAD3    = 0.2 * np.arange(1,4)
    PS      = 0.025 * np.arange(8)
    EM1     = 0.003125 * np.arange(64)
    EM2     = 0.025 * np.arange(8)
    EM3     = 0.05 * np.arange(8)
    HAD1    = 0.1 * np.arange(4)
    HAD2    = 0.1 * np.arange(4)
    HAD3    = 0.2 * np.arange(4)
    rings   = np.concatenate((PS,EM1,EM2,EM3,HAD1,HAD2,HAD3))
    self._rVec = np.power( rings, self._beta )

  def __str__(self):
    """
      String representation of the object.
    """
    return ("RingerRp_a%g_b%g" % (self._alpha, self._beta)).replace('.','dot')

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "Rp"

  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    return Norm1._Norm1__retrieveNorm(self, data)

  def rVec(self):
    """
      Retrieves the ring pseudo-distance vector
    """
    return self._rVec

  def _apply(self, data):
    self._info('(alpha, beta) = (%f,%f)', self._alpha, self._beta)
    mask = np.ones(100, dtype=bool)
    mask[np.cumsum([0,8,64,8,8,4,4])] = False
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        rpEnergy = np.sign(cdata)*np.power( abs(cdata), self._alpha )
        #rpEnergy = rpEnergy[ npCurrent.access( pidx=mask, oidx=':') ]
        norms = self.__retrieveNorm(rpEnergy)
        rpRings = ( ( rpEnergy * self._rVec ) / norms[i][ npCurrent.access( pidx=':', oidx=np.newaxis) ] ).astype( npCurrent.fp_dtype )
        ret.append(rpRings)
    else:
      rpEnergy = np.sign(data)*np.power( abs(cdata), self._alpha )
      #rpEnergy = rpEnergy[ npCurrent.access( pidx=mask, oidx=':') ]
      norms = self.__retrieveNorm(rpEnergy)
      rpRings = ( ( rpEnergy * self._rVec ) / norms[i][ npCurrent.access( pidx=':', oidx=np.newaxis) ] ).astype( npCurrent.fp_dtype )
      ret = rpRings
    return ret

class MapStd( PrepObj ):
  """
    Remove data mean and set unitary standard deviation.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mean', '_invRMS'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mean','_invRMS'})

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    del d
    self._mean = np.array( [], dtype=npCurrent.dtype )
    self._invRMS  = np.array( [], dtype=npCurrent.dtype )

  @property
  def mean(self):
    return self._mean

  @property
  def rms(self):
    return 1 / self._invRMS

  @property
  def invRMS(self):
    return self._invRMS

  def params(self):
    return self.mean(), self.rms()

  def takeParams(self, trnData):
    """
      Calculate mean and rms for transformation.
    """
    # Put all classes information into only one representation
    # TODO Make transformation invariant to each class mass.
    import copy
    data = copy.deepcopy(trnData)
    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    self._mean = np.mean( data, axis=npCurrent.odim, dtype=data.dtype ).reshape(
            npCurrent.access( pidx=data.shape[npCurrent.pdim],
                              oidx=1 ) )
    data = data - self._mean
    tmpArray = np.sqrt( np.mean( np.square( data ), axis=npCurrent.odim ) ).reshape(
                npCurrent.access( pidx=data.shape[npCurrent.pdim],
                                  oidx=1 ) )
    tmpArray[tmpArray==0] = 1
    self._invRMS = 1 / tmpArray
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "MapStd"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "std"

  def _apply(self, data):
    self._beforenorm = data
    if not self._mean.size or not self._invRMS.size:
      self._fatal("Attempted to apply MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        ret.append( ( cdata - self._mean ) * self._invRMS )
    else:
      ret = ( data - self._mean ) * self._invRMS
    self._afternorm = ret
    return ret

  def _undo(self, data):
    if not self._mean.size or not self._invRMS.size:
      self._fatal("Attempted to undo MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( ( cdata / self._invRMS ) + self._mean )
    else:
      ret = ( data / self._invRMS ) + self._mean
    return ret

class StackedAutoEncoder( PrepObj ):
  """
    Train the encoders in order to stack them as pre-processing afterwards.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {}, transientAttrs = {'_SAE',})
  _cnvObj = RawDictCnv(toProtectedAttrs = {})


  def __init__(self,n_inits=1,hidden_activation='tanh',output_activation='linear',n_epochs=5000,patience=30,batch_size=200,layer=1, d = {}, **kw):
    d.update( kw ); del kw
    from RingerCore import retrieve_kw
    self._hidden_neurons = retrieve_kw(d,'hidden_neurons',[80])
    self._caltype = retrieve_kw(d,'caltype','allcalo')
    self._aetype = retrieve_kw(d,'aetype','vanilla') #VANILLA, SPARSE, DENOISING, CONTRACTIVE
    self._dataEncoded = retrieve_kw(d,'dataEncoded','all') #all, signal, background
    self._nIterations = retrieve_kw(d,'nIterations',9) #all, signal, background
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    self._n_inits = n_inits
    self._hidden_activation = hidden_activation
    self._output_activation = output_activation
    self._n_epochs = n_epochs
    self._patience = patience
    self._batch_size = batch_size
    self._layer= layer
    del d
    self._sort = ''
    self._etBinIdx = ''
    self._etaBinIdx = ''
    self._SAE = ''
    self._trn_params = ''
    self._trn_desc = ''
    self._weights = ''


    #self._mean = np.array( [], dtype=npCurrent.dtype )
    #self._invRMS  = np.array( [], dtype=npCurrent.dtype )

    ####self.paramametros de nosso interesse

  #def __call__(self, data, revert = False,sort,etBinIdx,etaBinIdx):
  #  """
  #    The revert should be used to undo the pre-processing.
  #  """
  #  if revert:
  #    try:
  #      self._debug('Reverting %s...', self.__class__.__name__)
  #      data = self._undo(data)
  #    except AttributeError:
  #      self._fatal("It is impossible to revert PreProc ")#%s" % \
  #      #    self.__class__.___name__)
  #  else:
  #    self._info('SAE Applying %s...', self.__class__.__name__)
  #    data = self._apply(data,sort,etBinIdx,etaBinIdx)
  #  return data

  #def SAE(self):
  #  return self._SAE

  #def trn_desc(self):
  #  return self._trn_desc

  #def weights(self):
  #  return self._weights

  #def trn_params(self):
  #  return self._trn_params

  #def params(self):
  #  return self.SAE(), self.trn_params(),self.trn_desc(), self.weights()

  def takeParams(self, trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder,layerNumber):

  ###trainlayer

    """
      Perform the layerwise algorithm to train the SAE
    """

    # TODO...
    self._sort = sort
    self._etBinIdx = etBinIdx
    self._etaBinIdx = etaBinIdx
    print(self._caltype)

    import copy
    data = copy.deepcopy(trnData)
    val_Data = copy.deepcopy(valData)

    if self._caltype == 'emcalo' and data[0].shape[1] == 100:
      print 'EMMMMMMMMMMM'
      data = [d[:,:88] for d in data]
      val_Data = [d[:,:88] for d in val_Data]
    elif self._caltype == 'hadcalo' and data[0].shape[1] == 100:
      print 'HAAAAAAAAAAAD'
      data = [d[:,88:] for d in data]
      val_Data = [d[:,88:] for d in val_Data]

    ###teste AE para EM1
    em1ae=False
    if em1ae:
      data = [d[:,14:72] for d in data]
      val_Data = [d[:,14:72] for d in val_Data]


    self._info('Training Data Shape: '+str(data[0].shape)+str(data[1].shape))
    self._info('Validation Data Shape: '+str(val_Data[0].shape)+str(val_Data[1].shape))

    #data = [d[:100] for d in data]
    #val_Data = [d[:100] for d in val_Data]

    #print "TESTEEEE"+tuning_folder

    self._batch_size = min(data[0].shape[0],data[1].shape[0])

    from TuningTools.discriminant_autoencoder import *
    class_sgn = [1]*data[0].shape[0]
    class_bkg = [-1]*data[1].shape[0]
    train_target = np.concatenate((class_sgn, class_bkg))

    if self._dataEncoded == 'all':
      if isinstance(data, (tuple, list,)):
        data = np.concatenate( data, axis=npCurrent.odim )
      if isinstance(val_Data, (tuple, list,)):
        val_Data = np.concatenate( val_Data, axis=npCurrent.odim )
    elif self._dataEncoded == 'signal':
      if isinstance(data, (tuple, list,)):
        data = data[0]
      if isinstance(val_Data, (tuple, list,)):
        val_Data = val_Data[0]
    elif self._dataEncoded == 'background':
      if isinstance(data, (tuple, list,)):
        data = data[1]
      if isinstance(val_Data, (tuple, list,)):
        val_Data = val_Data[1]

    if self._aetype == 'discriminant':
      transformed_data = transform_data(data,train_target,nIteration=self._nIterations)


    import numpy
    work_path='//home/caducovas/run/' #'/scratch/22061a/caducovas/run/'
    results_path = work_path+"StackedAutoEncoder_preproc/"
    numpy.save(results_path+'val_Data_sort_'+str(self._sort)+'_hidden_neurons_'+str(self._hidden_neurons[0]),val_Data)
    trn_params_folder = results_path+'trnparams_sort_'+str(self._sort)+'_hidden_neurons_'+str(self._hidden_neurons[0])+'.jbl'

    #Autoencoder Types paraemters and definitions
    if self._aetype == 'sparse':
      regularizer='l1'
      regularizer_param=10e-5
    else:
      #regularizer='dropout'
      #regularizer_param=0.5
      regularizer=None
      regularizer_param=None

    ###if self._aetype == 'contractive':
    ###  from TuningTools.MetricsLosses import contractive_loss
    ####  loss=contractive_loss
    ####else:
    ####  loss='mean_squared_error'

    if os.path.exists(trn_params_folder):
        os.remove(trn_params_folder)
    if not os.path.exists(trn_params_folder):
        trn_params = trnparams.NeuralClassificationTrnParams(n_inits=self._n_inits,
                                                             hidden_activation=self._hidden_activation,
                                                             output_activation=self._output_activation,
                                                             n_epochs=self._n_epochs,
                                                             patience=self._patience,
                                                             batch_size=self._batch_size) #,
                                                             #loss=loss)
    trn_params.save(trn_params_folder)

    self._trn_params = trn_params

    self._info(trn_params.get_params_str())



    # Train Process
    SAE = StackedAutoEncoders(params = trn_params,
                              development_flag = False,
                              n_folds = 1,
                              save_path = results_path,
                              prefix_str=self._caltype,
                              aetype=self._aetype,
                              dataEncoded=self._dataEncoded,
                              nIteration=self._nIterations,
                              layerNumber=layerNumber
                              )

    self._SAE = SAE

    # Choose layer to be trained
    layer = self._layer

    #self._info(self._hidden_neurons)
    if self._aetype == 'discriminant':
      f, model, trn_desc = SAE.trainLayer(data=data,
                                          transformed_data=transformed_data,
                                          ifold=0,
                                          hidden_neurons=self._hidden_neurons,
                                          layer = self._layer,sort=sort,etBinIdx=etBinIdx, etaBinIdx=etaBinIdx, tuning_folder = tuning_folder,regularizer=regularizer,regularizer_param=regularizer_param)
    else:
      f, model, trn_desc = SAE.trainLayer(data=data,
                                          trgt=val_Data,
                                          ifold=0,
                                          hidden_neurons=self._hidden_neurons,
                                          layer = self._layer,sort=sort,etBinIdx=etBinIdx, etaBinIdx=etaBinIdx, tuning_folder = tuning_folder,regularizer=regularizer,regularizer_param=regularizer_param)

    self._trn_desc = trn_desc
    self._weights = model.get_weights()
    self._trn_params = model.get_config()
    #self._info(self._SAE)

    return self._apply(trnData,layerNumber)



  def __str__(self):
    """
      String representation of the object.
    """
    if self._caltype == 'emcalo':
      #print 'EMMMMMMMMMMM'
      sname ='EMAutoencoder'

    elif self._caltype == 'hadcalo':
      sname='HADAutoencoder'
    else:
      sname='Autoencoder'

    if self._aetype == 'vanilla':
      aetypename = ''
    elif self._aetype == 'sparse':
      aetypename = 'Sparse'
    elif self._aetype == 'denoising':
      aetypename = 'Denoising'
    elif self._aetype == 'contractive':
      aetypename = 'Contractive'
    elif self._aetype == 'discriminant':
      aetypename = 'Discriminant'+str(self._nIterations)+'Iter'

    if self._dataEncoded == 'all':
      encodetypename=''
    elif self._dataEncoded == 'signal':
      encodetypename='Signal'
    elif self._dataEncoded == 'background':
      encodetypename='Background'
    return (encodetypename+aetypename+sname+"_%d" % self._hidden_neurons[0])

  def shortName(self):
    """
      Short string representation of the object.
    """
    if self._caltype == 'emcalo':
      #print 'EMMMMMMMMMMM'
      sname ='EMAE'

    elif self._caltype == 'hadcalo':
      sname='HADAE'
    else:
      sname='AE'

    if self._aetype == 'vanilla':
      aetypename = ''
    elif self._aetype == 'sparse':
      aetypename = 'S'
    elif self._aetype == 'denoising':
      aetypename = 'D'
    elif self._aetype == 'contractive':
      aetypename = 'C'
    elif self._aetype == 'discriminant':
      aetypename = 'Disc'+str(self._nIterations)+'Iter'

    if self._dataEncoded == 'all':
      encodetypename=''
    elif self._dataEncoded == 'signal':
      encodetypename='Sgn'
    elif self._dataEncoded == 'background':
      encodetypename='Bkg'
    return (encodetypename+aetypename+sname+"_%d" % self._hidden_neurons[0])

  def _apply(self, data,layerNumber):
    #self._info(pp.shortName())
    #self._info(self._etBinIdx)
    #self._info(self._etaBinIdx)
    ###get data projection
    #if not self._mean.size or not self._invRMS.size:
    #  self._fatal("Attempted to apply MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      if self._caltype == 'emcalo' and data[0].shape[1] == 100:
        #print 'EMMMMMMMMMMM'
        data = [d[:,:88] for d in data]
        #val_Data = [d[:,:88] for d in val_Data]
      elif self._caltype == 'hadcalo' and data[0].shape[1] == 100:
        #print 'HAAAAAAAAAAAD'
        data = [d[:,88:] for d in data]
        #val_Data = [d[:,88:] for d in val_Data]
      #data = [d[:100] for d in data]
      em1ae=False
      if em1ae:
        data = [d[:,14:72] for d in data]
      for cdata in data:
	#self._info(cdata.shape)
        ret.append(self._SAE.getDataProjection(cdata, cdata, hidden_neurons=self._hidden_neurons, layer=self._layer, ifold=0,sort=self._sort,etBinIdx=self._etBinIdx,etaBinIdx=self._etaBinIdx,))
    else:
      ret = self._SAE.getDataProjection(cdata, cdata, hidden_neurons=self._hidden_neurons, layer=self._layer, ifold=0,sort=self._sort,etBinIdx=self._etBinIdx,etaBinIdx=self._etaBinIdx)
    return ret

  # def _undo(self, data):
    # if not self._mean.size or not self._invRMS.size:
      # self._fatal("Attempted to undo MapStd before taking its parameters.")
    # if isinstance(data, (tuple, list,)):
      # ret = []
      # for i, cdata in enumerate(data):
        # ret.append( ( cdata / self._invRMS ) + self._mean )
    # else:
      # ret = ( data / self._invRMS ) + self._mean
    #
    return ret

class LSTMAutoEncoder( PrepObj ):
  """
    Train the encoders in order to stack them as pre-processing afterwards.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {}, transientAttrs = {'_LSTMAE',})
  _cnvObj = RawDictCnv(toProtectedAttrs = {})


  def __init__(self,n_inits=1,units=16,bidirection=False,layers=1, hidden_neurons=180,model_name="ringer_N1_et1_eta1",global_step=None,batch_size=1000,layer=1, d = {}, **kw):
    d.update( kw ); del kw
    from RingerCore import retrieve_kw
    from audeep.backend.training.base import BaseFeatureLearningWrapper
    from audeep.backend.training.time_autoencoder import TimeAutoencoderWrapper
    from audeep.backend.models.rnn_base import CellType, RNNArchitecture
    from audeep.backend.models.rnn_base import CellType, RNNArchitecture
    #import numpy as np
    from pathlib import Path

    # self._hidden_neurons = retrieve_kw(d,'hidden_neurons',[80])
    # self._caltype = retrieve_kw(d,'caltype','allcalo')
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    #self._model_name=model_name,
    #self._model_filename='/home/users/caducovas/deep_output/ringer_1_1_24_epochs_2_layer_90_units/logs/model'
    #print self._model_filename
    #self._model_filename=Path('/home/users/caducovas/output/'+str(self._model_name)+'/t-1x256-x-b/logs/model'),
    self._global_step=global_step,
    #self._data_set=input_data,
    self._batch_size=batch_size

    self._num_layers = layers
    self._num_units = units
    self._bidirectional = bidirection
    self._cell_type = CellType.GRU
    self._num_epochs = 64
    self._mask_silence = False
    self._trn_batch_size = 256
    self._checkpoints_to_keep = None
    self._learning_rate = 0.001
    self._keep_prob = 0.5
    self._encoder_noise = 0.0
    self._feed_previous_prob = 0.0


    if self._bidirectional:
      brnn='_bidirectional_'
      bname='B'
    else:
      brnn=''
      bname=''

    if self._cell_type == CellType.GRU:
      c_type='_GRU'
    else:
      c_type='_LSTM'

    self._model_shortname=bname+c_type.replace('_','')+'_'+str(self._num_layers)+'L'

    self._save_path='/home/users/caducovas/lstm_output/'

    self._model_name= str(self._num_epochs)+'_epochs_'+str(self._num_layers)+'_layers_'+str(self._num_units)+'_units_'+brnn+c_type
    #self._model_filename = save_path+model_name / "logs" / "model"

    self._record_path = '/scratch/22061a/caducovas/run/tfr/'
    self._record_files = []

    architecture = RNNArchitecture(num_layers=self._num_layers,
                                   num_units=self._num_units,
                                   bidirectional=self._bidirectional,
                                   cell_type=self._cell_type)
    self._hidden_neurons  = architecture.state_size

    # self._n_inits = n_inits
    # self._hidden_activation = hidden_activation
    # self._output_activation = output_activation
    # self._n_epochs = n_epochs
    # self._patience = patience
    # self._batch_size = batch_size
    # self._layer= layer
    del d
    self._sort = ''
    self._etBinIdx = ''
    self._etaBinIdx = ''
    # self._SAE = ''
    # self._trn_params = ''
    # self._trn_desc = ''
    # self._weights = ''


  def takeParams(self, trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder,layerNumber):

  ###trainlayer

    """
      Perform the layerwise algorithm to train the SAE
    """
    import abc
    import shutil
    import tempfile
    from pathlib import Path
    from cliff.command import Command
    from audeep.backend.data.export import export_ringer_tfrecords
    from audeep.backend.models.rnn_base import CellType, RNNArchitecture
    from RingerCore import retrieve_kw
    from audeep.backend.training.base import BaseFeatureLearningWrapper
    from audeep.backend.training.time_autoencoder import TimeAutoencoderWrapper
    import tensorflow as tf
    from audeep.backend.data.records import to_tensor
    import os
    #from audeep.backend.decorators import to_tensor
    import copy
    import subprocess
    data = copy.deepcopy(trnData)
    val_Data = copy.deepcopy(valData)

    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    if isinstance(val_Data, (tuple, list,)):
      val_Data = np.concatenate( val_Data, axis=npCurrent.odim )

    self._feature_shape = [data.shape[1],1]
    self._num_instances = data.shape[0]
    self._record_files = [self._record_path+'ringer_tfrecords_et_'+str(etBinIdx)+'_eta_'+str(etaBinIdx)+'_sort_'+str(sort)]
    model_filename = self._model_name+'_et_'+str(etBinIdx)+'_eta_'+str(etaBinIdx)+'_sort_'+str(sort)
    self._model_filename = self._save_path+'/et_'+str(etBinIdx)+'_eta_'+str(etaBinIdx)+'_sort_'+str(sort)+'/'+model_filename+'/logs/model'

    # # TODO...
    self._sort = sort
    self._etBinIdx = etBinIdx
    self._etaBinIdx = etaBinIdx
    # print(self._caltype)

    if not os.path.exists(self._model_filename ):
      os.makedirs(self._model_filename )

    # if self._caltype == 'emcalo' and data[0].shape[1] == 100:
      # print 'EMMMMMMMMMMM'
      # data = [d[:,:88] for d in data]
      # val_Data = [d[:,:88] for d in val_Data]
    # elif self._caltype == 'hadcalo' and data[0].shape[1] == 100:
      # print 'HAAAAAAAAAAAD'
      # data = [d[:,88:] for d in data]
      # val_Data = [d[:,88:] for d in val_Data]

    self._info('Training Data Shape: '+str(data[0].shape)+str(data[1].shape))
    self._info('Validation Data Shape: '+str(val_Data[0].shape)+str(val_Data[1].shape))

    #applyOnly = True
    #if applyOnly:
    #  return self._apply(trnData)
    export_ringer_tfrecords(self._record_files[0],data)

    # filename_tensor = tf.constant(value=[str(file) for file in self._record_files], dtype=tf.string,
                                  # name="filenames")
    # filename_queue = tf.train.string_input_producer(filename_tensor, num_epochs=None)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # tensor_data = to_tensor(serialized_example, self._feature_shape)
    # data_batches = tf.train.batch([tensor_data],
                                  # enqueue_many=False,
                                  # batch_size=self._trn_batch_size,
                                  # num_threads=2,
                                  # capacity=100 * self._trn_batch_size,
                                  # #min_after_dequeue=50 * self._batch_size,
                                  # allow_smaller_final_batch=True)

    # # batches are batch major; also we assume that features are stored as [time, frequency]
    # data_batches = tf.transpose(data_batches, perm=[1, 0, 2])

    # # [time, batch, frequency]
    # #return data_batches

    encoder_architecture = RNNArchitecture(num_layers=self._num_layers,
                                           num_units=self._num_units,
                                           bidirectional=self._bidirectional,
                                           cell_type=self._cell_type)
    decoder_architecture = RNNArchitecture(num_layers=self._num_layers,
                                           num_units=self._num_units,
                                           bidirectional=self._bidirectional,
                                           cell_type=self._cell_type)


    wrapper = TimeAutoencoderWrapper()

    wrapper.initialize_model(feature_shape=self._feature_shape,
                             model_filename=Path(self._model_filename),
                             encoder_architecture=encoder_architecture,
                             decoder_architecture=decoder_architecture,
                             mask_silence=False)

    wrapper.train_model(model_filename=Path(self._model_filename),
                        record_files=self._record_files,
                        feature_shape=self._feature_shape,
                        num_instances=self._num_instances,
                        num_epochs=self._num_epochs,
                        batch_size=self._trn_batch_size,
                        checkpoints_to_keep=self._checkpoints_to_keep,
                        learning_rate=self._learning_rate,
                        keep_prob=self._keep_prob,
                        encoder_noise=self._encoder_noise,
                        decoder_feed_previous_prob=self._feed_previous_prob)

    try:
      subprocess.call('scp -r '+self._save_path+'/et_'+str(etBinIdx)+'_eta_'+str(etaBinIdx)+'_sort_'+str(sort)+'/'+model_filename+' caducovas@bastion.lps.ufrj.br:/home/caducovas/lstm_output/',shell=True)
    except:
      print "NAO deu pra copiar pra bastion.."

    # # Train Process
    # SAE = StackedAutoEncoders(params = trn_params,
                              # development_flag = False,
                              # n_folds = 1,
                              # save_path = results_path,
                              # prefix_str=self._caltype
                              # )

    # self._SAE = SAE

    # # Choose layer to be trained
    # layer = self._layer

    # #self._info(self._hidden_neurons)

    # f, model, trn_desc = SAE.trainLayer(data=data,
                                        # trgt=val_Data,
                                        # ifold=0,
                                        # hidden_neurons=self._hidden_neurons,
                                        # layer = self._layer,sort=sort,etBinIdx=etBinIdx, etaBinIdx=etaBinIdx, tuning_folder = tuning_folder) #,regularizer='l1',regularizer_param=10e-5)
    # self._trn_desc = trn_desc
    # self._weights = model.get_weights()
    # self._trn_params = model.get_config()
    # #self._info(self._SAE)

    return self._apply(trnData,layerNumber)



  def __str__(self):
    """
      String representation of the object.
    """

    return (self._model_shortname+"_AutoEncoder_%d" % self._hidden_neurons)

  def shortName(self):
    """
      Short string representation of the object.
    """
    return (self._model_shortname+"_AE_%d" % self._hidden_neurons)

  def _apply(self, data,layerNumber):
    from RingerCore import retrieve_kw
    from audeep.backend.training.base import BaseFeatureLearningWrapper
    from audeep.backend.training.time_autoencoder import TimeAutoencoderWrapper
    #import numpy as np
    from pathlib import Path
    #self._info(pp.shortName())
    #self._info(self._etBinIdx)
    #self._info(self._etaBinIdx)
    ###get data projection
    print 'model_filename',self._model_filename
    #if not self._mean.size or not self._invRMS.size:
    #  self._fatal("Attempted to apply MapStd before taking its parameters.")
    wrapper = TimeAutoencoderWrapper()
    if isinstance(data, (tuple, list,)):
      ret = []
      # if self._caltype == 'emcalo' and data[0].shape[1] == 100:
        # #print 'EMMMMMMMMMMM'
        # data = [d[:,:88] for d in data]
        # #val_Data = [d[:,:88] for d in val_Data]
      # elif self._caltype == 'hadcalo' and data[0].shape[1] == 100:
        # #print 'HAAAAAAAAAAAD'
        # data = [d[:,88:] for d in data]
        #val_Data = [d[:,88:] for d in val_Data]
      #data = [d[:100] for d in data]
      for cdata in data:
        #self._info(cdata.shape)
        new_features = wrapper.generate_np_features(model_filename=Path(self._model_filename),
                                                    global_step=None,
                                                    data_set=cdata,
                                                    batch_size=10000)
        ret.append(new_features)
    else:
      new_features = wrapper.generate_np_features(model_filename=Path(self._model_filename),
                                                    global_step=self._global_step,
                                                    data_set=data,
                                                    batch_size=self._batch_size)
      ret = new_features
    return ret

  # def _undo(self, data):
    # if not self._mean.size or not self._invRMS.size:
      # self._fatal("Attempted to undo MapStd before taking its parameters.")
    # if isinstance(data, (tuple, list,)):
      # ret = []
      # for i, cdata in enumerate(data):
        # ret.append( ( cdata / self._invRMS ) + self._mean )
    # else:
      # ret = ( data / self._invRMS ) + self._mean
    #
    return ret

class NLPCA( PrepObj ):
  """
    Train the encoders in order to stack them as pre-processing afterwards.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {}, transientAttrs = {'_NLPCA',})
  _cnvObj = RawDictCnv(toProtectedAttrs = {})


  def __init__(self,n_inits=1,n_nlpcas=30, n_neurons_mapping=50, hidden_activation='tanh',output_activation='linear',n_epochs=5000,patience=30,batch_size=200,layer=1, d = {}, **kw):
    d.update( kw ); del kw
    from RingerCore import retrieve_kw
    #self._caltype = retrieve_kw(d,'caltype','allcalo')
    self._n_nlpcas = retrieve_kw(d,'nlpcs',30)
    self._n_neurons_mapping = retrieve_kw(d,'nmapping',50)
    #self._aetype = retrieve_kw(d,'aetype','vanilla') #VANILLA, SPARSE, DENOISING, CONTRACTIVE
    #self._dataEncoded = retrieve_kw(d,'dataEncoded','all') #all, signal, background
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    self._n_inits = n_inits

    #self._n_nlpcas=n_nlpcas
    #self._n_neurons_mapping=n_neurons_mapping

    self._hidden_activation = hidden_activation
    self._output_activation = output_activation
    self._n_epochs = n_epochs
    self._patience = patience
    self._batch_size = batch_size
    self._layer= layer
    del d
    self._sort = ''
    self._etBinIdx = ''
    self._etaBinIdx = ''
    self._NLPCA = ''
    self._trn_params = ''
    self._trn_desc = ''
    self._weights = ''

  def takeParams(self, trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder):

  ###trainlayer

    """
      Perform the layerwise algorithm to train the SAE
    """

    # TODO...
    self._sort = sort
    self._etBinIdx = etBinIdx
    self._etaBinIdx = etaBinIdx
    #print(self._caltype)

    import copy
    data = copy.deepcopy(trnData)
    val_Data = copy.deepcopy(valData)

    self._info('Training Data Shape: '+str(data[0].shape)+str(data[1].shape))
    self._info('Validation Data Shape: '+str(val_Data[0].shape)+str(val_Data[1].shape))

    #data = [d[:100] for d in data]
    #val_Data = [d[:100] for d in val_Data]

    #print "TESTEEEE"+tuning_folder

    self._batch_size = min(data[0].shape[0],data[1].shape[0])

    import numpy
    work_path='//home/caducovas/run/' #'/scratch/22061a/caducovas/run/'
    results_path = work_path+"nlpca_preproc/"
    #numpy.save(results_path+'val_Data_sort_'+str(self._sort)+'_nlpcas_'+str(self._n_nlpcas[0]),val_Data)
    #trn_params_folder = results_path+'trnparams_sort_'+str(self._sort)+'_nlpcas_'+str(self._n_nlpcas[0])+'.jbl'

    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    if isinstance(val_Data, (tuple, list,)):
      val_Data = np.concatenate( val_Data, axis=npCurrent.odim )

    # #Autoencoder Types paraemters and definitions
    # if self._aetype == 'sparse':
      # regularizer='l1'
      # regularizer_param=10e-5
    # else:
      # #regularizer='dropout'
      # #regularizer_param=0.5
      # regularizer=None
      # regularizer_param=None

    ###if self._aetype == 'contractive':
    ###  from TuningTools.MetricsLosses import contractive_loss
    ####  loss=contractive_loss
    ####else:
    ####  loss='mean_squared_error'

    # if os.path.exists(trn_params_folder):
        # os.remove(trn_params_folder)
    # if not os.path.exists(trn_params_folder):
        # trn_params = trnparams.NeuralClassificationTrnParams(n_inits=self._n_inits,
                                                             # hidden_activation=self._hidden_activation,
                                                             # output_activation=self._output_activation,
                                                             # n_epochs=self._n_epochs,
                                                             # patience=self._patience,
                                                             # batch_size=self._batch_size) #,
                                                             # #loss=loss)
    # trn_params.save(trn_params_folder)

    # self._trn_params = trn_params

    # self._info(trn_params.get_params_str())



    # Train Process
    NonLinearPCA = NonLinPCA(save_path = results_path)

    self._NLPCA = NonLinearPCA

    # Choose layer to be trained
    #layer = self._layer

    self._info(NonLinearPCA)

    NonLinearPCA.trainNLPCA(data=data,
                           trgt=val_Data,
                           n_inits = self._n_inits,
                           n_nlpcas=self._n_nlpcas,
                           n_neurons_mapping=self._n_neurons_mapping,
                           learning_rate=0.01,
                           learning_decay=0.00001,
                           momentum=0.3,
                           nesterov=True,
                           train_verbose=True,
                           n_epochs=self._n_epochs,
                           batch_size=self._batch_size,
                           sort=sort,
                           etBinIdx=etBinIdx,
                           etaBinIdx=etaBinIdx,
                           tuning_folder = tuning_folder)

    # self._trn_desc = trn_desc
    # self._weights = model.get_weights()
    # self._trn_params = model.get_config()
    #self._info(self._SAE)

    return self._apply(trnData)



  def __str__(self):
    """
      String representation of the object.
    """
    return ("NLPCA_"+str(self._n_neurons_mapping)+'-'+str(self._n_nlpcas))

  def shortName(self):
    """
      Short string representation of the object.
    """
    return ("NLPCA_"+str(self._n_neurons_mapping)+'-'+str(self._n_nlpcas))

  def _apply(self, data):
    #self._info(pp.shortName())
    #self._info(self._etBinIdx)
    #self._info(self._etaBinIdx)
    ###get data projection
    #if not self._mean.size or not self._invRMS.size:
    #  self._fatal("Attempted to apply MapStd before taking its parameters.")
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
	#self._info(cdata.shape)

        ret.append(self._NLPCA.getDataProjection(cdata, cdata, n_inits = self._n_inits, n_nlpcas=self._n_nlpcas, n_neurons_mapping=self._n_neurons_mapping,n_epochs=self._n_epochs,sort=self._sort,etBinIdx=self._etBinIdx, etaBinIdx=self._etaBinIdx))
    else:
      ret = self._NLPCA.getDataProjection(cdata, cdata, n_inits = self._n_inits, n_nlpcas=self._n_nlpcas, n_neurons_mapping=self._n_neurons_mapping,n_epochs=self._n_epochs,sort=self._sort,etBinIdx=self._etBinIdx, etaBinIdx=self._etaBinIdx)
    return ret

  # def _undo(self, data):
    # if not self._mean.size or not self._invRMS.size:
      # self._fatal("Attempted to undo MapStd before taking its parameters.")
    # if isinstance(data, (tuple, list,)):
      # ret = []
      # for i, cdata in enumerate(data):
        # ret.append( ( cdata / self._invRMS ) + self._mean )
    # else:
      # ret = ( data / self._invRMS ) + self._mean
    #
    return ret


class SAE_FineTuning( PrepObj ):
  """
    Fine Tune the SAE.
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {}, transientAttrs = {'_sae_ft',})
  _cnvObj = RawDictCnv(toProtectedAttrs = {})


  def __init__(self,n_inits=1,n_nlpcas=30, hidden_activation='tanh',output_activation='linear',n_epochs=5000,patience=30,batch_size=1024,layer=1, d = {}, **kw):
    d.update( kw ); del kw
    from RingerCore import retrieve_kw
    #self._caltype = retrieve_kw(d,'caltype','allcalo')
    self._n_nlpcas = retrieve_kw(d,'nlpcs',30)
    self._n_neurons_mapping = retrieve_kw(d,'nmapping',50)
    #self._aetype = retrieve_kw(d,'aetype','vanilla') #VANILLA, SPARSE, DENOISING, CONTRACTIVE
    #self._dataEncoded = retrieve_kw(d,'dataEncoded','all') #all, signal, background
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    self._n_inits = n_inits

    #self._n_nlpcas=n_nlpcas
    #self._n_neurons_mapping=n_neurons_mapping

    self._hidden_activation = hidden_activation
    self._output_activation = output_activation
    self._n_epochs = n_epochs
    self._patience = patience
    self._batch_size = batch_size
    self._layer= layer
    del d
    self._sort = ''
    self._etBinIdx = ''
    self._etaBinIdx = ''
    self._sae_ft= ''
    self._trn_params = ''
    self._trn_desc = ''
    self._weights = ''

  def takeParams(self, trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder):

  ###trainlayer

    """
      Perform the layerwise algorithm to train the SAE
    """

    # TODO...
    self._sort = sort
    self._etBinIdx = etBinIdx
    self._etaBinIdx = etaBinIdx
    #print(self._caltype)

    import copy
    data = copy.deepcopy(trnData)
    val_Data = copy.deepcopy(valData)

    self._info('Training Data Shape: '+str(data[0].shape)+str(data[1].shape))
    self._info('Validation Data Shape: '+str(val_Data[0].shape)+str(val_Data[1].shape))

    #data = [d[:100] for d in data]
    #val_Data = [d[:100] for d in val_Data]

    #print "TESTEEEE"+tuning_folder

    self._batch_size = min(data[0].shape[0],data[1].shape[0])

    import numpy
    work_path='//home/caducovas/run/' #'/scratch/22061a/caducovas/run/'
    results_path = work_path+"StackedAutoEncoder_preproc/"

    #numpy.save(results_path+'val_Data_sort_'+str(self._sort)+'_nlpcas_'+str(self._n_nlpcas[0]),val_Data)
    #trn_params_folder = results_path+'trnparams_sort_'+str(self._sort)+'_nlpcas_'+str(self._n_nlpcas[0])+'.jbl'

    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data, axis=npCurrent.odim )
    if isinstance(val_Data, (tuple, list,)):
      val_Data = np.concatenate( val_Data, axis=npCurrent.odim )

    from SAE_Evaluation import getSAE_Model
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD,Adam
    import keras.callbacks as callbacks
    from keras.utils import np_utils
    import keras

    # Train Process
    self._sae_ft = getSAE_Model(results_path+tuning_folder,trnData,self._sort)

    adamOpt = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
    self._sae_ft.compile(loss='mean_squared_error',
        optimizer=adamOpt,
        metrics=['accuracy'])

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    verbose=0,
                                                    mode='auto')
    checkpoints = keras.callbacks.ModelCheckpoint(work_path+'files/'+tuning_folder+'/models/fine_tuned_sort_'+str(self._sort)+'_et_'+str(self._etBinIdx)+'_eta_'+str(self._etaBinIdx)+'.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    #nlcpca_encoding_name  = 'nlpca_adam_bottleneck_%i_mapping_%i_epochs_sort_%i_etbin_%i_etabin_%i'%(n_nlpcas, n_neurons_mapping,sort,etBinIdx, etaBinIdx)
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/caducovas/tensorboard/graphs/'+nlcpca_encoding_name, histogram_freq=10, write_graph=True, write_images=True,write_grads=True,update_freq='epoch')
    import time
    import datetime
    start_run = time.time()


    hist = self._sae_ft.fit(data, data,
                                nb_epoch=5000,
                                batch_size= 1024, #train_info['batch_size'], #1024
                                callbacks=[earlyStopping,checkpoints],
                                verbose=1,
                                validation_data=(val_Data,
                                                val_Data),
                                shuffle=True)

    end_run = time.time()
    print 'Model SAE with Fine Tuning took '+ str(datetime.timedelta(seconds=(end_run - start_run))) +' to finish.'

    print self._sae_ft.layers

    return self._apply(trnData)



  def __str__(self):
    """
      String representation of the object.
    """
    return ("FineTuning")

  def shortName(self):
    """
      Short string representation of the object.
    """
    return ("FT")

  def _apply(self, data):
    import keras.backend as K
    #self._info(pp.shortName())
    #self._info(self._etBinIdx)
    #self._info(self._etaBinIdx)
    ###get data projection
    #if not self._mean.size or not self._invRMS.size:
    #  self._fatal("Attempted to apply MapStd before taking its parameters.")
    encoderLayerNum= int((float(len(self._sae_ft.layers))/2)-1)
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        #self._info(cdata.shape)
        get_layer_output = K.function([self._sae_ft.layers[0].input],
                                    [self._sae_ft.layers[encoderLayerNum].output])
        # Projection of layer
        proj_all_data = get_layer_output([cdata])[0]

        ret.append(proj_all_data)
    #else:
    #  ret = self._sae_ft.getDataProjection(cdata, cdata, n_inits = self._n_inits, n_nlpcas=self._n_nlpcas, n_neurons_mapping=self._n_neurons_mapping,n_epochs=self._n_epochs,sort=self._sort,etBinIdx=self._etBinIdx, etaBinIdx=self._etaBinIdx)
    return ret

  # def _undo(self, data):
    # if not self._mean.size or not self._invRMS.size:
      # self._fatal("Attempted to undo MapStd before taking its parameters.")
    # if isinstance(data, (tuple, list,)):
      # ret = []
      # for i, cdata in enumerate(data):
        # ret.append( ( cdata / self._invRMS ) + self._mean )
    # else:
      # ret = ( data / self._invRMS ) + self._mean
    #
    return ret



class MapStd_MassInvariant( MapStd ):
  """
    Remove data mean and set unitary standard deviation but "invariant" to each
    class mass.
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    MapStd.__init__( self, d )
    del d

  def takeParams(self, trnData):
    """
      Calculate mean and rms for transformation.
    """
    # Put all classes information into only one representation
    self._fatal('MapStd_MassInvariant still needs to be validated.')
    #if isinstance(trnData, (tuple, list,)):
    #  means = []
    #  means = np.zeros(shape=( trnData[0].shape[npCurrent.odim], len(trnData) ), dtype=trnData.dtype )
    #  for idx, cTrnData in enumerate(trnData):
    #    means[:,idx] = np.mean( cTrnData, axis=npCurrent.pdim, dtype=npCurrent.fp_dtype )
    #  self._mean = np.mean( means, axis=npCurrent.pdim )
    #  trnData = np.concatenate( trnData )
    #else:
    #  self._mean = np.mean( trnData, axis=0 )
    #trnData = trnData - self._mean
    #self._invRMS = 1 / np.sqrt( np.mean( np.square( trnData ), axis=0 ) )
    #self._invRMS[self._invRMS==0] = 1 # FIXME, not on invRMS, but before dividing it.
    #trnData *= self._invRMS
    #return trnData

  def __str__(self):
    """
      String representation of the object.
    """
    return "MapStd_MassInv"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "stdI"


class PCA( PrepObj ):
  """
    PCA preprocessing
  """
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self.energy = d.pop('energy' , None)

    checkForUnusedVars(d, self._warning )
    from sklearn import decomposition
    self._pca = decomposition.PCA(n_components = self.energy)

    #fix energy value
    if self.energy:  self.energy=int(self.energy)
    else:  self.energy=100 #total energy
    del d

  def params(self):
    return self._pca

  def variance(self):
    return self._pca.explained_variance_ratio_

  def cov(self):
    return self._pca.get_covariance()

  def ncomponents(self):
    return self.variance().shape[0]

  def takeParams(self, trnData):
    import copy
    data = copy.deepcopy(trnData)
    #val_Data = copy.deepcopy(valData)

    if isinstance(data, (tuple, list,)):
      data = np.concatenate( data )
    self._meanVector = np.mean(data.T, axis=1)
    #data -= self._meanVector #np.mean(data.T, axis=1)
    self._pca.fit(data)
    print 'PCA WTF', self.variance().shape, 'IS PCA REMOVING MEAN? ', np.array_equal(self._pca.mean_,np.mean(data.T,axis=1))
    self._info('PCA are aplied (%d of energy). Using only %d components of %d',
                      self.energy, self.ncomponents(), data.shape[1])
    #return trnData
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "PrincipalComponentAnalysis_"+str(self.energy)

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "PCA_"+str(self.energy)

  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        #self._meanVector = np.mean(cdata.T, axis=1)
        #self._meanVector2 = np.mean(cdata.T, axis=1)
        #cdata -= self._meanVector2
        #self._meanVector
        # FIXME Test this!
        if npCurrent.isfortran:
          print cdata.shape #,self._pca.transform(cdata.T).shape
          ret.append( self._pca.transform(cdata) )
        else:
          ret.append( self._pca.transform(cdata) )
    else:
      ret = self._pca.transform(data)
      if npCurrent.isfortran:
        ret = self._pca.transform(cdata.T).T
      else:
        ret = self._pca.transform(cdata)
    print len(ret),ret[0].shape,ret[1].shape
    return ret

  #def _undo(self, data):
  #  if isinstance(data, (tuple, list,)):
  #    ret = []
  #    for i, cdata in enumerate(data):
  #      ret.append( self._pca.inverse_transform(cdata) )
  #  else:
  #    ret = self._pca.inverse_transform(cdata)
  #  return ret

class RemoveRing( PrepObj ):
  """
    Remove one Ring preprocessing
  """
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self.RingRemoved = d.pop('remove' , None)

    checkForUnusedVars(d, self._warning )

    del d

  def takeParams(self, trnData):
    import copy
    data = copy.deepcopy(trnData)
    #val_Data = copy.deepcopy(valData)

    # if isinstance(data, (tuple, list,)):
      # data = np.concatenate( data )
    # #self._pca.fit(data)
    # print 'WTF', self.variance().shape
    # #self._info('PCA are aplied (%d of energy). Using only %d components of %d',
    # #                  self.energy, self.ncomponents(), data.shape[1])
    # #return trnData
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "RemoveRing"+str(self.RingRemoved)

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "RemoveRing"+str(self.RingRemoved)

  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        # FIXME Test this!
        if npCurrent.isfortran:
          print cdata.shape #,self._pca.transform(cdata.T).shape
          cdata = np.delete(cdata,int(self.RingRemoved)-1,axis=1)
          ret.append( cdata )
        else:
          cdata = np.delete(cdata,int(self.RingRemoved)-1,axis=1)
          ret.append( cdata )
    else:
      data = np.delete(data,int(self.RingRemoved)-1,axis=1)
      ret = data
    print len(ret),ret[0].shape,ret[1].shape
    return ret

  #def _undo(self, data):
  #  if isinstance(data, (tuple, list,)):
  #    ret = []
  #    for i, cdata in enumerate(data):
  #      ret.append( self._pca.inverse_transform(cdata) )
  #  else:
  #    ret = self._pca.inverse_transform(cdata)
  #  return ret

class RemoveCalSection( PrepObj ):
  """
    Remove one Ring preprocessing
  """
  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self.SectionRemoved = d.pop('remove' , None)
    self._listOfrings = []

    if self.SectionRemoved == "PS":
      self._listOfrings = [0,1,2,3,4,5,6,7]
    elif self.SectionRemoved == "EM1":
      self._listOfrings = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]
    elif self.SectionRemoved == "EM2":
      self._listOfrings = [72,73,74,75,76,77,78,79]
    elif self.SectionRemoved == "EM3":
      self._listOfrings = [80,81,82,83,84,85,86,87]
    elif self.SectionRemoved == "EM":
      self._listOfrings = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87]
    elif self.SectionRemoved == "HAD1":
      self._listOfrings = [88,89,90,91]
    elif self.SectionRemoved == "HAD2":
      self._listOfrings = [92,93,94,95]
    elif self.SectionRemoved == "HAD3":
      self._listOfrings = [96,97,98,99]
    elif self.SectionRemoved == "HAD":
      self._listOfrings = [88,89,90,91,92,93,94,95,96,97,98,99]

    checkForUnusedVars(d, self._warning )

    del d

  def takeParams(self, trnData):
    import copy
    data = copy.deepcopy(trnData)
    #val_Data = copy.deepcopy(valData)

    # if isinstance(data, (tuple, list,)):
      # data = np.concatenate( data )
    # #self._pca.fit(data)
    # print 'WTF', self.variance().shape
    # #self._info('PCA are aplied (%d of energy). Using only %d components of %d',
    # #                  self.energy, self.ncomponents(), data.shape[1])
    # #return trnData
    return self._apply(trnData)

  def __str__(self):
    """
      String representation of the object.
    """
    return "RemoveCalSection-"+self.SectionRemoved

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "RemoveCalSection-"+self.SectionRemoved

  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        # FIXME Test this!
        if npCurrent.isfortran:
          print cdata.shape #,self._pca.transform(cdata.T).shape
          cdata = np.delete(cdata,self._listOfrings,axis=1)
          ret.append( cdata )
        else:
          cdata = np.delete(cdata,self._listOfrings,axis=1)
          ret.append( cdata )
    else:
      data = np.delete(data,self._listOfrings,axis=1)
      ret = data
    print len(ret),ret[0].shape,ret[1].shape
    return ret

  #def _undo(self, data):
  #  if isinstance(data, (tuple, list,)):
  #    ret = []
  #    for i, cdata in enumerate(data):
  #      ret.append( self._pca.inverse_transform(cdata) )
  #  else:
  #    ret = self._pca.inverse_transform(cdata)
  #  return ret


class KernelPCA( PrepObj ):
  """
    Kernel PCA preprocessing
  """
  _explained_variance_ratio = None
  _cov = None

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )

    self._kernel                    = d.pop('kernel'                , 'rbf' )
    self._gamma                     = d.pop('gamma'                 , None  )
    self._n_components              = d.pop('n_components'          , None  )
    self._energy                    = d.pop('energy'                , None  )
    self._max_samples               = d.pop('max_samples'           , 5000  )
    self._fit_inverse_transform     = d.pop('fit_inverse_transform' , False )
    self._remove_zero_eig           = d.pop('remove_zero_eig'       , False )


    checkForUnusedVars(d, self._warning )

    if (self._energy) and (self._energy > 1):
      self._fatal('Energy value must be in: [0,1]')

    from sklearn import decomposition
    self._kpca  = decomposition.KernelPCA(kernel = self._kernel,
                                          n_components = self._n_components,
                                          eigen_solver = 'auto',
                                          gamma=self._gamma,
                                          fit_inverse_transform= self._fit_inverse_transform,
                                          remove_zero_eig=self._remove_zero_eig)
    del d

  def params(self):
    return self._kpca

  def takeParams(self, trnData):

    #FIXME: try to reduze the number of samples for large
    #datasets. There is some problem into sklearn related
    #to datasets with more than 20k samples. (lock to 16K samples)
    data = trnData
    if isinstance(data, (tuple, list,)):
      # FIXME Test kpca dimensions
      pattern=0
      for cdata in data:
        if cdata.shape[npCurrent.odim] > self._max_samples*0.5:
          self._warning('Pattern with more than %d samples. Reduce!',self._max_samples*0.5)
          data[pattern] = cdata[
            npCurrent.access( pdim=':',
                              odim=(0,np.random.permutation(cdata.shape[npCurrent.odim])[0:self._max_samples]))
                               ]
        pattern+=1
      data = np.concatenate( data, axis=npCurrent.odim )
      trnData = np.concatenate( trnData, axis=npCurrent.odim )
    else:
      if data.shape[0] > self._max_samples:
        data = data[
          npCurrent.access( pdim=':',
                            odim=(0,np.random.permutation(data.shape[npCurrent.odim])[0:self._max_samples]))
                   ]

    self._info('fitting dataset...')
    # fitting kernel pca
    if npCurrent.isfotran:
      self._kpca.fit(data.T)
      # apply transformation into data
      data_transf = self._kpca.transform(data.T).T
      self._cov = np.cov(data_transf.T)
    else:
      self._kpca.fit(data)
      # apply transformation into data
      data_transf = self._kpca.transform(data)
      self._cov = np.cov(data_transf)
    #get load curve from variance accumulation for each component
    explained_variance = np.var(data_transf,axis=npCurrent.pdim)
    self._explained_variance_ratio = explained_variance / np.sum(explained_variance)
    max_components_found = data_transf.shape[0]
    # release space
    data = []
    data_transf = []

    #fix n components by load curve
    if self._energy:
      cumsum = np.cumsum(self._explained_variance_ratio)
      self._n_components = np.where(cumsum > self._energy)[0][0]
      self._energy=int(self._energy*100) #fix representation
      self._info('Variance cut. Using components = %d of %d',self._n_components,max_components_found)
    #free, the n components will be max
    else:
      self._n_components = max_components_found

    return trnData[:,0:self._n_components]

  def kernel(self):
    return self._kernel

  def variance(self):
    return self._explained_variance_ratio

  def cov(self):
    return self._cov

  def ncomponents(self):
    return self._n_components

  def __str__(self):
    """
      String representation of the object.
    """
    if self._energy:
      return "KernelPCA_energy_"+str(self._energy)
    else:
      return "KernelPCA_ncomp_"+str(self._n_components)


  def shortName(self):
    """
      Short string representation of the object.
    """
    if self._energy:
      return "kPCAe"+str(self._energy)
    else:
      return "kPCAc"+str(self._n_components)


  def _apply(self, data):
    if isinstance(data, (tuple, list,)):
      ret = []
      for cdata in data:
        if npCurrent.isfortran:
          ret.append( self._kpca.transform(cdata.T).T[0:self._n_components,:] )
        else:
          ret.append( self._kpca.transform(cdata)[:,0:self._n_components] )
    else:
      ret = self._kpca.transform(data)[0:self._n_components]
      if npCurrent.isfortran:
        ret = self._kpca.transform(data.T).T[0:self._n_components,:]
      else:
        ret = self._kpca.transform(data)[:,0:self._n_components]
    return ret

  #def _undo(self, data):
  #  if isinstance(data, (tuple, list,)):
  #    ret = []
  #    for i, cdata in enumerate(data):
  #      ret.append( self._kpca.inverse_transform(cdata) )
  #  else:
  #    ret = self._kpca.inverse_transform(cdata)
  #  return ret

class RingerEtaMu(Norm1):
  """
    Applies norm-1+MapMinMax to data
  """

  def __init__(self, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    self._etamin           = d.pop( 'etamin'           , 0   )
    self._etamax           = d.pop( 'etamax'           , 2.5 )
    self._pileupThreshold  = d.pop( 'pileupThreshold'  , 60  )
    checkForUnusedVars(d, self._warning )
    del d

  def __str__(self):
    """
      String representation of the object.
    """
    return "ExpertNormalizationRingerEtaMu"

  def shortName(self):
    """
      Short string representation of the object.
    """
    return "ExNREM"


  def __retrieveNorm(self, data):
    """
      Calculate pre-processing parameters.
    """
    if isinstance(data, (tuple, list,)):
      norms = []
      for cdata in data:
        cnorm = cdata.sum(axis=npCurrent.pdim).reshape(
            npCurrent.access( pidx=1,
                              oidx=cdata.shape[npCurrent.odim] ) )
        cnorm[cnorm==0] = 1
        norms.append( cnorm )
    else:
      norms = data.sum(axis=npCurrent.pdim).reshape(
            npCurrent.access( pidx=1,
                              oidx=data.shape[npCurrent.odim] ) )
      norms[norms==0] = 1
    return norms

  def concatenate(self, data, extra):

    self._info('Concatenate extra patterns...')
    from TuningTools.dataframe import BaseInfo
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        cdata = np.concatenate((cdata, extra[i][BaseInfo.Eta], extra[i][BaseInfo.PileUp]),axis=1)
        ret.append(cdata)
    else:
      ret = np.concatenate((data, extra[i][BaseInfo.Eta], extra[i][BaseInfo.PileUp]),axis=1)
    return ret

  def _apply(self, data):

    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        norms = self.__retrieveNorm(cdata[ npCurrent.access( pidx=(0, 100) ) ])
        rings = cdata[ npCurrent.access( pidx=(0, 100) ) ] / norms[i]
        eta   = cdata[ npCurrent.access( pidx=(100,101), oidx=':' )]
        eta   = ((np.abs(eta) - np.abs(self._etamin))*np.sign(eta))/np.max(self._etamax)
        mu    = cdata[ npCurrent.access( pidx=(101,102) ,oidx=':') ]
        mu[mu > self._pileupThreshold] = self._pileupThreshold
        mu = mu/self._pileupThreshold
        cdata  = np.concatenate((rings,eta,mu),axis=1)
        ret.append(cdata)
    else:
      norms = self.__retrieveNorm(data)
      norms = self.__retrieveNorm(data[ npCurrent.access( pidx=(0, 100) ) ])
      rings = data[ npCurrent.access( pidx=(0, 100) ) ] / norms[i]
      eta   = data[ npCurrent.access( pidx=(100,101), oidx=':' )]
      eta   = ((np.abs(eta) - np.abs(self._etamin))*np.sign(eta))/np.max(self._etamax)
      mu[mu > self._pileupThreshold] = self._pileupThreshold
      mu = mu/self._pileupThreshold
      ret  = np.concatenate((rings,eta,mu),axis=1)

    return ret


class RingerFilterMu(PrepObj):
  '''
  Applies a filter at the mu values for training.
  '''

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_mu_min','_mu_max'})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_mu_min','_mu_max'})

  def __init__(self, mu_min=0, mu_max=70):
    super(RingerFilterMu, self).__init__()
    self._mu_min = mu_min
    self._mu_max = mu_max

  def __str__(self):
    return 'RingerFilterMu'

  def shortName(self):
    return 'RinFMU'

  def concatenate(self, data, extra):
    from TuningTools.dataframe import BaseInfo
    if isinstance(data, (tuple, list,)):
      self._verbose('Data is in list format...')
      ret = []
      for i, (cdata, cextra) in enumerate(zip(data, extra)):
        self._debug('Filtering mu values %d...' % i)
        cdata = cdata[np.squeeze(((cextra[BaseInfo.PileUp] >= self._mu_min) & (cextra[BaseInfo.PileUp] <= self._mu_max)),1)]
        ret.append(cdata)
    else:
      self._debug('Filtering mu values...')
      ret = np.concatenate((data, extra[i][BaseInfo.PileUp]),axis=npCurrent.pdim)
      ret = ret[npCurrent.access(oidx=((ret[:,-1] >= self._mu_min) & (ret[:,-1] <= self._mu_max)))]
    return ret

class StatReductionFactor(PrepObj):
  """
    Reduce the statistics available by the specified reduction factor
  """

  _streamerObj = LoggerRawDictStreamer(toPublicAttrs = {'_factor',})
  _cnvObj = RawDictCnv(toProtectedAttrs = {'_factor',})

  def __init__(self, factor = 1.1, d = {}, **kw):
    d.update( kw ); del kw
    PrepObj.__init__( self, d )
    checkForUnusedVars(d, self._warning )
    self._factor = factor
    del d

  @property
  def factor(self):
    return self._factor

  def __str__(self):
    "String representation of the object."
    return "StatReductionFactor(%.2f)" % self._factor

  def shortName(self):
    "Short string representation of the object."
    return ( "SRed%.2f" % self._factor ).replace('.','_')

  def concatenate(self, data, _):
    " Apply stats reduction "
    # We remove stats at random in the beginning of the process, so all sorts
    # have the same statistics available
    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        # We want to shuffle the observations, since shuffle method only
        # shuffles the first dimension, then:
        if npCurrent.odim == 0:
          np.random.shuffle( cdata )
        else:
          cdata = cdata.T
          np.random.shuffle( cdata.T )
          cdata = cdata.T
        origSize = cdata.shape[npCurrent.odim]
        newSize = int(round(origSize/self._factor))
        self._info("Reducing class %i size from %d to %d", i, origSize, newSize)
        cdata = cdata[ npCurrent.access( oidx=(0,newSize) ) ]
        ret.append(cdata)
    else:
      self._fatal('Cannot reduce classes size with concatenated data input')
    return ret


class PreProcChain ( Logger ):
  """
    The PreProcChain is the object to hold the pre-processing chain to be
    applied to data. They will be sequentially applied to the input data.
  """

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  #_streamerObj  = LoggerLimitedTypeListRDS( level = LoggingLevel.VERBOSE )
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )

  # These are the list (LimitedTypeList) accepted objects:
  _acceptedTypes = (PrepObj,)

  def __init__(self, *args, **kw):
    from RingerCore.LimitedTypeList import _LimitedTypeList____init__
    _LimitedTypeList____init__(self, *args)
    Logger.__init__(self, kw)

  def __call__(self, data, revert = False,sort=None,etBinIdx=None,etaBinIdx=None,tuning_folder=None):
    """
      Apply/revert pre-processing chain.
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
      return
    for pp in self:
      #if pp.shortName()[:2] == 'AE':
      #  data = pp(data,revert,sort,etBinIdx,etaBinIdx)
      #else:
      data = pp(data, revert)
    return data

  def __str__(self):
    """
      String representation of the object.
    """
    string = 'NoPreProc'
    if self:
      string = ''
      for pp in self:
        string += (str(pp) + '->')
      string = string[:-2]
    return string

  def shortName(self):
    string = 'NoPreProc'
    if self:
      string = ''
      for pp in self:
        string += (pp.shortName() + '-')
      string = string[:-1]
    return string

  def isRevertible(self):
    """
      Check whether the PreProc is revertible
    """
    for pp in self:
      if not pp.isRevertible():
        return False
    return True

  def takeParams(self,trnData,valData,sort=None,etBinIdx=None, etaBinIdx=None,tuning_folder=None):
    """
      Take pre-processing parameters for all objects in chain.
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
      return
    self._layerNumber=0
    for pp in self:
      self._info(pp.shortName())
      if 'NLPCA' in str(pp.shortName()): #pp.shortName()[-2:] == 'AE':
        trnData = pp.takeParams(trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder)
        valData = pp(valData, False)
      elif 'AE' in str(pp.shortName()): #pp.shortName()[-2:] == 'AE':
        self._layerNumber+=1
        trnData = pp.takeParams(trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder,self._layerNumber)
        valData = pp._apply(valData, self._layerNumber)
      elif 'FT' in str(pp.shortName()): #pp.shortName()[-2:] == 'AE':
          trnData,valData = self.getNorm1()
          trnData = pp.takeParams(trnData,valData,sort,etBinIdx, etaBinIdx,tuning_folder)
          valData = pp._apply(valData)
      else:
        trnData = pp.takeParams(trnData)
        pp._trn_norm1 = trnData
        valData = pp(valData, False)
        pp._val_norm1 = valData
    pp._SAE = ''
    return trnData,valData

  def getNorm1(self):
    """
      Returns the output of Norm1. Used when layer wise pre training is performed on Deep AutoEncoders so that they can be stacked and the fine tuning can be made with the output of Norm1
    """
    if not self:
      self._warning("No pre-processing available in this chan.")
      return
    emcalo = False
    hadcalo = False
    for pp in self:
      if pp.shortName() == 'N1' or pp.shortName() == 'std':
        #print trnData
        trnData = pp._trn_norm1
        valData = pp._val_norm1
        #pp._trn_norm1=''
        #pp._val_norm1=''
      if 'EM' in str(pp.shortName()):
        emcalo=True
      if 'HAD' in str(pp.shortName()):
        hadcalo=True
    if emcalo:
      #print 'EMMMMMMMMMMM'
      trnData = [d[:,:88] for d in trnData]
      valData = [d[:,:88] for d in valData]
    if hadcalo:
      #print 'HAAAAAAAAAAAD'
      trnData = [d[:,88:] for d in trnData]
      valData = [d[:,88:] for d in valData]

    return trnData,valData

  def getNorm1Parameters(self):
    """
      Returns the output of Reconstruction.
    """
    from SAE_Evaluation import *
    norm1Par=[]
    if not self:
      self._warning("No pre-processing available in this chan.")
      return
    emcalo = False
    hadcalo = False
    em1ae = False

    for pp in self:
      if pp.shortName() == 'N1':
        ##starts
        if em1ae:
          norm1Par.append([d[:,14:72] for d in pp._beforenorm])
          norm1Par.append(pp._normslist)
          norm1Par.append([d[:,14:72] for d in pp._afternorm])
        else:
          norm1Par.append(pp._beforenorm)
          norm1Par.append(pp._normslist)
          norm1Par.append(pp._afternorm)
        print 'NORM1 PAR'
        print len(pp._beforenorm),pp._beforenorm[0].shape,pp._beforenorm[1].shape
        print len(pp._afternorm),pp._afternorm[0].shape,pp._afternorm[1].shape
        print len(pp._normslist),pp._normslist[0].shape,pp._normslist[1].shape

        self._normslist = ''
        self._beforenorm = ''
        self._afternorm = ''
        pp._trn_norm1=''
        pp._val_norm1=''
      elif pp.shortName() == 'std':
        normslist=[]
        normslist.append(pp._mean)
        normslist.append(pp._invRMS)
        norm1Par.append(pp._beforenorm)
        norm1Par.append(normslist)
        norm1Par.append(pp._afternorm)
        # valData = pp._val_norm1
        # pp._trn_norm1=''
        # pp._val_norm1=''
      # if 'EM' in str(pp.shortName()):
        # emcalo=True
      # if 'HAD' in str(pp.shortName()):
        # hadcalo=True
    # if emcalo:
      # #print 'EMMMMMMMMMMM'
      # trnData = [d[:,:88] for d in trnData]
      # valData = [d[:,:88] for d in valData]
    # if hadcalo:
      # #print 'HAAAAAAAAAAAD'
      # trnData = [d[:,88:] for d in trnData]
      # valData = [d[:,88:] for d in valData]

    return norm1Par

  def getPCAModel(self):
    """
      Return the PCA model in order to make it possible to return to the original space
    """
    for pp in self:
      if 'PCA' in str(pp.shortName()):
        return pp._pca,pp._meanVector

  def getHiddenLayer(self):
    """
      Return a list with the number of neurons on the hidden layer,weights and configuration for Stacked AutoEncoders.
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
      return
    hidden_neurons=[]
    weights=[]
    config=[]
    for pp in self:
      if 'AE' in str(pp.shortName()): #pp.shortName()[-2:] == 'AE':
            #self._info(pp._weights)
            hidden_neurons.append(int(pp.shortName().split('_')[1]))
            weights.append(pp._weights)
            config.append(pp._trn_params)
    return hidden_neurons,weights,config

  def getLSTM_Model_filename(self):
    """
      Return a list with the number of neurons on the hidden layer,weights and configuration for Stacked AutoEncoders.
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
      return

    for pp in self:
      if 'LSTM' in str(pp.shortName()) or 'GRU' in str(pp.shortName()): #pp.shortName()[-2:] == 'AE':
        #self._info(pp._weights)
        model_filename = pp._model_filename
    return model_filename

  def concatenate(self, trnData, extraData):
    """
      Concatenate extra patterns into the data input
    """
    if not self:
      self._warning("No pre-processing available in this chain.")
      return
    for pp in self:
      trnData = pp.concatenate(trnData, extraData)

    return trnData

  def setLevel(self, value):
    """
      Override Logger setLevel method so that we can be sure that every
      pre-processing will have same logging level than that was set for the
      PreProcChain instance.
    """
    for pp in self:
      pp.level = LoggingLevel.retrieve( value )
    self._level = LoggingLevel.retrieve( value )
    self._logger.setLevel(self._level)

  level = property( Logger.getLevel, setLevel )


class PreProcCollection( object ):
  """
    The PreProcCollection will hold a series of PreProcChain objects to be
    tested. The TuneJob will apply them one by one, looping over the testing
    configurations for each case.
  """

  # Use class factory
  __metaclass__ = LimitedTypeStreamableList
  #_streamerObj  = LimitedTypeListRDS( level = LoggingLevel.VERBOSE )
  #_cnvObj       = LimitedTypeListRDC( level = LoggingLevel.VERBOSE )
  _acceptedTypes = type(None),
# The PreProcCollection can hold a collection of itself besides PreProcChains:
PreProcCollection._acceptedTypes = PreProcChain, PreProcCollection
