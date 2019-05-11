from sklearn.metrics import mutual_info_score
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.axes_grid import AxesGrid
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import ScalarFormatter
import scipy
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.regularizers import l1,l2
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import pandas as pd
import numpy
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
from sklearn.metrics import confusion_matrix
from SAE_Evaluation import *
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.axes_grid import AxesGrid
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import telepot
from grid_strategy import *
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
bot = telepot.Bot('578139897:AAEJBs9F21TojbPoXM8SIJtHrckaBLZWkpo')

def calc_MI2(x, y):
  max_value = max(max(x),max(y))
  min_value = min(min(x),min(y))
  bins = min( len(np.histogram(x,'fd')[0]), len(np.histogram(y,'fd')[0]))
  bins_list = np.linspace(min_value, max_value, num=bins)
  c_xy,xaaa,yaaa = np.histogram2d(x, y, bins=(bins_list,bins_list))
  mi = mutual_info_score(None, None, contingency=c_xy)
  return mi #,xaaa,yaaa,bins

def calc_kl(x, y):
    from scipy import stats
    max_value = max(max(x),max(y))
    min_value = min(min(x),min(y))
    bins = min( len(np.histogram(x,'fd',density=True)[0]), len(np.histogram(y,'fd',density=True)[0]))
    bins_list = np.linspace(min_value, max_value, num=bins)
    p,phist_bins=np.histogram(x,bins_list,density=True)
    q,qhist_bins=np.histogram(y,bins_list,density=True)
    #print(len(p),len(q))
    kl = stats.entropy(pk=p+0.00001, qk=q+0.00001)
    return kl

def calc_chisquare(x, y):
    from scipy import stats
    max_value = max(max(x),max(y))
    min_value = min(min(x),min(y))
    bins = min( len(np.histogram(x,'fd')[0]), len(np.histogram(y,'fd')[0]))
    bins_list = np.linspace(min_value, max_value, num=bins)
    p,phist_bins=np.histogram(x,bins_list)
    q,qhist_bins=np.histogram(y,bins_list)
    #print(len(p),len(q))
    chi = stats.chisquare(q+0.00001, p+0.00001)
    return chi


def layer2number(x, y):
  return int(y.split('x')[1]) - int(x.split('x')[1])

def layer2number2(x,y):
    return int(x[1]) - int(y[1])

def avgNestedLists(nested_vals):
  """
  Averages a 2-D array and returns a 1-D array of all of the columns
  averaged together, regardless of their dimensions.
  """
  output = []
  maximum = 0
  for lst in nested_vals:
    if len(lst) > maximum:
      maximum = len(lst)
  for index in range(maximum): # Go through each index of longest list
    temp = []
    for lst in nested_vals: # Go through each list
      if index < len(lst): # If not an index error
        temp.append(lst[index])
    output.append(np.nanmean(temp))
  return output

def stdNestedLists(nested_vals):
  """
  Averages a 2-D array and returns a 1-D array of all of the columns
  averaged together, regardless of their dimensions.
  """
  output = []
  maximum = 0
  for lst in nested_vals:
    if len(lst) > maximum:
      maximum = len(lst)
  for index in range(maximum): # Go through each index of longest list
    temp = []
    for lst in nested_vals: # Go through each list
      if index < len(lst): # If not an index error
        temp.append(lst[index])
    output.append(np.nanstd(temp))
  return output

def plot_AE_training(fname,dirout):
  png_files=[]
  with open(fname) as f:
    content = f.readlines()
  f.close()

  layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  layers=sorted(list(set(layers_list)),cmp=layer2number)
  print(layers)
  list_t=[]
  for layer in layers:
    epochs = {}
    loss = {}
    kl = {}
    val_loss = {}
    val_kl = {}
    files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
    for file in files:
      job = joblib.load(file.replace('\n','')+'_trn_desc.jbl')
      file_name = file.split('/')[-1]
      print file,file.split('_')[27]
      epochs[int(file_name.split('_')[27])] = job[0]['epochs']
      loss[int(file_name.split('_')[27])] = job[0]['loss']
      kl[int(file_name.split('_')[27])] = job[0]['kullback_leibler_divergence']
      val_loss[int(file_name.split('_')[27])] = job[0]['val_losS']
      val_kl[int(file_name.split('_')[27])] = job[0]['val_kullback_leibler_divergence']
    #print len(loss.values())
    #print list(loss.values())
    max_epochs = np.max(epochs.values())
    print 'max_epochs', max_epochs, type(max_epochs)
    loss_mean = avgNestedLists(list(loss.values())) #np.mean(list(loss.values()),axis=0)
    loss_std = stdNestedLists(list(loss.values())) #np.std(loss.values(),axis=0)
    val_loss_mean = avgNestedLists(list(val_loss.values())) #np.mean(val_loss.values(),axis=0)
    val_loss_std = stdNestedLists(list(val_loss.values())) #np.std(val_loss.values(),axis=0)
    kl_mean = avgNestedLists(list(kl.values())) #np.mean(kl.values(),axis=0)
    kl_std = stdNestedLists(list(kl.values())) #np.std(kl.values(),axis=0)
    val_kl_mean = avgNestedLists(list(val_kl.values())) #np.mean(val_kl.values(),axis=0)
    val_kl_std = stdNestedLists(list(val_kl.values())) #np.std(val_kl.values(),axis=0)



    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    plt.figure(1)
    ##PLOT MSE TREINAMENTO
    #list_t = []
    plt.subplot(221)
    plt.errorbar(range(len(loss_mean)),y=loss_mean,yerr=loss_std,errorevery=10)
    #3#for i in range(len(epochs.keys())):
      ###plt.plot(epochs[i],loss[i])
      #plt.plot(max_epochs,loss_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
    print list_t
    plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    #plt.title('SAE - '+layer.replace('x','-')+'-'+layer.split('x')[0],fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel('Erro de Treinamento (MSE)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')

    ##PLOT MSE Val
    #list_t = []
    plt.subplot(222)

    plt.errorbar(range(len(val_loss_mean)),y=val_loss_mean,yerr=val_loss_std,errorevery=10)
    #for i in range(len(epochs.keys())):
    #  plt.plot(epochs[i],val_loss[i])
      #plt.plot(max_epochs,val_loss_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
    plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel(r'Erro de $Validac\c{}\~ao$ (MSE)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')

    ##PLOT KL TREINAMENTO

    #list_t = []
    plt.subplot(223)

    plt.errorbar(range(len(kl_mean)),y=kl_mean,yerr=kl_std,errorevery=10)
    #for i in range(len(epochs.keys())):
    #  plt.plot(epochs[i],kl[i])
      #plt.plot(max_epochs,kl_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
    plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel('Erro de Treinamento (KL)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')

    ##PLOT KL Val

    #list_t = []
    plt.subplot(224)
    plt.errorbar(range(len(val_kl_mean)),y=val_kl_mean,yerr=val_kl_std,errorevery=10)

    #for i in range(len(epochs.keys())):
    #  plt.plot(epochs[i],val_kl[i])
      #plt.plot(max_epochs,val_kl_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
    plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel(r'Erro de $Validac\c{}\~ao$ (KL)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')

  #plt.grid()
  plt.savefig(dirout+'layer_'+layer+'_'+fname.split('/')[-1]+'.png')
  png_files.append(dirout+'layer_'+layer+'_'+fname.split('/')[-1]+'.png')
  plt.clf()
  plt.close()
  return png_files

def plot_NLPCA_training(fname,dirout):
  png_files=[]
  with open(fname) as f:
      content = f.readlines()
  f.close()
  epochs = {}
  loss = {}
  kl = {}
  val_loss = {}
  val_kl = {}

  #files = [f for f in content if f.split('_')[9] == str(sort)]
  files = [f for f in content]
  for file_name in files:
      file = '//home/caducovas/run/nlpca_preproc/output_files/'+file_name #.replace('\n','')+'_train_desc.jbl'
      job = joblib.load(file.replace('\n','')+'_train_desc.jbl')
      #file_name = file.split('/')[-1]
      print file,file_name.split('_')[9]
      epochs[int(file_name.split('_')[9])] = job[0][file_name.split('_')[3]]['epochs']
      loss[int(file_name.split('_')[9])] = job[0][file_name.split('_')[3]]['perf']
      ###kl[int(file_name.split('_')[9])] = job[0]['kullback_leibler_divergence']
      val_loss[int(file_name.split('_')[9])] = job[0][file_name.split('_')[3]]['vperf']
      ###val_kl[int(file_name.split('_')[9])] = job[0]['val_kullback_leibler_divergence']
  #print len(loss.values())
  #print list(loss.values())
  bottleneck = file_name.split('_')[3]
  max_epochs = np.max(epochs.values())
  print 'max_epochs', max_epochs, type(max_epochs)
  loss_mean = avgNestedLists(list(loss.values())) #np.mean(list(loss.values()),axis=0)
  loss_std = stdNestedLists(list(loss.values())) #np.std(loss.values(),axis=0)
  val_loss_mean = avgNestedLists(list(val_loss.values())) #np.mean(val_loss.values(),axis=0)
  val_loss_std = stdNestedLists(list(val_loss.values())) #np.std(val_loss.values(),axis=0)
  #kl_mean = avgNestedLists(list(kl.values())) #np.mean(kl.values(),axis=0)
  #kl_std = stdNestedLists(list(kl.values())) #np.std(kl.values(),axis=0)
  #val_kl_mean = avgNestedLists(list(val_kl.values())) #np.mean(val_kl.values(),axis=0)
  #val_kl_std = stdNestedLists(list(val_kl.values())) #np.std(val_kl.values(),axis=0)

  fig, axs = plt.subplots(1, 2, figsize=(24, 18))
  plt.figure(1)
  ##PLOT MSE TREINAMENTO
  #list_t = []
  plt.subplot(121)
  plt.errorbar(range(len(loss_mean)),y=loss_mean,yerr=loss_std,errorevery=10)
  #3#for i in range(len(epochs.keys())):
    ###plt.plot(epochs[i],loss[i])
    #plt.plot(max_epochs,loss_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  print list_t
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  #plt.title('SAE - '+layer.replace('x','-')+'-'+layer.split('x')[0],fontsize= 'xx-large')
  plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel('Erro de Treinamento (MSE)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  plt.yscale('log')

  ##PLOT MSE Val
  #list_t = []
  plt.subplot(122)

  plt.errorbar(range(len(val_loss_mean)),y=val_loss_mean,yerr=val_loss_std,errorevery=10)
  #for i in range(len(epochs.keys())):
  #  plt.plot(epochs[i],val_loss[i])
    #plt.plot(max_epochs,val_loss_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel(r'Erro de $Validac\c{}\~ao$ (MSE)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  plt.yscale('log')

  ##PLOT KL TREINAMENTO

  #list_t = []
  # plt.subplot(223)
  #
  # plt.errorbar(range(len(kl_mean)),y=kl_mean,yerr=kl_std,errorevery=10)
  # #for i in range(len(epochs.keys())):
  # #  plt.plot(epochs[i],kl[i])
  #   #plt.plot(max_epochs,kl_mean)
  # #plt.plot(T[i].history['val_loss'])
  # #list_t.append('Sorteio %.f'%(i+1))
  # #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  # plt.legend(list_t, loc='best',fontsize = 'xx-large')
  # #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  # plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  # plt.ylabel('Erro de Treinamento (KL)',fontsize= 'xx-large')
  # plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  # plt.tick_params(axis='both',labelsize=16)
  # #plt.xlim(0)
  # #plt.grid()
  # plt.yscale('log')
  #
  # ##PLOT KL Val
  #
  # #list_t = []
  # plt.subplot(224)
  # plt.errorbar(range(len(val_kl_mean)),y=val_kl_mean,yerr=val_kl_std,errorevery=10)
  #
  # #for i in range(len(epochs.keys())):
  # #  plt.plot(epochs[i],val_kl[i])
  #   #plt.plot(max_epochs,val_kl_mean)
  # #plt.plot(T[i].history['val_loss'])
  # #list_t.append('Sorteio %.f'%(i+1))
  # #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  # plt.legend(list_t, loc='best',fontsize = 'xx-large')
  # #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  # plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  # plt.ylabel(r'Erro de $Validac\c{}\~ao$ (KL)',fontsize= 'xx-large')
  # plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  # plt.tick_params(axis='both',labelsize=16)
  # #plt.xlim(0)
  # #plt.grid()
  # plt.yscale('log')

  #plt.grid()
  plt.savefig(dirout+'bottleneck_'+bottleneck+'_'+fname.split('/')[-1]+'.png')
  png_files.append(dirout+'bottleneck_'+bottleneck+'_'+fname.split('/')[-1]+'.png')
  plt.clf()
  plt.close()
  return png_files


def plot_classifier_training(fname,dirout):
  import os
  history_files=[x for x in os.listdir(fname) if x.endswith(".pkl")]
  png_files=[]
  #with open(fname) as f:
  #  content = f.readlines()
  #f.close()

  #layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  #layers=sorted(list(set(layers_list)),cmp=layer2number)
  #print(layers)
  list_t=[]
  #for layer in layers:
  epochs = {}
  loss = {}
  acc = {}
  val_loss = {}
  val_acc = {}
  #files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
  for file in history_files:
    job = load_dl_history(fname+'/'+file) #joblib.load(file.replace('\n','')+'_trn_desc.jbl')
    #print job.keys()
    sort = int(file.split('/')[-1].split('_')[2])
    #print sort
    #print file.split('_')[27]
    epochs[sort] = len(job['loss'])
    loss[sort] = job['loss']
    acc[sort] = job['acc']
    val_loss[sort] = job['val_loss']
    val_acc[sort] = job['val_acc']
  #print len(loss.values())
  #print list(loss.values())
  #max_epochs = np.max(epochs.values())
  #loss_mean = avgNestedLists(list(loss.values())) #np.mean(list(loss.values()),axis=0)
  #loss_std = stdNestedLists(list(loss.values())) #np.std(loss.values(),axis=0)
  #val_loss_mean = avgNestedLists(list(val_loss.values())) #np.mean(val_loss.values(),axis=0)
  #val_loss_std = stdNestedLists(list(val_loss.values())) #np.std(val_loss.values(),axis=0)
  #acc_mean = avgNestedLists(list(acc.values())) #np.mean(kl.values(),axis=0)
  #acc_std = stdNestedLists(list(acc.values())) #np.std(kl.values(),axis=0)
  #val_acc_mean = avgNestedLists(list(val_acc.values())) #np.mean(val_kl.values(),axis=0)
  #val_acc_std = stdNestedLists(list(val_acc.values())) #np.std(val_kl.values(),axis=0)
    #print loss[sort]
  fig, axs = plt.subplots(2, 2, figsize=(24, 18))
  plt.figure(1)
  ##PLOT MSE TREINAMENTO
  #list_t = []
  plt.subplot(221)
  #print loss[0]
  #plt.errorbar(range(max_epochs+1),y=loss_mean,yerr=loss_std,errorevery=10)
  for i in range(len(epochs.keys())):
    #print i,loss[i]
    plt.plot(loss[i])

    #plt.plot(max_epochs,loss_mean)
  #plt.plot(T[i].history['val_loss'])
    list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  #print list_t
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  #plt.title('SAE - '+layer.replace('x','-')+'-'+layer.split('x')[0],fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel('Erro de Treinamento (MSE)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  plt.yscale('log')
  ##PLOT MSE Val
  #list_t = []
  plt.subplot(222)

  #plt.errorbar(range(max_epochs+1),y=val_loss_mean,yerr=val_loss_std,errorevery=10)
  for i in range(len(epochs.keys())):
    plt.plot(val_loss[i])
    #plt.plot(max_epochs,val_loss_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel(r'Erro de $Validac\c{}\~ao$ (MSE)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  plt.yscale('log')

  ##PLOT KL TREINAMENTO

  #list_t = []
  plt.subplot(223)

  #plt.errorbar(range(max_epochs+1),y=kl_mean,yerr=kl_std,errorevery=10)
  for i in range(len(epochs.keys())):
    plt.plot(acc[i])
    #plt.plot(max_epochs,kl_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel('Erro de Treinamento (Acc)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  #plt.yscale('log')
  ##PLOT KL Val

  #list_t = []
  plt.subplot(224)
  #plt.errorbar(range(max_epochs+1),y=val_kl_mean,yerr=val_kl_std,errorevery=10)

  for i in range(len(epochs.keys())):
    plt.plot(val_acc[i])
    #plt.plot(max_epochs,val_kl_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel(r'Erro de $Validac\c{}\~ao$ (Acc)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  #plt.yscale('log')

  #plt.grid()
  plt.savefig(dirout+'dl_'+fname.split('/')[-1]+'.png')
  png_files.append(dirout+'dl_'+fname.split('/')[-1]+'.png')
  return png_files

def save_dl_history(path,obj):
  import pickle
  with open(path + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
  f.close()

def load_dl_history(path ):
  import pickle
  with open(path, 'rb') as f:
    return pickle.load(f)
  f.close()
def save_dl_model(path=None,model=None):
  # serialize model to JSON
  model_json = model.to_json()
  with open(path+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(path+".h5")
  #print("Saved model to disk")

def load_dl_model(path=None,model=None):
  from keras.models import model_from_json
  import json
  # load json and create model
  json_file = open(path+".json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(path+".h5")
  print("Loaded model from disk")
  return loaded_model

def print_metrics(metricsDict):
  for key in metricsDict.keys():
    if isinstance(metricsDict[key], float):
      print("{:15}: {:.2f}".format(key, metricsDict[key]))
    else:
      print("{:15}: {}".format(key, metricsDict[key]))

    return 0

def report_performance(labels, predictions, elapsed=0, model_name="",hl_neuron=None,time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=None,points=None,fine_tuning=None,report=True):
  from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, mean_squared_error
  import dataset
  db = dataset.connect('sqlite://///home/caducovas/run/ringerMLlab.db')
  print float(mean_squared_error(labels, predictions)),  roc_auc_score(labels, predictions)
  mse_score = float(mean_squared_error(labels, predictions))

  #print point.sp_value
  tabela = db['classifiers']
  print "QNT DE PONTOS",len(points)
  for refName,point in points:
    data = OrderedDict()
    print len(predictions)
    if 'log_reg' in model_name:
      predictions[predictions >= point.thres_value] = 1
      predictions[predictions < point.thres_value] = 0
    else:
      predictions[predictions >= point.thres_value] = 1
      predictions[predictions < point.thres_value] = -1

    print 'debugging report_performance????'
    #print labels
    #print predictions
    print 'REF',refName
    print 'SP',float(point.sp_value)
    print 'WTF'
    data['Point'] = refName
    data['Model'] = model_name
    data['HL_Neuron'] = hl_neuron
    data['time'] = time
    data['sort'] = sort
    data['etBinIdx'] = etBinIdx
    data['etaBinIdx'] = etaBinIdx
    data['phase'] = phase
    data['Elapsed'] = elapsed
    data['fine_tuning'] = fine_tuning
    data['signal_samples'] = len(labels[labels==1])
    data['bkg_samples'] = len(labels[labels==-1])
    data['signal_pred_samples'] = len(predictions[predictions==1])
    data['bkg_pred_samples'] = len(predictions[predictions==-1])
    data['threshold']=float(point.thres_value)
    data['sp'] = float(point.sp_value)
    data['pd'] = float(point.pd_value)
    data['pf'] = float(point.pf_value)
    data['accuracy'] = accuracy_score(labels, predictions, normalize=True)
    data['MSE'] = mse_score
    data['f1'] = f1_score(labels, predictions)
    data['auc'] = roc_auc_score(labels, predictions)
    data['precision'] = precision_score(labels, predictions)
    data['recall'] = recall_score(labels, predictions)
    print 'OK insert'
    print data
    tabela.insert(data,ensure=True)
  #if report == True:
  #  print_metrics(data)

  return data

def cross_val_analysis_nn(n_split=10, classifier=None, x=None, y=None, model_name="",
              patience=30, train_verbose=2, n_epochs=500):
  '''
    Classification and ROC analysis
    Run classifier with cross-validation and plot ROC curves
  '''
  kf = KFold(n_splits=n_split)
  kf.get_n_splits(x)

  tprs = []
  fpr_ = []
  tpr_ = []
  aucs = []
  accuracy_ = []
  f1_score_ = []
  precision_ = []
  recall_ = []
  roc_auc_ = []

  metrics_ = {}
  trn_desc = {}
  mean_fpr = np.linspace(0, 1, 100)

  batch_size = min(x[y==-1].shape[0],x[y==1].shape[0])

  i = 0
  #start_time = time.time()
  for train, val in kf.split(x,y):
    print('Train Process for %i Fold'%(i+1))
    #print("TRAIN:", train_index, "TEST:", test_index)
    #trainX, valX = trainDf[train_index], trainDf[val_index]
    #trainY, valY = y_train[train_index], y_train[val_index]

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=train_verbose, mode='auto')
    model = classifier.fit(x.iloc[train], y[train], nb_epoch=n_epochs, callbacks=[earlyStopping], verbose=train_verbose, validation_data=(x.iloc[val], y[val]))
    trn_desc[i] = model
    #model = classifier.fit(x.iloc[train], y[train])
    pred_ = model.predict(x.iloc[val])
    probas_ = model.predict_proba(x.iloc[val])

    # Metrics evaluation
    accuracy_.append(100*accuracy_score(y[val],pred_ , normalize=True))
    f1_score_.append(100*f1_score(y[val], pred_))
    roc_auc_.append(100*roc_auc_score(y[val], pred_))
    precision_.append(100*precision_score(y[val], pred_))
    recall_.append(100*recall_score(y[val], pred_))


    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[val], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    fpr_.append(fpr)
    tpr_.append(tpr)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold %d (AUC = %0.2f)' % (i, 100*roc_auc))

    i += 1
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
       label='Luck', alpha=.8)

  #Store average and std metrics in dict
  metrics_['model']=model_name
  metrics_['accuracy']=round(np.mean(accuracy_),2)
  metrics_['accuracy_std']=round(np.std(accuracy_),2)
  #metrics_['fpr']=round(np.mean(fpr_),2)
  #metrics_['fpr_std']=round(np.std(fpr_),2)
  #metrics_['tpr']=round(np.mean(tpr_),2)
  #metrics_['tpr_std']=round(np.std(tpr_),2)
  metrics_['precision']=round(np.mean(precision_),2)
  metrics_['precision_std']=round(np.std(precision_),2)
  metrics_['recall']=round(np.mean(recall_),2)
  metrics_['recall_std']=round(np.std(recall_),2)
  metrics_['roc_auc']=round(np.mean(roc_auc_),2)
  metrics_['roc_auc_std']=round(np.std(roc_auc_),2)
  metrics_['f1']=round(np.mean(f1_score_),2)
  metrics_['f1_std']=round(np.std(f1_score_),2)


  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  plt.plot(mean_fpr, mean_tpr, color='b',
       label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (100*mean_auc, 100*std_auc),
       lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
           label=r'$\pm$ 1 std. dev.')

  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(model_name+' Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()

  return metrics_,trn_desc

def createClassifierTable(model_name,script_time,Point):
  import dataset
  from prettytable import PrettyTable
  print Point
  x = PrettyTable()
  x.field_names = ["KPI", "Train", "Validation"]
  db = dataset.connect('sqlite://///home/caducovas/run/ringerMLlab.db')
  #table = db['classifier']

  #query = 'select model,time,phase, avg(elapsed) as elapsed, avg(signal_samples) as signal_samples,avg(bkg_samples) as bkg_samples,avg(signal_pred_samples) as signal_pred_samples,avg(bkg_pred_samples) as bkg_pred_samples,avg(threshold) as threshold,  avg(sp) || "+-" || stdev(sp) as sp, avg(pd) || "+-" || stdev(pd) as pd, avg(pf) || "+-" || stdev(pf) as pf, avg(accuracy) || "+-" || stdev(accuracy) as accuracy, avg(f1) || "+-" || stdev(f1) as f1, avg(auc) || "+-" || stdev(auc) as auc,  avg(precision) || "+-" || stdev(precision) as precision, avg(recall) || "+-" || stdev(recall) as recall from classifier group by model,time,phase'

  query = 'select point,model,time,phase,fine_tuning, max(elapsed) as elapsed, avg(signal_samples) as signal_samples,avg(bkg_samples) as bkg_samples,avg(signal_pred_samples) as signal_pred_samples,avg(bkg_pred_samples) as bkg_pred_samples, 100*round(avg(threshold),5) as threshold,  100*round(avg(sp),5) as sp, 100*round(avg(pd),5) as pd, 100*round(avg(pf),5) as pf, 100*round(avg(accuracy),5) as accuracy, 100*round(avg(f1),5) as f1, 100*round(avg(auc),5) as auc, 100*round(avg(precision),5) as precision, 100*round(avg(recall),5) as recall from classifiers where model = "'+model_name+'" and time = "'+script_time+'" group by point,model,time,phase,fine_tuning'
  trnquery = 'select point,model,time,phase,fine_tuning, max(elapsed) as elapsed, cast(avg(signal_samples) as integer)  as signal_samples,cast(avg(bkg_samples) as integer) as bkg_samples,cast(avg(signal_pred_samples) as integer) as signal_pred_samples,cast(avg(bkg_pred_samples) as integer) as bkg_pred_samples, 100*round(avg(threshold),5) as threshold,  100*round(avg(sp),5) as sp, 100*round(avg(pd),5) as pd, 100*round(avg(pf),5) as pf, 100*round(avg(accuracy),5) as accuracy, 100*round(avg(f1),5) as f1, 100*round(avg(auc),5) as auc, 100*round(avg(precision),5) as precision, 100*round(avg(recall),5) as recall from classifiers where model = "'+model_name+'" and (sort,time,pf) in (select sort,time,min(pf) from classifiers where time = "'+script_time+'" and Point = "'+Point+'" and phase = "Train" group by sort,time) group by point,model,time,phase,fine_tuning'
  valquery = 'select point,model,time,phase,fine_tuning, max(elapsed) as elapsed, cast(avg(signal_samples) as integer) as signal_samples,cast(avg(bkg_samples) as integer) as bkg_samples,cast(avg(signal_pred_samples) as integer) as signal_pred_samples,cast(avg(bkg_pred_samples) as integer) as bkg_pred_samples, 100*round(avg(threshold),5) as threshold,  100*round(avg(sp),5) as sp, 100*round(avg(pd),5) as pd, 100*round(avg(pf),5) as pf, 100*round(avg(accuracy),5) as accuracy, 100*round(avg(f1),5) as f1, 100*round(avg(auc),5) as auc, 100*round(avg(precision),5) as precision, 100*round(avg(recall),5) as recall from classifiers where model = "'+model_name+'" and (sort,time,pf) in (select sort,time,min(pf) from classifiers where time = "'+script_time+'" and Point = "'+Point+'" and phase = "Validation" group by sort,time) group by point,model,time,phase,fine_tuning'

  result = db.query(trnquery)
  trnresult = db.query(trnquery)
  valresult = db.query(valquery)
  for row in result:
    chave= row.keys()

  for row in trnresult:
    trn= row #.values()

  for row in valresult:
    val= row #.values()

  for k in chave:
    x.add_row([k,trn[k],val[k]])

  return x

def create_reconstruction_table(model_name,script_time,normed='no'):
  #from SAE_Evaluation import *
  from prettytable import PrettyTable
  import sqlite3
  from decimal import Decimal
  cnx=sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
  #df = pd.read_sql_query("select point,sort,100*round(sp,4) as sp, 100*round(pd,4) as pd, 100*round(pf,4) as pf, 100*round(f1,4) as f1, 100*round(auc,4) as auc, 100*round(precision,4) as precision,100*round(recall,4) as recall from classifiers where model = '"+model_name+"' and time = '"+script_time+"' and phase = 'Validation'",cnx)
  df = pd.read_sql_query("select measure, ETotal from reconstruction_metrics  where class= 'All' and normed = '"+normed+"' and model = '"+model_name[:-1]+"' and time = '"+script_time+"' order by Measure",cnx)
  #df = pd.read_sql_query("select point,sort,100*round(sp,4) as sp, 100*round(pd,4) as pd, 100*round(pf,4) as pf, 100*round(f1,4) as f1, 100*round(auc,4) as auc, 100*round(precision,4) as precision,100*round(recall,4) as recall from classifiers where id in (select id from (select max(sp) as maxsp,id from classifiers where model = '"+model_name+"' and time = '"+script_time+"' and phase = 'Validation' group by Point,Model,HL_Neuron,time,sort,etBinIdx,etaBinIdx,phase,fine_tuning))",cnx)
  #df['ETotal'] = df['ETotal'].apply(lambda x: '%.4E' % Decimal(x))
  print df

  a = df.groupby(['Measure']).agg({'ETotal':['mean','std']}).values
  #df_values = np.round(a,2)
  x = PrettyTable()
  if normed=='yes':
      strnorm='(Normed)'
  else:
      strnorm=''

  x.field_names = ["Measure"+strnorm, "ETotal"]
  x.add_row(['Correlation', str(np.format_float_scientific(a[0][0], precision=3))+' '+str(np.format_float_scientific(a[0][1], precision=1))])
  x.add_row(['DeltaEnergy', str(np.format_float_scientific(a[1][0], precision=3))+' '+str(np.format_float_scientific(a[1][1], precision=1))])
  x.add_row(['KLDiv', str(np.format_float_scientific(a[2][0], precision=3))+' '+str(np.format_float_scientific(a[2][1], precision=1))])
  x.add_row(['MI', str(np.format_float_scientific(a[3][0], precision=3))+' '+str(np.format_float_scientific(a[3][1], precision=1))])
  x.add_row(['MSE', str(np.format_float_scientific(a[4][0], precision=3))+' '+str(np.format_float_scientific(a[4][1], precision=1))])
  x.add_row(['Norm DeltaEnergy', str(np.format_float_scientific(a[5][0], precision=3))+' '+str(np.format_float_scientific(a[5][1], precision=1))])
  x.add_row(['Norm MI', str(np.format_float_scientific(a[6][0], precision=3))+' '+str(np.format_float_scientific(a[6][1], precision=1))])
  x.add_row(['Wasserstein', str(np.format_float_scientific(a[7][0], precision=3))+' '+str(np.format_float_scientific(a[7][1], precision=1))])
  x.add_row(['KS', str(np.format_float_scientific(a[8][0], precision=3))+' '+str(np.format_float_scientific(a[8][1], precision=1))])
  return x

def create_reconstruction_table_complete(model_name,script_time,normed='no'):
  #from SAE_Evaluation import *
  from prettytable import PrettyTable
  import sqlite3
  from decimal import Decimal
  cnx=sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
  #df = pd.read_sql_query("select point,sort,100*round(sp,4) as sp, 100*round(pd,4) as pd, 100*round(pf,4) as pf, 100*round(f1,4) as f1, 100*round(auc,4) as auc, 100*round(precision,4) as precision,100*round(recall,4) as recall from classifiers where model = '"+model_name+"' and time = '"+script_time+"' and phase = 'Validation'",cnx)
  df = pd.read_sql_query("select measure, ETotal,EM,HAD from reconstruction_metrics  where class= 'All' and normed = '"+normed+"' and model = '"+model_name[:-1]+"' and time = '"+script_time+"' order by Measure",cnx)
  #df = pd.read_sql_query("select point,sort,100*round(sp,4) as sp, 100*round(pd,4) as pd, 100*round(pf,4) as pf, 100*round(f1,4) as f1, 100*round(auc,4) as auc, 100*round(precision,4) as precision,100*round(recall,4) as recall from classifiers where id in (select id from (select max(sp) as maxsp,id from classifiers where model = '"+model_name+"' and time = '"+script_time+"' and phase = 'Validation' group by Point,Model,HL_Neuron,time,sort,etBinIdx,etaBinIdx,phase,fine_tuning))",cnx)
  #df['ETotal'] = df['ETotal'].apply(lambda x: '%.4E' % Decimal(x))
  print df

  a = df.groupby(['Measure']).agg({'ETotal':['mean','std'],'EM':['mean','std'],'HAD':['mean','std']}).values
  #df_values = np.round(a,2)
  x = PrettyTable()
  if normed=='yes':
      strnorm='(Normed)'
  else:
      strnorm=''

  x.field_names = ["Measure"+strnorm, "ETotal", "EM", "HAD"]
  x.add_row(['Correlation', str(np.format_float_scientific(a[0][2], precision=1)),str(np.format_float_scientific(a[0][0], precision=1)),str(np.format_float_scientific(a[0][4], precision=1))])
  x.add_row(['DeltaEnergy', str(np.format_float_scientific(a[1][2], precision=1)),str(np.format_float_scientific(a[1][0], precision=1)),str(np.format_float_scientific(a[1][4], precision=1))])
  x.add_row(['KLDiv', str(np.format_float_scientific(a[2][2], precision=1)),str(np.format_float_scientific(a[2][0], precision=1)),str(np.format_float_scientific(a[2][4], precision=1))])
  x.add_row(['MI', str(np.format_float_scientific(a[3][2], precision=1)),str(np.format_float_scientific(a[3][0], precision=1)),str(np.format_float_scientific(a[3][4], precision=1))])
  x.add_row(['MSE', str(np.format_float_scientific(a[4][2], precision=1)),str(np.format_float_scientific(a[4][0], precision=1)),str(np.format_float_scientific(a[4][4], precision=1))])
  x.add_row(['Norm DeltaEnergy', str(np.format_float_scientific(a[5][2], precision=1)),str(np.format_float_scientific(a[5][0], precision=1)),str(np.format_float_scientific(a[5][4], precision=1))])
  x.add_row(['Norm MI', str(np.format_float_scientific(a[6][2], precision=1)),str(np.format_float_scientific(a[6][0], precision=1)),str(np.format_float_scientific(a[6][4], precision=1))])
  x.add_row(['Wasserstein', str(np.format_float_scientific(a[7][2], precision=1)),str(np.format_float_scientific(a[7][0], precision=1)),str(np.format_float_scientific(a[7][4], precision=1))])
  x.add_row(['KS', str(np.format_float_scientific(a[8][2], precision=1)),str(np.format_float_scientific(a[8][0], precision=1)),str(np.format_float_scientific(a[8][4], precision=1))])
  return x

def create_simple_table(model_name,script_time):
  #from SAE_Evaluation import *
  from prettytable import PrettyTable
  import sqlite3
  cnx=sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
  #df = pd.read_sql_query("select point,sort,100*round(sp,4) as sp, 100*round(pd,4) as pd, 100*round(pf,4) as pf, 100*round(f1,4) as f1, 100*round(auc,4) as auc, 100*round(precision,4) as precision,100*round(recall,4) as recall from classifiers where model = '"+model_name+"' and time = '"+script_time+"' and phase = 'Validation'",cnx)
  df = pd.read_sql_query("select point,sort,100*round(sp,4) as sp, 100*round(pd,4) as pd, 100*round(pf,4) as pf, round(mse,4) as mse, 100*round(f1,4) as f1, 100*round(auc,4) as auc, 100*round(precision,4) as precision,100*round(recall,4) as recall from classifiers where id in (select id from (select max(sp) as maxsp,id from classifiers where model = '"+model_name+"' and time = '"+script_time+"' and phase = 'Validation' group by Point,Model,HL_Neuron,time,sort,etBinIdx,etaBinIdx,phase,fine_tuning))",cnx)
  df['Point'] = df['Point'].apply(lambda x: x.split('_')[-1])
  a = df.groupby(['Point']).agg({'sp':['mean','std'],'pd':['mean','std'],'pf':['mean','std'],'mse':['mean','std'],'f1':['mean','std'],'auc':['mean','std'],'precision':['mean','std'],'recall':['mean','std']}).values
  #df_values = np.round(a,2)
  x = PrettyTable()

  x.field_names = ["Criteria", "Pd", "SP", "Fa","MSE", "F1","AUC","Precision","Recall"]
  x.add_row(["SP", str(round(a[2][6],2))+' '+str(round(a[2][7],2)),str(round(a[2][12],2))+' '+str(round(a[2][13],2)),str(round(a[2][4],2))+' '+str(round(a[2][5],2)),str(round(a[2][10],4))+' '+str(round(a[2][11],4)),str(round(a[2][0],2))+' '+str(round(a[2][1],2)),str(round(a[2][2],2))+' '+str(round(a[2][3],2)),str(round(a[2][14],2))+' '+str(round(a[2][15],2)),str(round(a[2][8],2))+' '+str(round(a[2][9],2))])
  x.add_row(["Pd", str(round(a[0][6],2))+' '+str(round(a[0][7],2)),str(round(a[0][12],2))+' '+str(round(a[0][13],2)),str(round(a[0][4],2))+' '+str(round(a[0][5],2)),str(round(a[0][10],4))+' '+str(round(a[0][11],4)),str(round(a[0][0],2))+' '+str(round(a[0][1],2)),str(round(a[0][2],2))+' '+str(round(a[0][3],2)),str(round(a[0][14],2))+' '+str(round(a[0][15],2)),str(round(a[0][8],2))+' '+str(round(a[0][9],2))])
  x.add_row(["Pf", str(round(a[1][6],2))+' '+str(round(a[1][7],2)),str(round(a[1][12],2))+' '+str(round(a[1][13],2)),str(round(a[1][4],2))+' '+str(round(a[1][5],2)),str(round(a[1][10],4))+' '+str(round(a[1][11],4)),str(round(a[1][0],2))+' '+str(round(a[1][1],2)),str(round(a[1][2],2))+' '+str(round(a[1][3],2)),str(round(a[1][14],2))+' '+str(round(a[1][15],2)),str(round(a[1][8],2))+' '+str(round(a[1][9],2))])

  ###fname = work_path+"files/"+tuning_folder_name+"/tuningMonitoring_et_2_eta_0.tex"
  ###with open(fname) as f:
  ###  content = f.readlines()
  ###f.close()

  ###x = PrettyTable()

  ###x.field_names = ["Criteria", "Pd", "SP", "Fa"]
  ###x.add_row(["Pd", content[244].split(' & ')[1].replace('\cellcolor[HTML]{9AFF99}','').replace('$\pm$',''), content[244].split(' & ')[2].replace('$\pm$',''), content[244].split(' & ')[3].replace('$\pm$','').replace(' \\','')])
  ###x.add_row(["SP", content[246].split(' & ')[1].replace('$\pm$',''), content[246].split(' & ')[2].replace('$\pm$',''), content[246].split(' & ')[3].replace('$\pm$','').replace(' \\','')])
  ###x.add_row(["Pf", content[248].split(' & ')[1].replace('$\pm$',''), content[248].split(' & ')[2].replace('$\pm$',''), content[248].split(' & ')[3].replace('\cellcolor[HTML]{BBDAFF}','').replace('$\pm$','').replace(' \\','')])
  ###x.add_row(["Reference", content[250].split(' & ')[1].replace('\cellcolor[HTML]{9AFF99}',''), content[250].split(' & ')[2], content[250].split(' & ')[3].replace('\cellcolor[HTML]{BBDAFF}','').replace(' \\','')])
  #bot.sendMessage('@ringer_tuning','Cross validation efficiencies for validation set. \n'+x.get_string())
  #bot.sendMessage('@ringer_tuning',x.get_string())
  ###x2 = PrettyTable()
  ###x2.field_names = ["Criteria", "Pd", "SP", "Fa"]
  ###x2.add_row(["Pd", content[265].split(' & ')[1], content[265].split(' & ')[2], content[265].split(' & ')[3].replace(' \\','')])
  ###x2.add_row(["SP", content[267].split(' & ')[1], content[267].split(' & ')[2], content[267].split(' & ')[3].replace(' \\','')])
  ###x2.add_row(["Pf", content[269].split(' & ')[1], content[269].split(' & ')[2], content[269].split(' & ')[3].replace(' \\','')])

  ###bot.sendMessage('@ringer_tuning','*Cross validation efficiencies for validation set.* \n'+x.get_string()+'\n*Operation efficiencies for the best model.* \n'+x2.get_string(),parse_mode='Markdown')
  #bot.sendMessage('@ringer_tuning',x2.get_string())
  #x3 = PrettyTable()
  #x3.field_names = list(trnMetrics.keys())
  #x3.add_row(list(trnMetrics.values()))
  #x3.add_row(list(valMetrics.values()))
  #bot.sendMessage('@ringer_tuning',x3.get_string())
  return x

def plot_Roc(fname,dirout, model_name=""):
  import os
  from RingerCore import load
  from sklearn.metrics import roc_curve, auc
  history_files=[x for x in os.listdir(fname) if x.endswith(".pic.gz")]
  png_files=[]

  fig, axs = plt.subplots(1, 1)
  plt.figure(1)

  list_t=[]
  #fig, axs = plt.subplots(1, 2, figsize=(24, 18))
  #fig, axs = plt.subplots(1, 1, figsize=(24, 18))
  #plt.figure(1)
  #@@plt.subplot(121)
  #@@for idx,file in enumerate(history_files):
    #@@disc=load(fname+'/'+file)
    #files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
    #@@pds=disc['tunedDiscr'][0][0]['summaryInfo']['roc_operation']['pds']
    #@@pfs=disc['tunedDiscr'][0][0]['summaryInfo']['roc_operation']['pfs']
    #@@sps=disc['tunedDiscr'][0][0]['summaryInfo']['roc_operation']['sps']
    #@@idxSP = np.argmax(sps)
    #@@sp=sps[idxSP]
    #@@roc_auc = auc(pfs,pds)
    #@@plt.plot(pfs, pds,label='ROC - AUC = '+str(round(roc_auc,4))+', SP = '+str(100*round(sp,4))+' - Sorteio '+str(idx+1)+' ' % roc_auc)
  #@@plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  #@@plt.xlim([0.0, 1.0])
  #@@plt.ylim([0.0, 1.05])
  #@@plt.xlabel('Probabilidade de Falso Positivo',fontsize= 'xx-large')
  #@@plt.ylabel(r'Probabilidade de $Detec\c{}\~ao$ ',fontsize= 'xx-large')
  #@@plt.title(model_name+' Curva ROC - Treino',fontsize= 'xx-large')
  #@@plt.legend(loc="lower right")

  #plt.subplot(111)
  for idx,file in enumerate(history_files):
    disc=load(fname+'/'+file)
    #files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
    pds=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['pds']
    pfs=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['pfs']
    sps=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['sps']
    ths=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['thresholds']
    idxSP = np.argmax(sps)
    sp=sps[idxSP]
    pd=pds[idxSP]
    pf=pfs[idxSP]
    print ths[idxSP]
    roc_auc = auc(pfs,pds)
    plt.plot(pfs, pds,label='ROC - AUC='+str(round(roc_auc,4))+', SP='+str(100*round(sp,4))+', Pd='+str(100*round(pd,4))+', Pf='+str(100*round(pf,4))+' - Sorteio '+str(idx+1)+' ' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  plt.xlim([0.0, 0.4])
  plt.ylim([0.6, 1.05])
  plt.xlabel('Probabilidade de Falso Positivo',fontsize= 'xx-large')
  plt.ylabel(r'Probabilidade de $Detec\c{}\~ao$ ',fontsize= 'xx-large')
  plt.title(model_name+' Curva ROC - Teste',fontsize= 'xx-large')
  plt.legend(loc="lower right")

  plt.savefig(dirout+'roc_'+fname.split('/')[-1]+'.png')
  plt.clf()
  plt.close()
  png_files.append(dirout+'roc_'+fname.split('/')[-1]+'.png')
  return png_files

def getLSTMReconstruct(norm1Par,sort,model_name=None):
  from audeep.backend.training.base import BaseFeatureLearningWrapper
  from audeep.backend.training.time_autoencoder import TimeAutoencoderWrapper
  import numpy as np
  from pathlib import Path
  afternorm = norm1Par[2]
  reconstruct= OrderedDict()
  target=[]
  wrapper = TimeAutoencoderWrapper()
  if isinstance(afternorm, (tuple, list,)):
    predict = []
    target = None #[]
    for i, cdata in enumerate(afternorm):
      print i,cdata.shape
      reconstruction = wrapper.generate_np_reconstruction(model_filename=Path(model_name),
                                                          global_step=None,
                                                          data_set=cdata,
                                                          batch_size=10000)
      #lstm_target = wrapper.generate_np_targets(model_filename=Path(model_name),
      #                                                    global_step=None,
      #                                                    data_set=cdata,
      #                                                    batch_size=10000)
      #model_predict = model.predict(cdata, batch_size=cdata.shape[0], verbose=2)
      #print 'what now?'
      predict.append(reconstruction)
      #target.append(lstm_target)
      print 'Reconstruction Done'
      bottleneck=reconstruction.shape[1]
  reconstruct[bottleneck]=predict
  return reconstruct,target

def getPCAReconstruct(data=None,model=None, means=None):
  reconstruct = OrderedDict()
  if isinstance(data, (tuple, list,)):
    ret = []
    for i, cdata in enumerate(data):
      print i,cdata.shape
      ret.append( model.inverse_transform(cdata) ) #+ means )
  else:
    ret = model.inverse_transform(cdata) + means
  reconstruct[data[0].shape[1]] = ret
  return reconstruct

def getNLPCAReconstruct(fname,data,sort,layer):
  reconstruct = OrderedDict()
  with open(fname) as f:
    content = f.readlines()
  f.close()
  #bottleneck = int(layer.split('x')[1])
  files = [f for f in content if f.split('_')[9] == str(sort)]
  ifile=files[0]
  model_load = load_model('//home/caducovas/run/nlpca_preproc/output_files/'+ifile.replace('\n','')+'.h5')
  get_layer_output = K.function([model_load.layers[0].input],
                              [model_load.layers[7].output])
  if isinstance(data, (tuple, list,)):
    ret = []
    for i, cdata in enumerate(data):
      ret.append( get_layer_output([cdata])[0] )
  else:
    ret = get_layer_output([data])[0]
  reconstruct[layer] = ret
  return reconstruct

def getPCACode(data=None):
  code = OrderedDict()
  code[data[0].shape[1]] = data
  return code

def getNLPCACode(data=None):
  code = OrderedDict()
  code[data[0].shape[1]] = data
  return code

def getReconstruct(fname,data,sort):
  #from SAE_Evaluation import *

  predict_data = {}
  reconstruct = OrderedDict()
  modelo={}
  enc_model={}
  dec_model={}

  #if K.backend() == 'tensorflow':
  #    K.clear_session()

  with open(fname) as f:
    content = f.readlines()
  f.close()
  layers_list =[(f.split('/')[-1].split('_')[24],f.split('/')[-1].split('_')[33]) for f in content]
  #layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  layers,layers_numbers=[x for x,y in sorted(list(set(layers_list)),cmp=layer2number2)],[y for x,y in sorted(list(set(layers_list)),cmp=layer2number2)]
  #layers=sorted(list(set(layers_list)),cmp=layer2number2)
  print layers,layers_numbers
  #dirin='/home/caducovas/DeepRinger/data/run_layer1/adam_80/'
  #layers = ['100x80','80x60','60x40','40x10']
  #nsorts=10

  #for i in [len(layers)]:
  for i in range(len(layers)):
    nlayers=i+1
    layers_list=layers[:nlayers]
    print range(len(layers)),nlayers,layers_list

    predict_data = {} ##predict data junta os sortes

    #for isort in range(nsorts):
    for isort in [sort]:
      enc_model={}
      dec_model={}
      print "Sort: "+str(isort)

      #Itera sobre os layers para adquirir o encoder e o decoder
      for iLayer,layer in enumerate(layers_list): #Different archtectures (each time one more autoencoder)
        #print "Reading files of: "+layer

        neuron = int(layer.split('x')[1])
        files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer and f.split('/')[-1].split('_')[27] == str(isort) and f.split('/')[-1].split('_')[33] == layers_numbers[iLayer])]
        ifile=files[0]
        #print ifile
        custom_obj={}
        if 'CAE' in fname:
          from TuningTools.MetricsLosses import contractive_loss
          par_list=ifile.split('/')[-1].split('_')
          custom_obj['contractive_loss']=contractive_loss(int(par_list[24].split('x')[1]),int(par_list[24].split('x')[0]),par_list[10],par_list[13])
          modelo = load_model(ifile.replace('\n','')+'_model.h5',custom_objects=custom_obj)
        else:
          modelo = load_model(ifile.replace('\n','')+'_model.h5')
        #modelo = load_model(dirin+ifile)
        enc_model[layer] = modelo.layers[0].get_weights()
        dec_model[layer] = modelo.layers[2].get_weights()

      #print "Creating the model"
      print len(data) #[1].shape[1]
      model = Sequential()
      print "just to make sure it is the first key "+list(enc_model.keys())[0]
      first_layer = [k for k in list(enc_model.keys()) if str(data[0].shape[1])+'x' in k][0]
      model.add(Dense(int(layers_list[0].split('x')[1]), input_dim=data[0].shape[1], weights=enc_model[first_layer]))
      model.add(Activation('tanh'))

      if(nlayers >1):
        ## Add encoders
        for layer in layers_list[1:]:
          neuron = int(layer.split('x')[1])
          model.add(Dense(neuron, weights=enc_model[layer]))
          model.add(Activation('tanh'))
      ## Add decoders
      for layer in reversed(layers_list):
        print layer
        neuron = int(layer.split('x')[0])
        model.add(Dense(neuron, weights=dec_model[layer]))
        model.add(Activation('tanh'))

      model.pop()
      print model.summary()
      model.compile('adam','mse')

      ###################
      bottleneck=int(layers_list[-1].split('x')[1])
      afternorm = data
      print type(afternorm)
      print len(afternorm)
      if isinstance(afternorm, (tuple, list,)):
        predict = []
        for i, cdata in enumerate(afternorm):
          print i,cdata.shape
          model_predict = model.predict(cdata, batch_size=cdata.shape[0], verbose=2)
          print 'what now?'
          predict.append(model_predict)
      print 'Predictions Done'
      #print isort
      #@@predict_data[int(isort)] = predict
      #print predict_data
      #@@reconstruct[bottleneck] = predict_data
      reconstruct[bottleneck] = predict
      #print predict[0].shape,predict[1].shape
  return reconstruct
  #if K.backend() == 'tensorflow':
  #    K.clear_session()

def getCode(fname,norm1Par,sort):
  #from SAE_Evaluation import *

  predict_data = {}
  code = OrderedDict()
  modelo={}
  enc_model={}
  dec_model={}

  #if K.backend() == 'tensorflow':
  #    K.clear_session()

  with open(fname) as f:
    content = f.readlines()
  f.close()
  layers_list =[(f.split('/')[-1].split('_')[24],f.split('/')[-1].split('_')[33]) for f in content]
  #layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  layers,layers_numbers=[x for x,y in sorted(list(set(layers_list)),cmp=layer2number2)],[y for x,y in sorted(list(set(layers_list)),cmp=layer2number2)]
  #layers=sorted(list(set(layers_list)),cmp=layer2number2)
  print layers,layers_numbers
  #dirin='/home/caducovas/DeepRinger/data/run_layer1/adam_80/'
  #layers = ['100x80','80x60','60x40','40x10']
  #nsorts=10

  #for i in [len(layers)]:
  for i in range(len(layers)):
    nlayers=i+1
    layers_list=layers[:nlayers]
    print range(len(layers)),nlayers,layers_list

    predict_data = {} ##predict data junta os sortes

    #for isort in range(nsorts):
    for isort in [sort]:
      enc_model={}
      dec_model={}
      print "Sort: "+str(isort)

      #Itera sobre os layers para adquirir o encoder e o decoder
      for iLayer,layer in enumerate(layers_list): #Different archtectures (each time one more autoencoder)
        #print "Reading files of: "+layer

        neuron = int(layer.split('x')[1])
        files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer and f.split('/')[-1].split('_')[27] == str(isort) and f.split('/')[-1].split('_')[33] == layers_numbers[iLayer])]
        ifile=files[0]
        #print ifile
        custom_obj={}
        if 'CAE' in fname:
          from TuningTools.MetricsLosses import contractive_loss
          par_list=ifile.split('/')[-1].split('_')
          custom_obj['contractive_loss']=contractive_loss(int(par_list[24].split('x')[1]),int(par_list[24].split('x')[0]),par_list[10],par_list[13])
          modelo = load_model(ifile.replace('\n','')+'_model.h5',custom_objects=custom_obj)
        else:
          modelo = load_model(ifile.replace('\n','')+'_model.h5')
        #modelo = load_model(dirin+ifile)
        enc_model[layer] = modelo.layers[0].get_weights()
        dec_model[layer] = modelo.layers[2].get_weights()

      #print "Creating the model"
      model = Sequential()
      print "just to make sure it is the first key "+list(enc_model.keys())[0]
      first_layer = [k for k in list(enc_model.keys()) if str(norm1Par[2][0].shape[1])+'x' in k][0]
      model.add(Dense(int(layers_list[0].split('x')[1]), input_dim=norm1Par[2][0].shape[1], weights=enc_model[first_layer]))
      model.add(Activation('tanh'))
      if(nlayers >1):
        ## Add encoders
        for layer in layers_list[1:]:
          neuron = int(layer.split('x')[1])
          model.add(Dense(neuron, weights=enc_model[layer]))
          model.add(Activation('tanh'))
      # ## Add decoders
      # for layer in reversed(layers_list):
        # print layer
        # neuron = int(layer.split('x')[0])
        # model.add(Dense(neuron, weights=dec_model[layer]))
      # model.add(Activation('tanh'))

      print model.summary()
      model.compile('adam','mse')

      ###################
      bottleneck=int(layers_list[-1].split('x')[1])
      afternorm = norm1Par[2]
      print type(afternorm)
      print len(afternorm)
      if isinstance(afternorm, (tuple, list,)):
        predict = []
        for i, cdata in enumerate(afternorm):
          print i,cdata.shape
          model_predict = model.predict(cdata, batch_size=cdata.shape[0], verbose=2)
          print 'what now?'
          predict.append(model_predict)
      print 'Predictions Done'
      #print isort
      #@@predict_data[int(isort)] = predict
      #print predict_data
      #@@reconstruct[bottleneck] = predict_data
      code[bottleneck] = predict
      #print predict[0].shape,predict[1].shape
  return code
  #if K.backend() == 'tensorflow':
  #    K.clear_session()

def getReconstructFT(fname,data,sort,etBinIdx,etaBinIdx,bottleneck):
  #from SAE_Evaluation import *

  predict_data = {}
  reconstruct = OrderedDict()

  model = load_model(fname+'/models/fine_tuned_sort_'+str(sort)+'_et_'+str(etBinIdx)+'_eta_'+str(etaBinIdx)+'.h5')

  afternorm = data
  print type(afternorm)
  print len(afternorm)
  if isinstance(afternorm, (tuple, list,)):
    predict = []
    for i, cdata in enumerate(afternorm):
      print i,cdata.shape
      model_predict = model.predict(cdata, batch_size=cdata.shape[0], verbose=2)
      print 'what now?'
      predict.append(model_predict)
  print 'Predictions Done'

  reconstruct[bottleneck] = predict
      #print predict[0].shape,predict[1].shape
  return reconstruct

def getCodeFT(fname,data,sort,etBinIdx,etaBinIdx,bottleneck):
    #from SAE_Evaluation import *

    predict_data = {}
    code = OrderedDict()

    model = load_model(fname+'/models/fine_tuned_sort_'+str(sort)+'_et_'+str(etBinIdx)+'_eta_'+str(etaBinIdx)+'.h5')
    encoderLayerNum= int((float(len(model.layers))/2)-1)
    if isinstance(data, (tuple, list,)):
      predict = []
      for cdata in data:
        #self._info(cdata.shape)
        get_layer_output = K.function([model.layers[0].input],
                                    [model.layers[encoderLayerNum].output])
        # Projection of layer
        proj_all_data = get_layer_output([cdata])[0]

        predict.append(proj_all_data)
    print 'Predictions Done'

    code[bottleneck] = predict
        #print predict[0].shape,predict[1].shape
    return code
  #if K.backend() == 'tensorflow':
  #    K.clear_session()

def getSAE_Model(fname,data,sort):
  #from SAE_Evaluation import *


  modelo={}
  enc_model={}
  dec_model={}

  #if K.backend() == 'tensorflow':
  #    K.clear_session()

  with open(fname) as f:
    content = f.readlines()
  f.close()
  layers_list =[(f.split('/')[-1].split('_')[24],f.split('/')[-1].split('_')[33]) for f in content]
  #layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  layers,layers_numbers=[x for x,y in sorted(list(set(layers_list)),cmp=layer2number2)],[y for x,y in sorted(list(set(layers_list)),cmp=layer2number2)]
  #layers=sorted(list(set(layers_list)),cmp=layer2number2)
  print layers,layers_numbers
  #dirin='/home/caducovas/DeepRinger/data/run_layer1/adam_80/'
  #layers = ['100x80','80x60','60x40','40x10']
  #nsorts=10

  #for i in [len(layers)]:
  #for i in range(len(layers)):
  nlayers=len(layers)
  layers_list=layers #[:nlayers]
  #print range(len(layers)),nlayers,layers_list
  predict_data = {} ##predict data junta os sortes
  #for isort in range(nsorts):
  for isort in [sort]:
    enc_model={}
    dec_model={}
    print "Sort: "+str(isort)
    #Itera sobre os layers para adquirir o encoder e o decoder
    for iLayer,layer in enumerate(layers_list): #Different archtectures (each time one more autoencoder)
      #print "Reading files of: "+layer
      neuron = int(layer.split('x')[1])
      files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer and f.split('/')[-1].split('_')[27] == str(isort) and f.split('/')[-1].split('_')[33] == layers_numbers[iLayer])]
      ifile=files[0]
      #print ifile
      custom_obj={}
      if 'CAE' in fname:
        from TuningTools.MetricsLosses import contractive_loss
        par_list=ifile.split('/')[-1].split('_')
        custom_obj['contractive_loss']=contractive_loss(int(par_list[24].split('x')[1]),int(par_list[24].split('x')[0]),par_list[10],par_list[13])
        modelo = load_model(ifile.replace('\n','')+'_model.h5',custom_objects=custom_obj)
      else:
        modelo = load_model(ifile.replace('\n','')+'_model.h5')
        #modelo = load_model(dirin+ifile)
      enc_model[layer] = modelo.layers[0].get_weights()
      dec_model[layer] = modelo.layers[2].get_weights()

    #print "Creating the model"
    print len(data) #[1].shape[1]
    model = Sequential()
    print "just to make sure it is the first key "+list(enc_model.keys())[0]
    first_layer = [k for k in list(enc_model.keys()) if str(data[0].shape[1])+'x' in k][0]
    model.add(Dense(int(layers_list[0].split('x')[1]), input_dim=data[0].shape[1], weights=enc_model[first_layer]))
    model.add(Activation('tanh'))
    if(nlayers >1):
      ## Add encoders
      for layer in layers_list[1:]:
        neuron = int(layer.split('x')[1])
        model.add(Dense(neuron, weights=enc_model[layer]))
        model.add(Activation('tanh'))
    ## Add decoders
    for layer in reversed(layers_list):
      print layer
      neuron = int(layer.split('x')[0])
      model.add(Dense(neuron, weights=dec_model[layer]))
      model.add(Activation('tanh'))
    model.pop()
    model.add(Activation('linear'))
    print model.summary()

    return model

def plot_input_reconstruction(model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
  import sqlite3
  import pandas as pd
  from numpy import nan
  #%matplotlib inline
  import matplotlib.pyplot as plt
  png_files=[]

  plt.style.use('ggplot')
  cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
  # Et and Eta indices
  et_index  = [0, 1, 2,3]
  etRange = ['[15, 20]','[20, 30]','[30, 40]','[40, 50000]']

  eta_index = [0, 1, 2, 3, 4,5,6,7,8]
  etaRange = ['[0, 0.6]','[0.6, 0.8]','[0.8, 1.15]','[1.15, 1.37]','[1.37, 1.52]','[1.52, 1.81]','[1.81, 2.01]','[2.01, 2.37]','[2.37, 2.47]']

  #et_index  = [1,2]
  #etRange = ['[20, 30]']

  #eta_index = [1,2]
  #etaRange = ['[0.6, 0.8]']

  #for iet, etrange in zip(et_index, etRange):
  #  for ieta, etarange in zip(eta_index, etaRange):
  iet =  etBinIdx
  etrange = etRange[etBinIdx]
  ieta = etaBinIdx
  etarange = etaRange[etaBinIdx]
      #sgn = data_file['signalPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
      #bkg = data_file['backgroundPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]

  dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'All' and Measure = 'Normalized_MI' and Normed='no'  and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  dfAll.fillna(value=nan, inplace=True)

  dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'Signal' and Measure = 'Normalized_MI' and Normed='no'  and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  #dfSignal.fillna(value=nan, inplace=True)

  dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'Background' and Measure = 'Normalized_MI' and Normed='no'  and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  #dfBkg.fillna(value=nan, inplace=True)


  allClasses=dfAll.values.astype(np.float32)
  sgn=dfSignal.values.astype(np.float32)
  bkg=dfBkg.values.astype(np.float32)

  #print 'all',allClasses
  #print 'sgn', sgn
  #print 'bkg', bkg

  plt.figure(figsize=(16,10))
  plt.errorbar(np.arange(allClasses.shape[1]), np.mean(allClasses, axis=0),yerr=np.std(allClasses, axis=0), fmt='go-',color='green')
  plt.errorbar(np.arange(allClasses.shape[1]), np.mean(sgn, axis=0),yerr=np.std(sgn, axis=0), fmt='D-', color='cornflowerblue')
  plt.errorbar(np.arange(allClasses.shape[1]), np.mean(bkg, axis=0),yerr=np.std(bkg, axis=0), fmt='ro-')
  #print np.mean(allClasses,axis=0),np.std(allClasses,axis=0)
  #print np.mean(sgn,axis=0),np.std(sgn,axis=0)
  #print np.mean(bkg,axis=0),np.std(bkg,axis=0)


  plt.legend(['All','Signal','Background'], loc='best', fontsize='xx-large')
  for i in [7, 71, 79, 87, 91, 95]:
    plt.axvline(i, color='gray', linestyle='--', linewidth=.8)

  plt.title(r''+model_name+' - Layer '+str(layer)+' - MI Input X Reconstruction $E_T$={} $\eta$={}'.format(etrange,etarange),fontsize= 20)
  plt.xlabel('#Rings', fontsize='xx-large')
  plt.ylabel('Normalized Mutual Information', fontsize='xx-large')
  plt.ylim(ymax=1)
  if log_scale:
    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
  else:
    y_position = .98#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    plt.text(x,y,text, fontsize=15, rotation=90)

  #plt.show()
  plt.savefig(dirout+'input_reconstruction_mi.png')
  png_files.append(dirout+'input_reconstruction_mi.png')
  plt.clf()
  plt.close()
  return png_files

def plot_confusion_matrix(cm, classes,
              normalize=False,
              title='Confusion matrix',
              cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  #plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.4f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
         horizontalalignment="center",
         color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  #plt.clf()
  #plt.close()

def send_confusion_matrix(fname,dirout,model,y_test,y_pred,points):
  refName,point = points
  print refName
  print point.thres_value
  #print 'Pd:',point.pd,' SP:',point.sp,' Pf:',point.pf
  y_pred[y_pred >= point.thres_value] = 1
  y_pred[y_pred < point.thres_value] = -1
  png_files = []
  class_names=['Background','Signal']
  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, y_pred)
  np.set_printoptions(precision=2)
  # Plot non-normalized confusion matrix
  #fig,axs = plt.subplots(1,2)
  plt.figure(figsize=(9,6))
  plt.subplot(1,2,1)
  plot_confusion_matrix(cnf_matrix, classes=class_names,
              title='Confusion matrix')
  # Plot normalized confusion matrix
  plt.subplot(1,2,2)
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
              title='Normalized confusion matrix')
  #plt.show()
  plt.tight_layout()
  plt.suptitle(refName.split('_')[-1]+" - "+model, fontsize=24)
  plt.savefig(dirout+'/confusion_matrix_'+fname.split('/')[-1]+'.png')
  plt.clf()
  plt.close()
  png_files.append(dirout+'/confusion_matrix_'+fname.split('/')[-1]+'.png')
  return png_files

def feature_relevance_mse(fname,dirout,tb_name,valdata,model,y_test,y_pred,points):
  # refName,point = points
  # print refName
  # print point.thres_value
  # #print 'Pd:',point.pd,' SP:',point.sp,' Pf:',point.pf
  # y_pred[y_pred >= point.thres_value] = 1
  # y_pred[y_pred < point.thres_value] = -1
  png_files = []
  # class_names=['Background','Signal']
  # # Compute confusion matrix
  # cnf_matrix = confusion_matrix(y_test, y_pred)
  # np.set_printoptions(precision=2)
  # # Plot non-normalized confusion matrix
  # #fig,axs = plt.subplots(1,2)
  # plt.figure(figsize=(9,6))
  # plt.subplot(1,2,1)
  # plot_confusion_matrix(cnf_matrix, classes=class_names,
  #             title='Confusion matrix')
  # # Plot normalized confusion matrix
  # plt.subplot(1,2,2)
  # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
  #             title='Normalized confusion matrix')
  model = load_model(fname+'/models/'+tb_name+'.h5')
  #valdata = np.load(fname+'results/Val_sort0_et_1_eta_1.npz')
  relevance_mse =[]
  import copy
  #data = copy.deepcopy(trnData)
  #print valdata
  for i in range(valdata.shape[1]):
      val = copy.deepcopy(valdata)
      #print i, val.shape[0],np.mean(val[:,i])
      val[:,i] = [np.mean(val[:,i])]*val.shape[0]
      valOutput = model.predict(valdata)
      valOutputModified = model.predict(val)
      r_mse = float(np.sum((valOutput - valOutputModified) ** 2))/float(val.shape[0])
      relevance_mse.append(r_mse)
      #print np.sum((valOutput - valOutputModified) ** 2)

  fig, ax = plt.subplots(figsize=(16,10))
  plt.xticks(np.arange(len(relevance_mse)),rotation='vertical',fontsize = 9)
  plt.bar(np.arange(len(relevance_mse)),relevance_mse)
  #plt.tight_layout()
  plt.margins(0)
  plt.grid()
  ax.set_title(r'Feature Contribution to Model Reconstruction Error',fontsize= 20)
  ax.set_xlabel('Feature', fontsize='xx-large')
  ax.set_ylabel(r'1/N sum_j(O(x_j) - O(x_j | x_j,i =x_medio))**2', fontsize='xx-large')

  #if valdata
  #ax[0,0].set_yscale('log')
  # for i in [7, 71, 79, 87, 91, 95]:
  #     ax.axvline(i, color='blue', linestyle='--', linewidth=.8)
  # y_position = .9 *np.max([np.max(relevance_mse, axis=0)])
  # for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),(76,y_position,r'EM2'),(80,y_position,r'EM3'),(88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
  #     ax.text(x,y,text, fontsize=15, rotation=90)

  plt.savefig(dirout+'/feature_relevance_mse_'+fname.split('/')[-1]+'.png')
  plt.clf()
  plt.close()
  png_files.append(dirout+'/feature_relevance_mse_'+fname.split('/')[-1]+'.png')
  return png_files

def feature_relevance_sp(fname,dirout,tb_name,valdata,model,y_test,y_pred,points):
    # refName,point = points
    # print refName
    # print point.thres_value
    # #print 'Pd:',point.pd,' SP:',point.sp,' Pf:',point.pf
    # y_pred[y_pred >= point.thres_value] = 1
    # y_pred[y_pred < point.thres_value] = -1
    png_files = []

    model = load_model(fname+'/models/'+tb_name+'.h5')
     #valdata = np.load(fname+'results/Val_sort0_et_1_eta_1.npz')

    from sklearn.metrics import roc_curve

    relevance_sp =[]
    import copy
    #data = copy.deepcopy(trnData)
    #print valdata
    for i in range(valdata.shape[1]):
        val = copy.deepcopy(valdata)
        #print i, val.shape[0],np.mean(val[:,i])
        val[:,i] = [np.mean(val[:,i])]*val.shape[0]
        valOutput = model.predict(valdata)
        pfs,pds,thresholds = roc_curve(y_test,valOutput, pos_label=1, drop_intermediate=True)
        pds = pds
        bps = 1. - pfs
        sps = np.sqrt( ( pds  + bps )*.5 * np.sqrt( pds * bps ) )
        idxSP = np.argmax(sps)
        sp=sps[idxSP]

        valOutputModified = model.predict(val)
        pfs,pds,thresholds = roc_curve(y_test,valOutputModified, pos_label=1, drop_intermediate=True)
        pds = pds
        bps = 1. - pfs
        sps_Modified = np.sqrt( ( pds  + bps )*.5 * np.sqrt( pds * bps ) )
        idxSP_Modified = np.argmax(sps_Modified)
        sp_Modified=sps_Modified[idxSP_Modified]

        r_sp = sp - sp_Modified
        r_sp = 100*round(r_sp,4)
        relevance_sp.append(r_sp)

    fig, ax = plt.subplots(figsize=(16,10))
    plt.xticks(np.arange(len(relevance_sp)),rotation='vertical',fontsize = 9)
    plt.bar(np.arange(len(relevance_sp)),relevance_sp)
    #plt.tight_layout()
    plt.margins(0)
    plt.grid()
    ax.set_title(r'Feature Contribution to SP',fontsize= 20)
    ax.set_xlabel('Feature', fontsize='xx-large')
    ax.set_ylabel(r'SP(x_j) - SP(x_j | x_j,i =x_medio)', fontsize='xx-large')

      #if valdata
      #ax[0,0].set_yscale('log')
      # for i in [7, 71, 79, 87, 91, 95]:
      #     ax.axvline(i, color='blue', linestyle='--', linewidth=.8)
      # y_position = .9 *np.max([np.max(relevance_mse, axis=0)])
      # for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),(76,y_position,r'EM2'),(80,y_position,r'EM3'),(88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
      #     ax.text(x,y,text, fontsize=15, rotation=90)

    plt.savefig(dirout+'/feature_relevance_sp_'+fname.split('/')[-1]+'.png')
    plt.clf()
    plt.close()
    png_files.append(dirout+'/feature_relevance_sp_'+fname.split('/')[-1]+'.png')
    return png_files

def reconstruct_performance(norm1Par=None,reconstruct=None,model_name="",time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=None,lstm_target=None,measure='Normalized_MI',Normed=False):
  #from SAE_Evaluation import *
  from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
  from sklearn.metrics import mean_squared_error
  from scipy.stats import wasserstein_distance
  from scipy.stats import ks_2samp
  import dataset
  import math
  db = dataset.connect('sqlite://///home/caducovas/run/ringerMLlab.db')
  #print point.sp_value
  table = db['reconstruction_metrics']
  metrics = OrderedDict()
  beforenorm = norm1Par[0]
  normlist = norm1Par[1]
  afternorm = norm1Par[2]
  #measure=#Normalized_MI,MI,KLdiv,chiSquared,Correlation
  for layer in reconstruct.keys():
    print 'LAYER: '+str(layer)
  #for nsort in reconstruct[layer].keys():
  #print "Sort: "+str(nsort)
    if isinstance(reconstruct[layer], (tuple, list,)):
      unnorm_reconstruct = []
      for i, cdata in enumerate(reconstruct[layer]):
        #print i,cdata.shape
        unnorm_reconstruct.append( cdata * normlist[i])

      ###if sort == 0:
      ###  np.savez_compressed('/scratch/22061a/caducovas/run/reconstruction_files/recEnergy_'+model_name+'_'+str(layer),iEnergy=beforenorm,rEnergy=unnorm_reconstruct)
      ###  np.savez_compressed('/scratch/22061a/caducovas/run/reconstruction_files/Normed_RecEnergy_'+model_name+'_'+str(layer),iEnergy=afternorm,rEnergy=reconstruct[layer])
      unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
      beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

      reconstruct_val_Data = np.concatenate( reconstruct[layer], axis=0 )
      afternorm_val_Data = np.concatenate( afternorm, axis=0 )

      if Normed:
        input_val_Data=afternorm_val_Data
        reconstruct_val_Data=reconstruct_val_Data
        input=afternorm
        reconstructed=reconstruct[layer]
        normalizacao='yes'
      else:
        input_val_Data=beforenorm_val_Data
        reconstruct_val_Data=unnorm_reconstruct_val_Data
        input=beforenorm
        reconstructed=unnorm_reconstruct
        normalizacao='no'

      ##ALL LABELS

      ##### MI/KL
      metrics['Class'] = 'All'
      metrics['Model'] = model_name
      metrics['Layer'] = str(layer)
      metrics['time'] = time
      metrics['Measure'] = measure
      metrics['Normed'] = normalizacao
      #metrics['HL_Neuron'] = hl_neuron
      metrics['sort'] = sort
      metrics['etBinIdx'] = etBinIdx
      metrics['etaBinIdx'] = etaBinIdx
      metrics['phase'] = phase
      #metrics['Elapsed'] = elapsed
      #metrics['fine_tuning'] = fine_tuning

      for anel in range(100):
        ###print anel
        #print beforenorm_val_Data[:,anel].shape,unnorm_reconstruct_val_Data[:,anel].shape
        try:
          if measure == 'Normalized_MI':
            rr = calc_MI2(input_val_Data[:,anel],reconstruct_val_Data[:,anel])
            score = np.sqrt(1. - np.exp(-2 * rr))
          elif measure == 'MI':
            score = calc_MI2(input_val_Data[:,anel],reconstruct_val_Data[:,anel])
          elif measure == 'KLdiv':
            score = calc_kl(input_val_Data[:,anel],reconstruct_val_Data[:,anel])
          elif measure == 'chiSquared':
            score,chi_pvalue =calc_chisquare(input_val_Data[:,anel],reconstruct_val_Data[:,anel])
          elif measure == 'Correlation':
            score,corr_pvalue= scipy.stats.pearsonr(input_val_Data[:,anel],reconstruct_val_Data[:,anel])
          elif measure == 'MSE':
            score= float(mean_squared_error(input_val_Data[:,anel],reconstruct_val_Data[:,anel]))
          elif measure == 'Wasserstein':
            score= float(wasserstein_distance(input_val_Data[:,anel],reconstruct_val_Data[:,anel]))
          elif measure == 'kolmogorov-smirnov':
            score, pvalue= ks_2samp(input_val_Data[:,anel],reconstruct_val_Data[:,anel])
          elif measure == 'DeltaEnergy':
            score= float(np.average(input_val_Data[:,anel]-reconstruct_val_Data[:,anel], axis=0))
          elif measure == 'Normalized_DeltaEnergy':
            score= float(np.average((input_val_Data[:,anel]-reconstruct_val_Data[:,anel])/input_val_Data[:,anel], axis=0))
          if math.isnan(score):
            score = None
          metrics[str(anel+1)] = score
          #print score
        except:
          print 'Anel '+str(anel)+' apresenta erros de calculo'
          metrics[str(anel+1)] = None
          #print ks_2samp(input_val_Data[:,anel],reconstruct_val_Data[:,anel])


      try:
      #for qqq in range(1):
        if measure == 'Normalized_MI':
          ###TOTAL
          rr = calc_MI2(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1))
          metrics['ETotal'] = np.sqrt(1. - np.exp(-2 * rr))
          ###PS
          rr = calc_MI2(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1))
          metrics['PS'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM1
          rr = calc_MI2(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1))
          metrics['EM1'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM2
          rr = calc_MI2(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1))
          metrics['EM2'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM3
          rr = calc_MI2(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1))
          metrics['EM3'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM
          rr = calc_MI2(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1))
          metrics['EM'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD1
          rr = calc_MI2(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1))
          metrics['HAD1'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD2
          rr = calc_MI2(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1))
          metrics['HAD2'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD3
          rr = calc_MI2(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1))
          metrics['HAD3'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD
          rr = calc_MI2(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1))
          metrics['HAD'] = np.sqrt(1. - np.exp(-2 * rr))
        elif measure == 'MI':
          ###TOTAL
          metrics['ETotal'] = calc_MI2(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1))
          ###PS
          metrics['PS'] = calc_MI2(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'] = calc_MI2(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'] = calc_MI2(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'] = calc_MI2(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'] = calc_MI2(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'] = calc_MI2(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'] = calc_MI2(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'] = calc_MI2(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'] = calc_MI2(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1))
        elif measure == 'KLdiv':
          #TOTAL
          metrics['ETotal'] = calc_kl(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1))
          ###PS
          metrics['PS'] = calc_kl(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'] = calc_kl(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'] = calc_kl(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'] = calc_kl(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'] = calc_kl(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'] = calc_kl(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'] = calc_kl(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'] = calc_kl(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'] = calc_kl(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1))
        elif measure == 'chiSquared':
          ###TOTAL
          metrics['ETotal'],chi_pvalue = calc_chisquare(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1))
          ###PS
          metrics['PS'],chi_pvalue = calc_chisquare(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'],chi_pvalue = calc_chisquare(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'],chi_pvalue = calc_chisquare(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'],chi_pvalue = calc_chisquare(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'],chi_pvalue = calc_chisquare(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'],chi_pvalue = calc_chisquare(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'],chi_pvalue = calc_chisquare(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'],chi_pvalue = calc_chisquare(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'],chi_pvalue = calc_chisquare(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1))
        elif measure == 'Correlation':
          ###TOTAL
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1))
          metrics['ETotal'] = float(score)
          ###PS
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1))
          metrics['PS'] = float(score)
          ###EM1
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1))
          metrics['EM1'] = float(score)
          ###EM2
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1))
          metrics['EM2'] = float(score)
          ###EM3
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1))
          metrics['EM3'] = float(score)
          ###EM
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1))
          metrics['EM'] = float(score)
          ###HAD1
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1))
          metrics['HAD1'] = float(score)
          ###HAD2
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1))
          metrics['HAD2'] = float(score)
          ###HAD3
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1))
          metrics['HAD3'] = float(score)
          ###HAD
          score,corr_pvalue = scipy.stats.pearsonr(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1))
          metrics['HAD'] = float(score)
        elif measure == 'MSE':
          ###TOTAL
          metrics['ETotal'] = float(mean_squared_error(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1)))
          ###PS
          metrics['PS'] = float(mean_squared_error(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(mean_squared_error(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(mean_squared_error(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(mean_squared_error(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(mean_squared_error(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(mean_squared_error(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(mean_squared_error(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(mean_squared_error(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(mean_squared_error(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1)))
        elif measure == 'Wasserstein':
          ###TOTAL
          metrics['ETotal'] = float(wasserstein_distance(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1)))
          ###PS
          metrics['PS'] = float(wasserstein_distance(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(wasserstein_distance(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(wasserstein_distance(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(wasserstein_distance(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(wasserstein_distance(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(wasserstein_distance(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(wasserstein_distance(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(wasserstein_distance(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(wasserstein_distance(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1)))
        elif measure == 'kolmogorov-smirnov':
          ###TOTAL
          metrics['ETotal'], pvalue = ks_2samp(input_val_Data.sum(axis=1),reconstruct_val_Data.sum(axis=1))
          ###PS
          metrics['PS'], pvalue = ks_2samp(input_val_Data[:,0:7,].sum(axis=1),reconstruct_val_Data[:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'], pvalue = ks_2samp(input_val_Data[:,8:71,].sum(axis=1),reconstruct_val_Data[:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'], pvalue = ks_2samp(input_val_Data[:,72:79,].sum(axis=1),reconstruct_val_Data[:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'], pvalue = ks_2samp(input_val_Data[:,80:87,].sum(axis=1),reconstruct_val_Data[:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'], pvalue = ks_2samp(input_val_Data[:,0:87,].sum(axis=1),reconstruct_val_Data[:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'], pvalue = ks_2samp(input_val_Data[:,88:91,].sum(axis=1),reconstruct_val_Data[:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'], pvalue = ks_2samp(input_val_Data[:,92:95,].sum(axis=1),reconstruct_val_Data[:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'], pvalue = ks_2samp(input_val_Data[:,96:99,].sum(axis=1),reconstruct_val_Data[:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'], pvalue = ks_2samp(input_val_Data[:,88:99,].sum(axis=1),reconstruct_val_Data[:,88:99,].sum(axis=1))
        elif measure == 'DeltaEnergy':
          ###TOTAL
          metrics['ETotal'] = float(np.average(input_val_Data.sum(axis=1)-reconstruct_val_Data.sum(axis=1)))
          ###PS
          metrics['PS'] = float(np.average(input_val_Data[:,0:7,].sum(axis=1)-reconstruct_val_Data[:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(np.average(input_val_Data[:,8:71,].sum(axis=1)-reconstruct_val_Data[:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(np.average(input_val_Data[:,72:79,].sum(axis=1)-reconstruct_val_Data[:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(np.average(input_val_Data[:,80:87,].sum(axis=1)-reconstruct_val_Data[:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(np.average(input_val_Data[:,0:87,].sum(axis=1)-reconstruct_val_Data[:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(np.average(input_val_Data[:,88:91,].sum(axis=1)-reconstruct_val_Data[:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(np.average(input_val_Data[:,92:95,].sum(axis=1)-reconstruct_val_Data[:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(np.average(input_val_Data[:,96:99,].sum(axis=1)-reconstruct_val_Data[:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(np.average(input_val_Data[:,88:99,].sum(axis=1)-reconstruct_val_Data[:,88:99,].sum(axis=1)))
        elif measure == 'Normalized_DeltaEnergy':
          ###TOTAL
          metrics['ETotal'] = float(np.average(input_val_Data.sum(axis=1)-reconstruct_val_Data.sum(axis=1)/input_val_Data.sum(axis=1)))
          ###PS
          metrics['PS'] = float(np.average(input_val_Data[:,0:7,].sum(axis=1)-reconstruct_val_Data[:,0:7,].sum(axis=1)/input_val_Data[:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(np.average(input_val_Data[:,8:71,].sum(axis=1)-reconstruct_val_Data[:,8:71,].sum(axis=1)/input_val_Data[:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(np.average(input_val_Data[:,72:79,].sum(axis=1)-reconstruct_val_Data[:,72:79,].sum(axis=1)/input_val_Data[:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(np.average(input_val_Data[:,80:87,].sum(axis=1)-reconstruct_val_Data[:,80:87,].sum(axis=1)/input_val_Data[:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(np.average(input_val_Data[:,0:87,].sum(axis=1)-reconstruct_val_Data[:,0:87,].sum(axis=1)/input_val_Data[:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(np.average(input_val_Data[:,88:91,].sum(axis=1)-reconstruct_val_Data[:,88:91,].sum(axis=1)/input_val_Data[:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(np.average(input_val_Data[:,92:95,].sum(axis=1)-reconstruct_val_Data[:,92:95,].sum(axis=1)/input_val_Data[:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(np.average(input_val_Data[:,96:99,].sum(axis=1)-reconstruct_val_Data[:,96:99,].sum(axis=1)/input_val_Data[:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(np.average(input_val_Data[:,88:99,].sum(axis=1)-reconstruct_val_Data[:,88:99,].sum(axis=1)/input_val_Data[:,88:99,].sum(axis=1)))
      except:
        metrics['ETotal'] = None
        metrics['PS'] = None
        metrics['EM1'] = None
        metrics['EM2'] = None
        metrics['EM3'] = None
        metrics['EM'] = None
        metrics['HAD1'] = None
        metrics['HAD2'] = None
        metrics['HAD3'] = None
        metrics['HAD'] = None

      table.insert(metrics)

      metrics = OrderedDict()
      print "SIGNAL"
      #SIGNAL

      ##### MI/KL
      metrics['Class'] = 'Signal'
      metrics['Model'] = model_name
      metrics['Layer'] = str(layer)
      metrics['time'] = time
      metrics['Measure'] = measure
      metrics['Normed'] = normalizacao
      #metrics['HL_Neuron'] = hl_neuron
      metrics['sort'] = sort
      metrics['etBinIdx'] = etBinIdx
      metrics['etaBinIdx'] = etaBinIdx
      metrics['phase'] = phase
      #metrics['Elapsed'] = elapsed
      #metrics['fine_tuning'] = fine_tuning


      for anel in range(100):
        #print anel
        try:
          if measure == 'Normalized_MI':
            rr = calc_MI2(input[0][:,anel],reconstructed[0][:,anel])
            score = np.sqrt(1. - np.exp(-2 * rr))
          elif measure == 'MI':
            score = calc_MI2(input[0][:,anel],reconstructed[0][:,anel])
          elif measure == 'KLdiv':
            score = calc_kl(input[0][:,anel],reconstructed[0][:,anel])
          elif measure == 'chiSquared':
            score,chi_pvalue =calc_chisquare(input[0][:,anel],reconstructed[0][:,anel])
          elif measure == 'Correlation':
            score,corr_pvalue= scipy.stats.pearsonr(input[0][:,anel],reconstructed[0][:,anel])
          elif measure == 'MSE':
            score= float(mean_squared_error(input[0][:,anel],reconstructed[0][:,anel]))
          elif measure == 'Wasserstein':
            score= float(wasserstein_distance(input[0][:,anel],reconstructed[0][:,anel]))
          elif measure == 'kolmogorov-smirnov':
            score, p_value= ks_2samp(input[0][:,anel],reconstructed[0][:,anel])
          elif measure == 'DeltaEnergy':
            score= float(np.average(input[0][:,anel]-reconstructed[0][:,anel], axis=0))
          elif measure == 'Normalized_DeltaEnergy':
            score= float(np.average((input[0][:,anel]-reconstructed[0][:,anel])/input[0][:,anel], axis=0))
          if math.isnan(score):
            score = None
          metrics[str(anel+1)] = float(score)
        except:
          print 'Anel '+str(anel)+' apresenta erros de calculo.'
          metrics[str(anel+1)] = None

      try:
        if measure == 'Normalized_MI':
          ###TOTAL
          rr = calc_MI2(input[0].sum(axis=1),reconstructed[0].sum(axis=1))
          metrics['ETotal'] = np.sqrt(1. - np.exp(-2 * rr))
          ###PS
          rr = calc_MI2(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1))
          metrics['PS'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM1
          rr = calc_MI2(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1))
          metrics['EM1'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM2
          rr = calc_MI2(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1))
          metrics['EM2'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM3
          rr = calc_MI2(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1))
          metrics['EM3'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM
          rr = calc_MI2(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1))
          metrics['EM'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD1
          rr = calc_MI2(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1))
          metrics['HAD1'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD2
          rr = calc_MI2(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1))
          metrics['HAD2'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD3
          rr = calc_MI2(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1))
          metrics['HAD3'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD
          rr = calc_MI2(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1))
          metrics['HAD'] = np.sqrt(1. - np.exp(-2 * rr))
        elif measure == 'MI':
          ###TOTAL
          metrics['ETotal'] = calc_MI2(input[0].sum(axis=1),reconstructed[0].sum(axis=1))
          ###PS
          metrics['PS'] = calc_MI2(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'] = calc_MI2(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'] = calc_MI2(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'] = calc_MI2(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'] = calc_MI2(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'] = calc_MI2(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'] = calc_MI2(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'] = calc_MI2(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'] = calc_MI2(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1))
        elif measure == 'KLdiv':
          #TOTAL
          metrics['ETotal'] = calc_kl(input[0].sum(axis=1),reconstructed[0].sum(axis=1))
          ###PS
          metrics['PS'] = calc_kl(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'] = calc_kl(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'] = calc_kl(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'] = calc_kl(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'] = calc_kl(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'] = calc_kl(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'] = calc_kl(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'] = calc_kl(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'] = calc_kl(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1))
        elif measure == 'chiSquared':
          ###TOTAL
          metrics['ETotal'],chi_pvalue = calc_chisquare(input[0].sum(axis=1),reconstructed[0].sum(axis=1))
          ###PS
          metrics['PS'],chi_pvalue = calc_chisquare(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'],chi_pvalue = calc_chisquare(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'],chi_pvalue = calc_chisquare(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'],chi_pvalue = calc_chisquare(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'],chi_pvalue = calc_chisquare(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'],chi_pvalue = calc_chisquare(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'],chi_pvalue = calc_chisquare(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'],chi_pvalue = calc_chisquare(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'],chi_pvalue = calc_chisquare(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1))
        elif measure == 'Correlation':
          ###TOTAL
          score,corr_pvalue = scipy.stats.pearsonr(input[0].sum(axis=1),reconstructed[0].sum(axis=1))
          metrics['ETotal'] = float(score)
          ###PS
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1))
          metrics['PS'] = float(score)
          ###EM1
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1))
          metrics['EM1'] = float(score)
          ###EM2
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1))
          metrics['EM2'] = float(score)
          ###EM3
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1))
          metrics['EM3'] = float(score)
          ###EM
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1))
          metrics['EM'] = float(score)
          ###HAD1
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1))
          metrics['HAD1'] = float(score)
          ###HAD2
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1))
          metrics['HAD2'] = float(score)
          ###HAD3
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1))
          metrics['HAD3'] = float(score)
          ###HAD
          score,corr_pvalue = scipy.stats.pearsonr(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1))
          metrics['HAD'] = float(score)
        elif measure == 'MSE':
          ###TOTAL
          metrics['ETotal'] = float(mean_squared_error(input[0].sum(axis=1),reconstructed[0].sum(axis=1)))
          ###PS
          metrics['PS'] = float(mean_squared_error(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(mean_squared_error(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(mean_squared_error(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(mean_squared_error(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(mean_squared_error(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(mean_squared_error(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(mean_squared_error(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(mean_squared_error(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(mean_squared_error(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1)))
        elif measure == 'Wasserstein':
          ###TOTAL
          metrics['ETotal'] = float(wasserstein_distance(input[0].sum(axis=1),reconstructed[0].sum(axis=1)))
          ###PS
          metrics['PS'] = float(wasserstein_distance(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(wasserstein_distance(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(wasserstein_distance(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(wasserstein_distance(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(wasserstein_distance(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(wasserstein_distance(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(wasserstein_distance(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(wasserstein_distance(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(wasserstein_distance(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1)))
        elif measure == 'kolmogorov-smirnov':
          ###TOTAL
          metrics['ETotal'], p_value = ks_2samp(input[0].sum(axis=1),reconstructed[0].sum(axis=1))
          ###PS
          metrics['PS'], p_value = ks_2samp(input[0][:,0:7,].sum(axis=1),reconstructed[0][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'], p_value = ks_2samp(input[0][:,8:71,].sum(axis=1),reconstructed[0][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'], p_value = ks_2samp(input[0][:,72:79,].sum(axis=1),reconstructed[0][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'], p_value = ks_2samp(input[0][:,80:87,].sum(axis=1),reconstructed[0][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'], p_value = ks_2samp(input[0][:,0:87,].sum(axis=1),reconstructed[0][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'], p_value = ks_2samp(input[0][:,88:91,].sum(axis=1),reconstructed[0][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'], p_value = ks_2samp(input[0][:,92:95,].sum(axis=1),reconstructed[0][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'], p_value = ks_2samp(input[0][:,96:99,].sum(axis=1),reconstructed[0][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'], p_value = ks_2samp(input[0][:,88:99,].sum(axis=1),reconstructed[0][:,88:99,].sum(axis=1))
        elif measure == 'DeltaEnergy':
          ###TOTAL
          metrics['ETotal'] = float(np.average(input[0].sum(axis=1)-reconstructed[0].sum(axis=1)))
          ###PS
          metrics['PS'] = float(np.average(input[0][:,0:7,].sum(axis=1)-reconstructed[0][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(np.average(input[0][:,8:71,].sum(axis=1)-reconstructed[0][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(np.average(input[0][:,72:79,].sum(axis=1)-reconstructed[0][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(np.average(input[0][:,80:87,].sum(axis=1)-reconstructed[0][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(np.average(input[0][:,0:87,].sum(axis=1)-reconstructed[0][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(np.average(input[0][:,88:91,].sum(axis=1)-reconstructed[0][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(np.average(input[0][:,92:95,].sum(axis=1)-reconstructed[0][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(np.average(input[0][:,96:99,].sum(axis=1)-reconstructed[0][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(np.average(input[0][:,88:99,].sum(axis=1)-reconstructed[0][:,88:99,].sum(axis=1)))
        elif measure == 'Normalized_DeltaEnergy':
          ###TOTAL
          metrics['ETotal'] = float(np.average(input[0].sum(axis=1)-reconstructed[0].sum(axis=1)/input[0].sum(axis=1)))
          ###PS
          metrics['PS'] = float(np.average(input[0][:,0:7,].sum(axis=1)-reconstructed[0][:,0:7,].sum(axis=1)/input[0][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(np.average(input[0][:,8:71,].sum(axis=1)-reconstructed[0][:,8:71,].sum(axis=1)/input[0][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(np.average(input[0][:,72:79,].sum(axis=1)-reconstructed[0][:,72:79,].sum(axis=1)/input[0][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(np.average(input[0][:,80:87,].sum(axis=1)-reconstructed[0][:,80:87,].sum(axis=1)/input[0][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(np.average(input[0][:,0:87,].sum(axis=1)-reconstructed[0][:,0:87,].sum(axis=1)/input[0][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(np.average(input[0][:,88:91,].sum(axis=1)-reconstructed[0][:,88:91,].sum(axis=1)/input[0][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(np.average(input[0][:,92:95,].sum(axis=1)-reconstructed[0][:,92:95,].sum(axis=1)/input[0][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(np.average(input[0][:,96:99,].sum(axis=1)-reconstructed[0][:,96:99,].sum(axis=1)/input[0][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(np.average(input[0][:,88:99,].sum(axis=1)-reconstructed[0][:,88:99,].sum(axis=1)/input[0][:,88:99,].sum(axis=1)))
      except:
        metrics['ETotal'] = None
        metrics['PS'] = None
        metrics['EM1'] = None
        metrics['EM2'] = None
        metrics['EM3'] = None
        metrics['EM'] = None
        metrics['HAD1'] = None
        metrics['HAD2'] = None
        metrics['HAD3'] = None
        metrics['HAD'] = None

      table.insert(metrics)

      metrics = OrderedDict()
      print "BACKGROUND"
      #BACKGROUND

      ##### MI/KL
      metrics['Class'] = 'Background'
      metrics['Model'] = model_name
      metrics['Layer'] = str(layer)
      metrics['time'] = time
      metrics['Measure'] = measure
      metrics['Normed'] = normalizacao
      #metrics['HL_Neuron'] = hl_neuron
      metrics['sort'] = sort
      metrics['etBinIdx'] = etBinIdx
      metrics['etaBinIdx'] = etaBinIdx
      metrics['phase'] = phase
      #metrics['Elapsed'] = elapsed
      #metrics['fine_tuning'] = fine_tuning

      for anel in range(100):
        try:
          if measure == 'Normalized_MI':
            rr = calc_MI2(input[1][:,anel],reconstructed[1][:,anel])
            score = np.sqrt(1. - np.exp(-2 * rr))
          elif measure == 'MI':
            score = calc_MI2(input[1][:,anel],reconstructed[1][:,anel])
          elif measure == 'KLdiv':
            score = calc_kl(input[1][:,anel],reconstructed[1][:,anel])
          elif measure == 'chiSquared':
            score,chi_pvalue =calc_chisquare(input[1][:,anel],reconstructed[1][:,anel])
          elif measure == 'Correlation':
            score,corr_pvalue= scipy.stats.pearsonr(input[1][:,anel],reconstructed[1][:,anel])
          elif measure == 'MSE':
            score= float(mean_squared_error(input[1][:,anel],reconstructed[1][:,anel]))
          elif measure == 'Wasserstein':
            score= float(wasserstein_distance(input[1][:,anel],reconstructed[1][:,anel]))
          elif measure == 'kolmogorov-smirnov':
            score, p_value = ks_2samp(input[1][:,anel],reconstructed[1][:,anel])
          elif measure == 'DeltaEnergy':
            score= float(np.average(input[1][:,anel]-reconstructed[1][:,anel], axis=0))
          elif measure == 'Normalized_DeltaEnergy':
            score= float(np.average((input[1][:,anel]-reconstructed[1][:,anel])/input[1][:,anel], axis=0))
          if math.isnan(score):
            score = None
          metrics[str(anel+1)] = float(score)
        except:
          print 'Anel '+str(anel)+' apresenta erros de calculo.'
          metrics[str(anel+1)] = None

      try:
        if measure == 'Normalized_MI':
          ###TOTAL
          rr = calc_MI2(input[1].sum(axis=1),reconstructed[1].sum(axis=1))
          metrics['ETotal'] = np.sqrt(1. - np.exp(-2 * rr))
          ###PS
          rr = calc_MI2(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1))
          metrics['PS'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM1
          rr = calc_MI2(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1))
          metrics['EM1'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM2
          rr = calc_MI2(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1))
          metrics['EM2'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM3
          rr = calc_MI2(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1))
          metrics['EM3'] = np.sqrt(1. - np.exp(-2 * rr))
          ###EM
          rr = calc_MI2(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1))
          metrics['EM'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD1
          rr = calc_MI2(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1))
          metrics['HAD1'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD2
          rr = calc_MI2(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1))
          metrics['HAD2'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD3
          rr = calc_MI2(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1))
          metrics['HAD3'] = np.sqrt(1. - np.exp(-2 * rr))
          ###HAD
          rr = calc_MI2(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1))
          metrics['HAD'] = np.sqrt(1. - np.exp(-2 * rr))
        elif measure == 'MI':
          ###TOTAL
          metrics['ETotal'] = calc_MI2(input[1].sum(axis=1),reconstructed[1].sum(axis=1))
          ###PS
          metrics['PS'] = calc_MI2(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'] = calc_MI2(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'] = calc_MI2(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'] = calc_MI2(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'] = calc_MI2(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'] = calc_MI2(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'] = calc_MI2(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'] = calc_MI2(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'] = calc_MI2(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1))
        elif measure == 'KLdiv':
          #TOTAL
          metrics['ETotal'] = calc_kl(input[1].sum(axis=1),reconstructed[1].sum(axis=1))
          ###PS
          metrics['PS'] = calc_kl(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'] = calc_kl(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'] = calc_kl(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'] = calc_kl(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'] = calc_kl(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'] = calc_kl(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'] = calc_kl(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'] = calc_kl(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'] = calc_kl(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1))
        elif measure == 'chiSquared':
          ###TOTAL
          metrics['ETotal'],chi_pvalue = calc_chisquare(input[1].sum(axis=1),reconstructed[1].sum(axis=1))
          ###PS
          metrics['PS'],chi_pvalue = calc_chisquare(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'],chi_pvalue = calc_chisquare(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'],chi_pvalue = calc_chisquare(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'],chi_pvalue = calc_chisquare(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'],chi_pvalue = calc_chisquare(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'],chi_pvalue = calc_chisquare(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'],chi_pvalue = calc_chisquare(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'],chi_pvalue = calc_chisquare(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'],chi_pvalue = calc_chisquare(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1))
        elif measure == 'Correlation':
          ###TOTAL
          score,corr_pvalue = scipy.stats.pearsonr(input[1].sum(axis=1),reconstructed[1].sum(axis=1))
          metrics['ETotal'] = float(score)
          ###PS
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1))
          metrics['PS'] = float(score)
          ###EM1
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1))
          metrics['EM1'] = float(score)
          ###EM2
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1))
          metrics['EM2'] = float(score)
          ###EM3
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1))
          metrics['EM3'] = float(score)
          ###EM
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1))
          metrics['EM'] = float(score)
          ###HAD1
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1))
          metrics['HAD1'] = float(score)
          ###HAD2
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1))
          metrics['HAD2'] = float(score)
          ###HAD3
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1))
          metrics['HAD3'] = float(score)
          ###HAD
          score,corr_pvalue = scipy.stats.pearsonr(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1))
          metrics['HAD'] = float(score)
        elif measure == 'MSE':
          ###TOTAL
          metrics['ETotal'] = float(mean_squared_error(input[1].sum(axis=1),reconstructed[1].sum(axis=1)))
          ###PS
          metrics['PS'] = float(mean_squared_error(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(mean_squared_error(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(mean_squared_error(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(mean_squared_error(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(mean_squared_error(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(mean_squared_error(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(mean_squared_error(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(mean_squared_error(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(mean_squared_error(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1)))
        elif measure == 'Wasserstein':
          ###TOTAL
          metrics['ETotal'] = float(wasserstein_distance(input[1].sum(axis=1),reconstructed[1].sum(axis=1)))
          ###PS
          metrics['PS'] = float(wasserstein_distance(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(wasserstein_distance(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(wasserstein_distance(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(wasserstein_distance(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(wasserstein_distance(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(wasserstein_distance(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(wasserstein_distance(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(wasserstein_distance(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(wasserstein_distance(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1)))
        elif measure == 'kolmogorov-smirnov':
          ###TOTAL
          metrics['ETotal'],p_value = ks_2samp(input[1].sum(axis=1),reconstructed[1].sum(axis=1))
          ###PS
          metrics['PS'],p_value = ks_2samp(input[1][:,0:7,].sum(axis=1),reconstructed[1][:,0:7,].sum(axis=1))
          ###EM1
          metrics['EM1'],p_value = ks_2samp(input[1][:,8:71,].sum(axis=1),reconstructed[1][:,8:71,].sum(axis=1))
          ###EM2
          metrics['EM2'],p_value = ks_2samp(input[1][:,72:79,].sum(axis=1),reconstructed[1][:,72:79,].sum(axis=1))
          ###EM3
          metrics['EM3'],p_value = ks_2samp(input[1][:,80:87,].sum(axis=1),reconstructed[1][:,80:87,].sum(axis=1))
          ###EM
          metrics['EM'],p_value = ks_2samp(input[1][:,0:87,].sum(axis=1),reconstructed[1][:,0:87,].sum(axis=1))
          ###HAD1
          metrics['HAD1'],p_value = ks_2samp(input[1][:,88:91,].sum(axis=1),reconstructed[1][:,88:91,].sum(axis=1))
          ###HAD2
          metrics['HAD2'],p_value = ks_2samp(input[1][:,92:95,].sum(axis=1),reconstructed[1][:,92:95,].sum(axis=1))
          ###HAD3
          metrics['HAD3'],p_value = ks_2samp(input[1][:,96:99,].sum(axis=1),reconstructed[1][:,96:99,].sum(axis=1))
          ###HAD
          metrics['HAD'],p_value = ks_2samp(input[1][:,88:99,].sum(axis=1),reconstructed[1][:,88:99,].sum(axis=1))
        elif measure == 'DeltaEnergy':
          ###TOTAL
          metrics['ETotal'] = float(np.average(input[1].sum(axis=1)-reconstructed[1].sum(axis=1)))
          ###PS
          metrics['PS'] = float(np.average(input[1][:,0:7,].sum(axis=1)-reconstructed[1][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(np.average(input[1][:,8:71,].sum(axis=1)-reconstructed[1][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(np.average(input[1][:,72:79,].sum(axis=1)-reconstructed[1][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(np.average(input[1][:,80:87,].sum(axis=1)-reconstructed[1][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(np.average(input[1][:,0:87,].sum(axis=1)-reconstructed[1][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(np.average(input[1][:,88:91,].sum(axis=1)-reconstructed[1][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(np.average(input[1][:,92:95,].sum(axis=1)-reconstructed[1][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(np.average(input[1][:,96:99,].sum(axis=1)-reconstructed[1][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(np.average(input[1][:,88:99,].sum(axis=1)-reconstructed[1][:,88:99,].sum(axis=1)))
        elif measure == 'Normalized_DeltaEnergy':
          ###TOTAL
          metrics['ETotal'] = float(np.average(input[1].sum(axis=1)-reconstructed[1].sum(axis=1)/input[1].sum(axis=1)))
          ###PS
          metrics['PS'] = float(np.average(input[1][:,0:7,].sum(axis=1)-reconstructed[1][:,0:7,].sum(axis=1)/input[1][:,0:7,].sum(axis=1)))
          ###EM1
          metrics['EM1'] = float(np.average(input[1][:,8:71,].sum(axis=1)-reconstructed[1][:,8:71,].sum(axis=1)/input[1][:,8:71,].sum(axis=1)))
          ###EM2
          metrics['EM2'] = float(np.average(input[1][:,72:79,].sum(axis=1)-reconstructed[1][:,72:79,].sum(axis=1)/input[1][:,72:79,].sum(axis=1)))
          ###EM3
          metrics['EM3'] = float(np.average(input[1][:,80:87,].sum(axis=1)-reconstructed[1][:,80:87,].sum(axis=1)/input[1][:,80:87,].sum(axis=1)))
          ###EM
          metrics['EM'] = float(np.average(input[1][:,0:87,].sum(axis=1)-reconstructed[1][:,0:87,].sum(axis=1)/input[1][:,0:87,].sum(axis=1)))
          ###HAD1
          metrics['HAD1'] = float(np.average(input[1][:,88:91,].sum(axis=1)-reconstructed[1][:,88:91,].sum(axis=1)/input[1][:,88:91,].sum(axis=1)))
          ###HAD2
          metrics['HAD2'] = float(np.average(input[1][:,92:95,].sum(axis=1)-reconstructed[1][:,92:95,].sum(axis=1)/input[1][:,92:95,].sum(axis=1)))
          ###HAD3
          metrics['HAD3'] = float(np.average(input[1][:,96:99,].sum(axis=1)-reconstructed[1][:,96:99,].sum(axis=1)/input[1][:,96:99,].sum(axis=1)))
          ###HAD
          metrics['HAD'] = float(np.average(input[1][:,88:99,].sum(axis=1)-reconstructed[1][:,88:99,].sum(axis=1)/input[1][:,88:99,].sum(axis=1)))
      except:
        metrics['ETotal'] = None
        metrics['PS'] = None
        metrics['EM1'] = None
        metrics['EM2'] = None
        metrics['EM3'] = None
        metrics['EM'] = None
        metrics['HAD1'] = None
        metrics['HAD2'] = None
        metrics['HAD3'] = None
        metrics['HAD'] = None

      table.insert(metrics)
  return metrics

def get_reconstructionErrVector(data=None,reconstruct=None):
  if isinstance(data, (tuple, list,)):
    reconstructionError = []
    for i, cdata in enumerate(data):
      # #print i,cdata.shape
      reconstructionError.append( (cdata - reconstruct[i]) ** 2)

  return reconstructionError

def concatReconstructionError(data=None,reconstructErrVector=None):
  reconstructErrVector = applyNorm1(data=reconstructErrVector)
  if isinstance(data, (tuple, list,)):
    concatReconstructionError = []
    for i, cdata in enumerate(data):
      # #print i,cdata.shape
      concatReconstructionError.append( (np.concatenate((cdata,reconstructErrVector[i]),axis=1)))
  return concatReconstructionError

def applyNorm1(data=None):
    from TuningTools.coreDef import npCurrent
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

    if isinstance(data, (tuple, list,)):
      ret = []
      for i, cdata in enumerate(data):
        ret.append( cdata / norms[i] )
    else:
      ret = data / norms
    return ret

def plot_pdfs(norm1Par=None,reconstruct=None,model_name="",time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=None, dirout=None):
    import matplotlib.pyplot as plt
    import seaborn as sb
    from sklearn.metrics import mean_squared_error

    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
        if isinstance(reconstruct[layer], (tuple, list,)):
            unnorm_reconstruct = []
            for i, cdata in enumerate(reconstruct[layer]):
                #print i,cdata.shape
                unnorm_reconstruct.append( cdata * normlist[i])
            unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
            beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )
            r=unnorm_reconstruct_val_Data
            b=beforenorm_val_Data
            #np.savez_compressed('//home/caducovas/run/pdfs',iEnergy=b,rEnergy=r)
            fig, axs = plt.subplots(8, 14, figsize=(60, 40))
            rings=0
            for j in range(14):
                for i in range(8): ###CODE 10
                    if j> 10 and i>3:
                        fig.delaxes(axs[i,j])
                        continue
                    #rings=int(str(i)+str(j))
                    #print i,j,rings
                    #ax2 = axs[i,j].twinx()
                    try:
                        max_value = max(max(b[:,rings]),max(r[:,rings]))
                        min_value = min(min(b[:,rings]),min(r[:,rings]))
                        bins = min( len(np.histogram(b[:,rings],'fd')[0]), len(np.histogram(r[:,rings],'fd')[0]))
                        bins_list = np.linspace(min_value, max_value, num=bins)

                        axs[i,j].hist(b[:,rings], bins=bins_list, alpha=.5, color='b', label='Input Energy')
                        axs[i,j].hist(r[:,rings], bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
                        ###sb.kdeplot(b[:,rings],label="Input Energy",ax=axs[i,j],color='crimson')
                        ###sb.kdeplot(r[:,rings],label="Reconstructed Energy",ax=axs[i,j],color='deepskyblue')
                        ###nbins = len(np.histogram(b[:,rings],'fd')[0])
                        ###axs[i,j].hist(b[:,rings], bins=nbins, normed=True,color='crimson',histtype='stepfilled')
                        ###nbins = len(np.histogram(r[:,rings],'fd')[0])
                        ###axs[i,j].hist(r[:,rings], bins=nbins, normed=True,color='deepskyblue')
                        #axs[i,j].grid()
                        #axs[i,j].set_title('Ring '+str(rings)+' - '+model_name)
                        axs[i,j].get_yaxis().set_ticks([])
                        rr = calc_MI2(b[:,rings],r[:,rings])
                        mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
                        kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
                        #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
                        mse_score = mean_squared_error(b[:,rings],r[:,rings])
                        corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
                        axs[i,j].legend(loc=2)
                        axs[i,j].set_ylabel('#'+str(rings+1), color='b')
                        at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(round(100*corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
                                          prop=dict(size=8), frameon=True,
                                          loc='center right',
                                          )
                        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                        axs[i,j].add_artist(at)
                    except:
                        print "Deu ruim no anel:"+str(rings+1)
                    rings+=1
        plt.suptitle('Input X Reconstruction - '+model_name+' - '+str(layer), fontsize=24)
        plt.savefig(dirout+'/pdf_'+str(layer)+'_'+model_name+'_'+time+'.png',dpi=120)
        plt.clf()
        plt.close()
        png_files.append(dirout+'/pdf_'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_pdfs_representation(norm1Par=None,code=None,layer=None,model_name=None,time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=False, dirout=None):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    # beforenorm = norm1Par[0]
    # normlist = norm1Par[1]
    # afternorm = norm1Par[2]
    png_files=[]

    ###All Classes
    b=code[layer][0] ###signal
    r=code[layer][1] ###background

    #diroutAllclasses = dirout+model_name+'_'+time+'/Representation/'
    #if not os.path.exists(diroutAllclasses):
    #    print 'Creating output folder AllClasses...'
    #    os.makedirs(diroutAllclasses)
    plt.clf()


#    for rings in range(b.shape[1]):
    fig = plt.figure(figsize=(60, 40))

    grid_arrangement = GridStrategy.get_grid(40)
    ax_specs = get_gridspec(grid_arrangement)
    rings=0
    for i, spec in enumerate(ax_specs):
        ax = fig.add_subplot(plt.Subplot(fig, spec))

        try:
            at = AnchoredText('Signal \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nBackground \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2)),
            #at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            max_value = max(max(b[:,rings]),max(r[:,rings]))
            min_value = min(min(b[:,rings]),min(r[:,rings]))
            bins = min( len(np.histogram(b[:,rings],'fd')[0]), len(np.histogram(r[:,rings],'fd')[0]))
            bins_list = np.linspace(min_value, max_value, num=bins)

            ax.hist(b[:,rings], bins=bins_list, alpha=.5, color='b', label='Signal Energy')
            ax.hist(r[:,rings], bins=bins_list, alpha=.5, color='r', label='Background Energy')
        except:
            print "Error on ring: "+str(rings+1)

        #ax.axvline(np.max(b[:,rings]), color='b', linestyle='--', linewidth=.8, label='Max Signal Energy')
        #ax.axvline(np.max(r[:,rings]), color='r', linestyle='--', linewidth=.8, label='Max Background Energy')
        ax.set_title(r'Code '+str(rings+1)+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        ax.set_xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        ax.legend(loc='best', fontsize='medium')
        rings+=1
    plt.tight_layout()
    plt.suptitle('Representation - '+model_name+' - '+str(layer), fontsize=24)
    plt.savefig(dirout+'/pdf_representation_'+str(layer)+'_'+model_name+'_'+time+'.png',dpi=120)
    plt.clf()
    plt.close()
    png_files.append(dirout+'/pdf_representation_'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_pdfs_byclass(norm1Par=None,reconstruct=None,model_name="",time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=None, dirout=None):
    import matplotlib.pyplot as plt
    import seaborn as sb
    from sklearn.metrics import mean_squared_error

    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]
    classes=['Signal','Background']

    for layer in reconstruct.keys():
        for cl,className in enumerate(classes):
            #print 'LAYER: '+str(layer)
            #for nsort in reconstruct[layer].keys():
            #print "Sort: "+str(nsort)
            if isinstance(reconstruct[layer], (tuple, list,)):
                unnorm_reconstruct = []
                for i, cdata in enumerate(reconstruct[layer]):
                    #print i,cdata.shape
                    unnorm_reconstruct.append( cdata * normlist[i])
                r=unnorm_reconstruct[cl]
                b=beforenorm[cl]
                #np.savez_compressed('//home/caducovas/run/pdfs',iEnergy=b,rEnergy=r)
                fig, axs = plt.subplots(8, 14, figsize=(60, 40))
                rings=0
                for j in range(14):
                    for i in range(8): ###CODE 10
                        if j> 10 and i>3:
                            fig.delaxes(axs[i,j])
                            continue
                        #rings=int(str(i)+str(j))
                        #print i,j,rings
                        #ax2 = axs[i,j].twinx()
                        try:
                            sb.kdeplot(b[:,rings],label="Input Energy",ax=axs[i,j],color='crimson')
                            sb.kdeplot(r[:,rings],label="Reconstructed Energy",ax=axs[i,j],color='deepskyblue')
                            nbins = len(np.histogram(b[:,rings],'fd')[0])
                            axs[i,j].hist(b[:,rings], bins=nbins, normed=True,color='crimson',histtype='stepfilled')
                            nbins = len(np.histogram(r[:,rings],'fd')[0])
                            axs[i,j].hist(r[:,rings], bins=nbins, normed=True,color='deepskyblue')
                            #axs[i,j].grid()
                            #axs[i,j].set_title('Ring '+str(rings)+' - '+model_name)
                            axs[i,j].get_yaxis().set_ticks([])
                            rr = calc_MI2(b[:,rings],r[:,rings])
                            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
                            kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
                            #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
                            mse_score = mean_squared_error(b[:,rings],r[:,rings])
                            corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
                            axs[i,j].legend(loc=2)
                            axs[i,j].set_ylabel('#'+str(rings+1), color='b')
                            #at = AnchoredText(r'ATLAS $\sqrt{s}$ = 13 TeV'+"\nMC16 Calo\n\nInput \nMean: "+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2)),
                            at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(round(100*corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
                                              prop=dict(size=8), frameon=True,
                                              loc='center right',
                                              )
                            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                            axs[i,j].add_artist(at)
                        except:
                            print "Deu ruim no anel:"+str(rings+1)
                        rings+=1
            plt.suptitle(className+' - Input X Reconstruction - '+model_name+' - '+str(layer), fontsize=24)
            plt.savefig(dirout+'/pdf_'+className+'_'+str(layer)+'_'+model_name+'_'+time+'.png',dpi=120)
            plt.clf()
            plt.close()
            png_files.append(dirout+'/pdf_'+className+'_'+str(layer)+'_'+model_name+'_'+time+'.png')
    print len(png_files)
    return png_files

def plot_scatter(norm1Par=None,reconstruct=None,model_name="",time=None,sort=None,etBinIdx=None,etaBinIdx=None,normed=True,phase=None, dirout=None):
    import matplotlib.pyplot as plt
    import seaborn as sb

    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]
    if normed:
        normalized = '_normed'
    else:
        normalized = ''

    for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
        if isinstance(reconstruct[layer], (tuple, list,)):
            unnorm_reconstruct = []
            for i, cdata in enumerate(reconstruct[layer]):
                #print i,cdata.shape
                unnorm_reconstruct.append( cdata * normlist[i])
            unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
            beforenorm_val_Data = np.concatenate( beforenorm, axis=0)
            if normed:
                r=unnorm_reconstruct_val_Data
                b=beforenorm_val_Data
            else:
                r=np.concatenate( reconstruct[layer], axis=0 )
                b=np.concatenate( afternorm, axis=0)

            #np.savez_compressed('//home/caducovas/run/pdfs',iEnergy=b,rEnergy=r)
            fig, axs = plt.subplots(8, 14, figsize=(60, 40))
            rings=0
            for j in range(14):
                for i in range(8): ###CODE 10
                    if j> 10 and i>3:
                        fig.delaxes(axs[i,j])
                        continue
                    #rings=int(str(i)+str(j))
                    #print i,j,rings
                    #ax2 = axs[i,j].twinx()
                    try:
                        axs[i,j].scatter(b[:,rings],r[:,rings],color='royalblue')
                        #axs[i,j].grid()
                        #axs[i,j].set_title('Ring '+str(rings)+' - '+model_name)
                        #axs[i,j].get_yaxis().set_ticks([])
                        #rr = calc_MI2(b[:,rings],r[:,rings])
                        #mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
                        axs[i,j].set_title('#'+str(rings+1), color='b')
                        #axs[i,j].set_ylabel('#'+str(rings+1)+' MI: '+str(mi_score), color='b')
                        #axs[i,j].set_ylabel('#'+str(rings+1)+' MI: '+str(mi_score), color='b')
                        #at = AnchoredText(r'ATLAS $\sqrt{s}$ = 13 TeV'+"\nMC16 Calo\n\nInput \nMean: "+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2)),
                        #                  prop=dict(size=8), frameon=True,
                        #                  loc='center right',
                        #                  )
                        #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                        #axs[i,j].add_artist(at)
                    except:
                        print "Deu ruim no anel:"+str(rings+1)
                    rings+=1
        plt.suptitle('Scatter - Input X Reconstruction - '+model_name+' - '+str(layer)+normalized, fontsize=24)
        plt.savefig(dirout+'/scatter_'+normalized+str(layer)+'_'+model_name+'_'+time+'.png',dpi=120)
        plt.clf()
        plt.close()
        png_files.append(dirout+'/scatter_'+normalized+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_input_reconstruction_separed(norm1Par=None,reconstruct=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
    import sqlite3
    import pandas as pd
    from numpy import nan
    #%matplotlib inline
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    classes=['Signal','Background']
##TEM DOIS IDENTS QUE TEM QUE TIRAR. O DO FOR LAYER E O DO IF IS INSTANCES
    #for a in reconstruct.keys():

    #print 'LAYER: '+str(layer)
    #for nsort in reconstruct[layer].keys():
    #print "Sort: "+str(nsort)
    if isinstance(reconstruct[layer], (tuple, list,)):
        unnorm_reconstruct = []
        for i, cdata in enumerate(reconstruct[layer]):
            #print i,cdata.shape
            unnorm_reconstruct.append( cdata * normlist[i])

    dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'Signal' and Measure = 'Normalized_MI' and Normed='no' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
    dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
    #dfSignal.fillna(value=nan, inplace=True)
    dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'Background' and Measure = 'Normalized_MI' and Normed='no' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
    dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
    #dfBkg.fillna(value=nan, inplace=True)
    sgn=dfSignal.values.astype(np.float32)
    bkg=dfBkg.values.astype(np.float32)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(25,10))
    ax1.errorbar(np.arange(beforenorm[0].shape[1]), np.mean(beforenorm[0], axis=0),yerr=np.std(beforenorm[0], axis=0), fmt='D-', color='crimson', label='Mean Profile')
    ax1.errorbar(np.arange(unnorm_reconstruct[0].shape[1]), np.mean(unnorm_reconstruct[0], axis=0),yerr=np.std(unnorm_reconstruct[0], axis=0), fmt='^-', color='deepskyblue', label='Mean Reconstructed Profile')

    ax1.set_title(r'Signal Profile',fontsize= 20)
    ax1.set_xlabel('#Rings', fontsize= 20)
    ax1.set_ylabel('Energy [MeV]',fontsize= 20)
    ax1.tick_params(labelsize= 15)
    ax1.legend(loc='best', fontsize='xx-large')
    ax12 = ax1.twinx()
    ax12.errorbar(np.arange(sgn.shape[1]), np.mean(sgn, axis=0),yerr=np.std(sgn, axis=0), fmt='gD-', color='cornflowerblue')
    ax12.set_ylabel('Normalized Mutual Information', fontsize='xx-large')
    #ax12.set_ylim(top=1)

    ax2.errorbar(np.arange(beforenorm[1].shape[1]), np.mean(beforenorm[1], axis=0),yerr=np.std(beforenorm[1], axis=0), fmt='o-',color='crimson', label='Mean Profile')
    ax2.errorbar(np.arange(unnorm_reconstruct[1].shape[1]), np.mean(unnorm_reconstruct[1], axis=0),yerr=np.std(unnorm_reconstruct[1], axis=0), fmt='^-', color='deepskyblue', label='Mean Reconstructed Profile')
    ax2.set_title(r'Background Patterns',fontsize= 20)
    ax2.set_xlabel('#Rings', fontsize= 20)
    ax2.set_ylabel('Energy [MeV]',fontsize= 20)
    ax2.tick_params(labelsize= 15)
    ax2.legend(loc='best', fontsize='xx-large')
    ax22 = ax2.twinx()
    ax22.errorbar(np.arange(bkg.shape[1]), np.mean(bkg, axis=0),yerr=np.std(bkg, axis=0), fmt='go-')
    ax22.set_ylabel('Normalized Mutual Information', fontsize='xx-large')
    #ax22.set_ylim(top=1)
    #plt.legend(['Electron', 'Background'], loc='best', fontsize='xx-large')

    for i in [7, 71, 79, 87, 91, 95]:
        ax1.axvline(i, color='gray', linestyle='--', linewidth=.8)
        ax2.axvline(i, color='gray', linestyle='--', linewidth=.8)
    #log_scale=False
    #if log_scale:
    #  y_position = #.8*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
    #else:
    y_position = 0.98 #.8*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)])

    for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
                      (76,y_position,r'EM2'),(80,y_position,r'EM3'),
                    (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
        ax1.text(x,y,text, fontsize=15, rotation=90)
        ax2.text(x,y,text, fontsize=15, rotation=90)

    #    plt.savefig('meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #else:
    #    plt.savefig(output_name+'_meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #plt.show()
    plt.suptitle('Energy Profile - Input X Reconstruction - '+model_name+' - '+str(layer), fontsize=24)
    plt.savefig(dirout+'/energy_prof_'+str(layer)+'_'+model_name+'_'+time+'.png')
    plt.clf()
    plt.close()
    png_files.append(dirout+'/energy_prof_'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_input_reconstruction_separed_noErrbar(norm1Par=None,reconstruct=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
    import sqlite3
    import pandas as pd
    from numpy import nan
    #%matplotlib inline
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    classes=['Signal','Background']
##TEM DOIS IDENTS QUE TEM QUE TIRAR. O DO FOR LAYER E O DO IF IS INSTANCES
    #for a in reconstruct.keys():

    #print 'LAYER: '+str(layer)
    #for nsort in reconstruct[layer].keys():
    #print "Sort: "+str(nsort)
    if isinstance(reconstruct[layer], (tuple, list,)):
        unnorm_reconstruct = []
        for i, cdata in enumerate(reconstruct[layer]):
            #print i,cdata.shape
            unnorm_reconstruct.append( cdata * normlist[i])

    dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'Signal' and Measure = 'KLdiv' and Normed='no' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
    dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
    #dfSignal.fillna(value=nan, inplace=True)
    dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Class = 'Background' and Measure = 'KLdiv' and Normed='no' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
    dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
    #dfBkg.fillna(value=nan, inplace=True)
    sgn=dfSignal.values.astype(np.float32)
    bkg=dfBkg.values.astype(np.float32)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(25,10))
    ax1.plot(np.arange(beforenorm[0].shape[1]), np.mean(beforenorm[0], axis=0), marker='<', color='crimson', label='Mean Profile')
    ax1.plot(np.arange(unnorm_reconstruct[0].shape[1]), np.mean(unnorm_reconstruct[0], axis=0),marker='>', color='deepskyblue', label='Mean Reconstructed Profile')

    ax1.set_title(r'Signal Profile',fontsize= 20)
    ax1.set_xlabel('#Rings', fontsize= 20)
    ax1.set_ylabel('Energy [MeV]',fontsize= 20)
    ax1.tick_params(labelsize= 15)
    ax1.legend(loc='best', fontsize='xx-large')
    ax12 = ax1.twinx()
    ax12.errorbar(np.arange(sgn.shape[1]), np.mean(sgn, axis=0),yerr=np.std(sgn, axis=0), fmt='gD-', color='cornflowerblue')
    ax12.set_ylabel('KL Divergence', fontsize='xx-large')
    #ax12.set_ylim(top=1)

    ax2.plot(np.arange(beforenorm[1].shape[1]), np.mean(beforenorm[1], axis=0),marker='<',color='crimson', label='Mean Profile')
    ax2.plot(np.arange(unnorm_reconstruct[1].shape[1]), np.mean(unnorm_reconstruct[1], axis=0),marker='>', color='deepskyblue', label='Mean Reconstructed Profile')
    ax2.set_title(r'Background Patterns',fontsize= 20)
    ax2.set_xlabel('#Rings', fontsize= 20)
    ax2.set_ylabel('Energy [MeV]',fontsize= 20)
    ax2.tick_params(labelsize= 15)
    ax2.legend(loc='best', fontsize='xx-large')
    ax22 = ax2.twinx()
    ax22.errorbar(np.arange(bkg.shape[1]), np.mean(bkg, axis=0),yerr=np.std(bkg, axis=0), fmt='go-')
    ax22.set_ylabel('KL Divergence', fontsize='xx-large')
    #ax22.set_ylim(top=1)
    #plt.legend(['Electron', 'Background'], loc='best', fontsize='xx-large')

    for i in [7, 71, 79, 87, 91, 95]:
        ax1.axvline(i, color='gray', linestyle='--', linewidth=.8)
        ax2.axvline(i, color='gray', linestyle='--', linewidth=.8)
    #log_scale=False
    #if log_scale:
    #  y_position = #.8*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
    #else:
    y_position = 0.8*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)])

    for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
                      (76,y_position,r'EM2'),(80,y_position,r'EM3'),
                    (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
        ax1.text(x,y,text, fontsize=15, rotation=90)
        ax2.text(x,y,text, fontsize=15, rotation=90)

    #    plt.savefig('meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #else:
    #    plt.savefig(output_name+'_meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #plt.show()
    plt.suptitle('Energy Profile - Input X Reconstruction - '+model_name+' - '+str(layer), fontsize=24)
    plt.savefig(dirout+'/energy_prof_'+str(layer)+'_'+model_name+'_'+time+'.png')
    plt.clf()
    plt.close()
    png_files.append(dirout+'/energy_prof_'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_input_reconstruction_error(norm1Par=None,reconstruct=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
  import sqlite3
  import pandas as pd
  from numpy import nan
  #%matplotlib inline
  import matplotlib.pyplot as plt
  png_files=[]


  plt.style.use('ggplot')

  cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
  # # Et and Eta indices
  # et_index  = [0, 1, 2,3]
  # etRange = ['[15, 20]','[20, 30]','[30, 40]','[40, 50000]']

  # eta_index = [0, 1, 2, 3, 4,5,6,7,8]
  # etaRange = ['[0, 0.6]','[0.6, 0.8]','[0.8, 1.15]','[1.15, 1.37]','[1.37, 1.52]','[1.52, 1.81]','[1.81, 2.01]','[2.01, 2.37]','[2.37, 2.47]']
  beforenorm = norm1Par[0]
  normlist = norm1Par[1]
  afternorm = norm1Par[2]

  if isinstance(reconstruct[layer], (tuple, list,)):
      unnorm_reconstruct = []
      for i, cdata in enumerate(reconstruct[layer]):
          #print i,cdata.shape
          unnorm_reconstruct.append( cdata * normlist[i])
      unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
      beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

  dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'All' and Normed='yes' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  dfAll.fillna(value=nan, inplace=True)
  dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'Signal' and Normed='yes' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  #dfSignal.fillna(value=nan, inplace=True)
  dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'Background' and Normed='yes' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

  allClasses_MSE=dfAll.values.astype(np.float32)
  sgn_MSE=dfSignal.values.astype(np.float32)
  bkg_MSE=dfBkg.values.astype(np.float32)

  #et_index  = [1,2]
  #etRange = ['[20, 30]']

  #eta_index = [1,2]
  #etaRange = ['[0.6, 0.8]']

  #for iet, etrange in zip(et_index, etRange):
  #  for ieta, etarange in zip(eta_index, etaRange):
  ###iet =  etBinIdx
  ###etrange = etRange[etBinIdx]
  ###ieta = etaBinIdx
  ###etarange = etaRange[etaBinIdx]
      #sgn = data_file['signalPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
      #bkg = data_file['backgroundPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]


  fig, ax = plt.subplots(figsize=(16,10))
  ax.errorbar(np.arange(beforenorm_val_Data.shape[1]), np.mean(beforenorm_val_Data, axis=0),yerr=np.std(beforenorm_val_Data, axis=0), fmt='D-', color='crimson', label='Mean Profile')
  ax.errorbar(np.arange(unnorm_reconstruct_val_Data.shape[1]), np.mean(unnorm_reconstruct_val_Data, axis=0),yerr=np.std(unnorm_reconstruct_val_Data, axis=0), fmt='^-', color='deepskyblue', label='Mean Reconstructed Profile')
  ax2 = ax.twinx()
  ax2.errorbar(np.arange(beforenorm_val_Data.shape[1]), np.mean(allClasses_MSE, axis=0),yerr=np.std(allClasses_MSE, axis=0), fmt='go-',color='green')
  #ax2.plot(np.arange(beforenorm_val_Data.shape[1]), allClasses_MSE, marker='<', color='green')
  ax2.set_ylabel('Reconstruction Error', fontsize='xx-large')
  #print np.mean(allClasses,axis=0),np.std(allClasses,axis=0)
  #print np.mean(sgn,axis=0),np.std(sgn,axis=0)
  #print np.mean(bkg,axis=0),np.std(bkg,axis=0)
  ax2.set_yscale('log')
  ax2.set_xlabel('#Rings', fontsize= 20)
  ax2.set_ylabel('Energy [MeV]',fontsize= 20)
  ax2.tick_params(labelsize= 15)
  ax.legend(loc='best', fontsize='xx-large')

  #plt.legend(['All','Signal','Background'], loc='best', fontsize='xx-large')
  for i in [7, 71, 79, 87, 91, 95]:
    plt.axvline(i, color='gray', linestyle='--', linewidth=.8)

  plt.title(r'Reconstruction Error - '+model_name+' - '+str(layer), fontsize=24)

  #if log_scale:
  y_position = 0 #*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
  #else:
  #  y_position = 0 #*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    plt.text(x,y,text, fontsize=15, rotation=90)

  #plt.show()
  plt.savefig(dirout+'/recError_energy_'+str(layer)+'_'+model_name+'_'+time+'.png')
  png_files.append(dirout+'/recError_energy_'+str(layer)+'_'+model_name+'_'+time+'.png')
  plt.clf()
  plt.close()
  return png_files

def plot_input_reconstruction_delta(norm1Par=None,reconstruct=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
  import sqlite3
  import pandas as pd
  from numpy import nan
  #%matplotlib inline
  import matplotlib.pyplot as plt
  png_files=[]

  plt.style.use('ggplot')
  # # Et and Eta indices
  # et_index  = [0, 1, 2,3]
  # etRange = ['[15, 20]','[20, 30]','[30, 40]','[40, 50000]']

  # eta_index = [0, 1, 2, 3, 4,5,6,7,8]
  # etaRange = ['[0, 0.6]','[0.6, 0.8]','[0.8, 1.15]','[1.15, 1.37]','[1.37, 1.52]','[1.52, 1.81]','[1.81, 2.01]','[2.01, 2.37]','[2.37, 2.47]']

  beforenorm = norm1Par[0]
  normlist = norm1Par[1]
  afternorm = norm1Par[2]

  if isinstance(reconstruct[layer], (tuple, list,)):
      unnorm_reconstruct = []
      for i, cdata in enumerate(reconstruct[layer]):
          #print i,cdata.shape
          unnorm_reconstruct.append( cdata * normlist[i])
      unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
      beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

  #et_index  = [1,2]
  #etRange = ['[20, 30]']

  #eta_index = [1,2]
  #etaRange = ['[0.6, 0.8]']

  #for iet, etrange in zip(et_index, etRange):
  #  for ieta, etarange in zip(eta_index, etaRange):
  ###iet =  etBinIdx
  ###etrange = etRange[etBinIdx]
  ###ieta = etaBinIdx
  ###etarange = etaRange[etaBinIdx]
      #sgn = data_file['signalPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
      #bkg = data_file['backgroundPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]


  fig, ax = plt.subplots(figsize=(16,10))
  ax.errorbar(np.arange(beforenorm_val_Data.shape[1]), np.mean(beforenorm_val_Data, axis=0),yerr=np.std(beforenorm_val_Data, axis=0), fmt='D-', color='crimson', label='Mean Profile')
  ax.errorbar(np.arange(unnorm_reconstruct_val_Data.shape[1]), np.mean(unnorm_reconstruct_val_Data, axis=0),yerr=np.std(unnorm_reconstruct_val_Data, axis=0), fmt='^-', color='deepskyblue', label='Mean Reconstructed Profile')
  ax2 = ax.twinx()
  ax2.plot(np.arange(beforenorm_val_Data.shape[1]), (np.mean(unnorm_reconstruct_val_Data, axis=0) - np.mean(beforenorm_val_Data, axis=0))/np.std(beforenorm_val_Data, axis=0), marker='<', color='green')
  ax2.set_ylabel('Delta Energy / Sigma(Input Energy', fontsize='xx-large')
  #print np.mean(allClasses,axis=0),np.std(allClasses,axis=0)
  #print np.mean(sgn,axis=0),np.std(sgn,axis=0)
  #print np.mean(bkg,axis=0),np.std(bkg,axis=0)

  ax2.set_xlabel('#Rings', fontsize= 20)
  ax2.set_ylabel('Energy [MeV]',fontsize= 20)
  ax2.tick_params(labelsize= 15)
  ax.legend(loc='best', fontsize='xx-large')

  #plt.legend(['All','Signal','Background'], loc='best', fontsize='xx-large')
  for i in [7, 71, 79, 87, 91, 95]:
    plt.axvline(i, color='gray', linestyle='--', linewidth=.8)

  plt.title(r'Delta Energy - '+model_name+' - '+str(layer), fontsize=24)

  if log_scale:
    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
  else:
    y_position = 0 #*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    plt.text(x,y,text, fontsize=15, rotation=90)

  #plt.show()
  plt.savefig(dirout+'/delta_energy_'+str(layer)+'_'+model_name+'_'+time+'.png')
  png_files.append(dirout+'/delta_energy_'+str(layer)+'_'+model_name+'_'+time+'.png')
  plt.clf()
  plt.close()
  return png_files

def plot_input_reconstruction_delta_separed(norm1Par=None,reconstruct=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
    import sqlite3
    import pandas as pd
    from numpy import nan
    #%matplotlib inline
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    classes=['Signal','Background']
##TEM DOIS IDENTS QUE TEM QUE TIRAR. O DO FOR LAYER E O DO IF IS INSTANCES
    #for a in reconstruct.keys():

    #print 'LAYER: '+str(layer)
    #for nsort in reconstruct[layer].keys():
    #print "Sort: "+str(nsort)
    if isinstance(reconstruct[layer], (tuple, list,)):
        unnorm_reconstruct = []
        for i, cdata in enumerate(reconstruct[layer]):
            #print i,cdata.shape
            unnorm_reconstruct.append( cdata * normlist[i])


    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(25,10))
    ax1.errorbar(np.arange(beforenorm[0].shape[1]), np.mean(beforenorm[0], axis=0),yerr=np.std(beforenorm[0], axis=0), fmt='D-', color='crimson', label='Mean Profile')
    ax1.errorbar(np.arange(unnorm_reconstruct[0].shape[1]), np.mean(unnorm_reconstruct[0], axis=0),yerr=np.std(unnorm_reconstruct[0], axis=0), fmt='^-', color='deepskyblue', label='Mean Reconstructed Profile')

    ax1.set_title(r'Signal Profile',fontsize= 20)
    ax1.set_xlabel('#Rings', fontsize= 20)
    ax1.set_ylabel('Energy [MeV]',fontsize= 20)
    ax1.tick_params(labelsize= 15)
    ax1.legend(loc='best', fontsize='xx-large')
    ax12 = ax1.twinx()
    ax12.plot(np.arange(beforenorm[0].shape[1]), (np.mean(unnorm_reconstruct[0], axis=0) - np.mean(beforenorm[0], axis=0))/np.std(beforenorm[0], axis=0), marker='<', color='green')
    ax12.set_ylabel('Delta Energy / Sigma(Input Energy)', fontsize='xx-large')
    #ax12.set_ylim(top=1)

    ax2.errorbar(np.arange(beforenorm[1].shape[1]), np.mean(beforenorm[1], axis=0),yerr=np.std(beforenorm[1], axis=0), fmt='o-',color='crimson', label='Mean Profile')
    ax2.errorbar(np.arange(unnorm_reconstruct[1].shape[1]), np.mean(unnorm_reconstruct[1], axis=0),yerr=np.std(unnorm_reconstruct[1], axis=0), fmt='^-', color='deepskyblue', label='Mean Reconstructed Profile')
    ax2.set_title(r'Background Patterns',fontsize= 20)
    ax2.set_xlabel('#Rings', fontsize= 20)
    ax2.set_ylabel('Energy [MeV]',fontsize= 20)
    ax2.tick_params(labelsize= 15)
    ax2.legend(loc='best', fontsize='xx-large')
    ax22 = ax2.twinx()
    ax22.plot(np.arange(beforenorm[1].shape[1]), (np.mean(unnorm_reconstruct[1], axis=0) - np.mean(beforenorm[1], axis=0))/np.std(beforenorm[1], axis=0), marker='>', color='green')
    ax22.set_ylabel('Delta Energy / Sigma(Input Energy)', fontsize='xx-large')
    #ax22.set_ylim(top=1)
    #plt.legend(['Electron', 'Background'], loc='best', fontsize='xx-large')

    for i in [7, 71, 79, 87, 91, 95]:
        ax1.axvline(i, color='gray', linestyle='--', linewidth=.8)
        ax2.axvline(i, color='gray', linestyle='--', linewidth=.8)
    #log_scale=False
    #if log_scale:
    #  y_position = #.8*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
    #else:
    y_position = 0 #0.98 *np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)])

    for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
                      (76,y_position,r'EM2'),(80,y_position,r'EM3'),
                    (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
        ax1.text(x,y,text, fontsize=15, rotation=90)
        ax2.text(x,y,text, fontsize=15, rotation=90)

    #    plt.savefig('meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #else:
    #    plt.savefig(output_name+'_meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #plt.show()
    plt.suptitle('Delta Energy - Signal and Background- '+model_name+' - '+str(layer), fontsize=24)
    plt.savefig(dirout+'/delta_energy_separated'+str(layer)+'_'+model_name+'_'+time+'.png')
    plt.clf()
    plt.close()
    png_files.append(dirout+'/delta_energy_separated'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files


def plot_reconstruction_error(trnReconError=None,valReconError=None,model_name=None,layer=None,time=None,dirout=None):
    png_files=[]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(25,10))
    ax1.plot(np.arange(trnReconError[0].shape[1]), np.mean(trnReconError[0], axis=0), marker='<', color='crimson', label='Signal')
    ax1.plot(np.arange(trnReconError[1].shape[1]), np.mean(trnReconError[1], axis=0), marker='>', color='deepskyblue', label='Background')

    ax1.set_title(r'Training Set Reconstruction Error',fontsize= 20)
    ax1.set_xlabel('#Rings', fontsize= 20)
    ax1.set_ylabel('Reconstruction Error',fontsize= 20)
    ax1.tick_params(labelsize= 15)
    ax1.legend(loc='best', fontsize='xx-large')
    ax1.set_yscale('log')

    ax2.plot(np.arange(valReconError[0].shape[1]), np.mean(valReconError[0], axis=0),marker='<',color='crimson', label='Signal')
    ax2.plot(np.arange(valReconError[1].shape[1]), np.mean(valReconError[1], axis=0),marker='>', color='deepskyblue', label='Background')
    ax2.set_title(r'Validation Set Reconstruction Error',fontsize= 20)
    ax2.set_xlabel('#Rings', fontsize= 20)
    ax2.set_ylabel('Reconstruction Error',fontsize= 20)
    ax2.tick_params(labelsize= 15)
    ax2.legend(loc='best', fontsize='xx-large')
    ax2.set_yscale('log')

    for i in [7, 71, 79, 87, 91, 95]:
        ax1.axvline(i, color='gray', linestyle='--', linewidth=.8)
        ax2.axvline(i, color='gray', linestyle='--', linewidth=.8)
    #log_scale=False
    #if log_scale:
    y_position = .8*np.max([np.mean(valReconError[0], axis=0), np.mean(valReconError[1], axis=0)]) + 1e3
    #else:
    #y_position = 0.8*np.max([np.mean(valReconError[0], axis=0), np.mean(valReconError[1], axis=0)])

    for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
                      (76,y_position,r'EM2'),(80,y_position,r'EM3'),
                    (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
        ax1.text(x,y,text, fontsize=15, rotation=90)
        ax2.text(x,y,text, fontsize=15, rotation=90)

    #    plt.savefig('meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #else:
    #    plt.savefig(output_name+'_meanProfile_et{}_eta{}.pdf'.format(iet, ieta))
    #plt.show()
    plt.suptitle('Reconstruction Error - '+model_name+' - '+str(layer), fontsize=24)
    plt.savefig(dirout+'/recons_error'+str(layer)+'_'+model_name+'_'+time+'.png')
    plt.clf()
    plt.close()
    png_files.append(dirout+'/recons_error'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_input_reconstruction_diff_measures(model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False,Normed=False, dirout=None):
  import sqlite3
  import pandas as pd
  from numpy import nan
  #%matplotlib inline
  import matplotlib.pyplot as plt
  png_files=[]
  if Normed:
    normalizacao='yes'
  else:
    normalizacao='no'

  plt.style.use('ggplot')
  cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
  # Et and Eta indices
  et_index  = [0, 1, 2,3]
  etRange = ['[15, 20]','[20, 30]','[30, 40]','[40, 50000]']

  eta_index = [0, 1, 2, 3, 4,5,6,7,8]
  etaRange = ['[0, 0.6]','[0.6, 0.8]','[0.8, 1.15]','[1.15, 1.37]','[1.37, 1.52]','[1.52, 1.81]','[1.81, 2.01]','[2.01, 2.37]','[2.37, 2.47]']

  #et_index  = [1,2]
  #etRange = ['[20, 30]']

  #eta_index = [1,2]
  #etaRange = ['[0.6, 0.8]']

  #for iet, etrange in zip(et_index, etRange):
  #  for ieta, etarange in zip(eta_index, etaRange):
  iet =  etBinIdx
  etrange = etRange[etBinIdx]
  ieta = etaBinIdx
  etarange = etaRange[etaBinIdx]
      #sgn = data_file['signalPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
      #bkg = data_file['backgroundPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
  #measure=#Normalized_MI,MI,KLdiv,chiSquared,Correlation

  dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_MI' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  dfAll.fillna(value=nan, inplace=True)
  dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_MI' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  #dfSignal.fillna(value=nan, inplace=True)
  dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_MI' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  #dfBkg.fillna(value=nan, inplace=True)

  allClasses_NMI=dfAll.values.astype(np.float32)
  sgn_NMI=dfSignal.values.astype(np.float32)
  bkg_NMI=dfBkg.values.astype(np.float32)

  # dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  # dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  # dfAll.fillna(value=nan, inplace=True)
  # dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  # dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  # #dfSignal.fillna(value=nan, inplace=True)
  # dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  # dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  # #dfBkg.fillna(value=nan, inplace=True)

  # allClasses_MI=dfAll.values.astype(np.float32)
  # sgn_MI=dfSignal.values.astype(np.float32)
  # bkg_MI=dfBkg.values.astype(np.float32)

  dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Correlation' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  dfAll.fillna(value=nan, inplace=True)
  dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Correlation' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  #dfSignal.fillna(value=nan, inplace=True)
  dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Correlation' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

  allClasses_Corr=dfAll.values.astype(np.float32)
  sgn_Corr=dfSignal.values.astype(np.float32)
  bkg_Corr=dfBkg.values.astype(np.float32)

  dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'KLdiv' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  dfAll.fillna(value=nan, inplace=True)
  dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'KLdiv' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  #dfSignal.fillna(value=nan, inplace=True)
  dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'KLdiv' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

  allClasses_KL=dfAll.values.astype(np.float32)
  sgn_KL=dfSignal.values.astype(np.float32)
  bkg_KL=dfBkg.values.astype(np.float32)

  # dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  # dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  # dfAll.fillna(value=nan, inplace=True)
  # dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  # dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
  # #dfSignal.fillna(value=nan, inplace=True)
  # dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  # dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)

  # allClasses_Chi=dfAll.values.astype(np.float32)
  # sgn_Chi=dfSignal.values.astype(np.float32)
  # bkg_Chi=dfBkg.values.astype(np.float32)


  dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  dfAll.fillna(value=nan, inplace=True)
  dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
  #dfSignal.fillna(value=nan, inplace=True)
  dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
  dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

  allClasses_MSE=dfAll.values.astype(np.float32)
  sgn_MSE=dfSignal.values.astype(np.float32)
  bkg_MSE=dfBkg.values.astype(np.float32)
  #print 'all',allClasses
  #print 'sgn', sgn
  #print 'bkg', bkg

  fig, ax = plt.subplots(2,2,figsize=(16,10))

  ####Normalized MI

  ax[0,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_NMI, axis=0),yerr=np.std(allClasses_NMI, axis=0), fmt='go-',color='green')
  ax[0,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_NMI, axis=0),yerr=np.std(sgn_NMI, axis=0), fmt='D-', color='cornflowerblue')
  ax[0,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_NMI, axis=0),yerr=np.std(bkg_NMI, axis=0), fmt='ro-')
  ax[0,0].legend(['All','Signal','Background'], loc='best', fontsize='medium')
  for i in [7, 71, 79, 87, 91, 95]:
    ax[0,0].axvline(i, color='gray', linestyle='--', linewidth=.8)

  ax[0,0].set_title(r'Normalized Mutual Information',fontsize= 20)
  ax[0,0].set_xlabel('#Rings', fontsize='xx-large')
  ax[0,0].set_ylabel('Normalized Mutual Information', fontsize='xx-large')
  #ax[0,0].ylim(ymax=1)
  if log_scale:
    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
  else:
    y_position = .9 #*np.max([np.mean(sgn_NMI, axis=0), np.mean(bkg_NMI, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    ax[0,0].text(x,y,text, fontsize=15, rotation=90)

  ####MI

  ax[0,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_MSE, axis=0),yerr=np.std(allClasses_MSE, axis=0), fmt='go-',color='green')
  ax[0,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_MSE, axis=0),yerr=np.std(sgn_MSE, axis=0), fmt='D-', color='cornflowerblue')
  ax[0,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_MSE, axis=0),yerr=np.std(bkg_MSE, axis=0), fmt='ro-')
  ax[0,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
  for i in [7, 71, 79, 87, 91, 95]:
    ax[0,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

  ax[0,1].set_title(r'Reconstruction Error',fontsize= 20)
  ax[0,1].set_xlabel('#Rings', fontsize='xx-large')
  ax[0,1].set_ylabel('Reconstruction Error', fontsize='xx-large')
  ax[0,1].set_yscale('log')
  #ax[0,0].ylim(ymax=1)
  #if log_scale:
  y_position = .9*np.max([np.mean(sgn_MSE, axis=0), np.mean(bkg_MSE, axis=0)]) + 1e3
  #else:
  #  y_position = .9*np.max([np.mean(sgn_MSE, axis=0), np.mean(bkg_MSE, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    ax[0,1].text(x,y,text, fontsize=15, rotation=90)

  ####Corr

  ax[1,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_Corr, axis=0),yerr=np.std(allClasses_Corr, axis=0), fmt='go-',color='green')
  ax[1,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_Corr, axis=0),yerr=np.std(sgn_Corr, axis=0), fmt='D-', color='cornflowerblue')
  ax[1,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_Corr, axis=0),yerr=np.std(bkg_Corr, axis=0), fmt='ro-')
  ax[1,0].legend(['All','Signal','Background'], loc='best', fontsize='medium')
  for i in [7, 71, 79, 87, 91, 95]:
    ax[1,0].axvline(i, color='gray', linestyle='--', linewidth=.8)

  ax[1,0].set_title(r'Pearson s Correlation Coefficient',fontsize= 20)
  ax[1,0].set_xlabel('#Rings', fontsize='xx-large')
  ax[1,0].set_ylabel('Correlation Coefficient', fontsize='xx-large')
  #ax[0,0].ylim(ymax=1)
  if log_scale:
    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
  else:
    y_position = .9 #*np.max([np.mean(sgn_Corr, axis=0), np.mean(bkg_Corr, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    ax[1,0].text(x,y,text, fontsize=15, rotation=90)

  ####KLDiv

  ax[1,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_KL, axis=0),yerr=np.std(allClasses_KL, axis=0), fmt='go-',color='green')
  ax[1,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_KL, axis=0),yerr=np.std(sgn_KL, axis=0), fmt='D-', color='cornflowerblue')
  ax[1,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_KL, axis=0),yerr=np.std(bkg_KL, axis=0), fmt='ro-')
  ax[1,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
  for i in [7, 71, 79, 87, 91, 95]:
    ax[1,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

  ax[1,1].set_title(r'Kullback-Leibler Divergence',fontsize= 20)
  ax[1,1].set_xlabel('#Rings', fontsize='xx-large')
  ax[1,1].set_ylabel('KL Divergence', fontsize='xx-large')
  #ax[0,0].ylim(ymax=1)
  if log_scale:
    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
  else:
    y_position = .9*np.max([np.mean(sgn_KL, axis=0), np.mean(bkg_KL, axis=0)])

  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
    ax[1,1].text(x,y,text, fontsize=15, rotation=90)

  ####ChiSquared

#  ax[1,1].errorbar(np.arange(100), np.mean(allClasses_Chi, axis=0),yerr=np.std(allClasses_Chi, axis=0), fmt='go-',color='green')
#  ax[1,1].errorbar(np.arange(100), np.mean(sgn_Chi, axis=0),yerr=np.std(sgn_Chi, axis=0), fmt='D-', color='cornflowerblue')
#  ax[1,1].errorbar(np.arange(100), np.mean(bkg_Chi, axis=0),yerr=np.std(bkg_Chi, axis=0), fmt='ro-')
#  ax[1,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
#  for i in [7, 71, 79, 87, 91, 95]:
#    ax[1,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

#  ax[1,1].set_title(r'Chi Squared',fontsize= 20)
#  ax[1,1].set_xlabel('#Rings', fontsize='xx-large')
#  ax[1,1].set_ylabel('Chi Squared', fontsize='xx-large')
#  #ax[0,0].ylim(ymax=1)
#  if log_scale:
#    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
#  else:
#    y_position = .9*np.max([np.mean(sgn_Chi, axis=0), np.mean(bkg_Chi, axis=0)])
#
#  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
#           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
#          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
#    ax[1,1].text(x,y,text, fontsize=15, rotation=90)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  if Normed:
    prefix='Normalized '
  else:
    prefix=''
  plt.suptitle(prefix+'Input X Reconstruction - '+model_name+' - '+str(layer)+' - $E_T$={} $\eta$={}'.format(etrange,etarange), fontsize=24)
  plt.savefig(dirout+'/measures_'+str(layer)+'_'+model_name+'_'+time+'.png')
  plt.clf()
  plt.close()
  png_files.append(dirout+'/measures_'+str(layer)+'_'+model_name+'_'+time+'.png')
  return png_files

def plot_input_reconstruction_diff_measures2(model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False,Normed=False, dirout=None):
   import sqlite3
   import pandas as pd
   from numpy import nan
   #%matplotlib inline
   import matplotlib.pyplot as plt
   png_files=[]
   if Normed:
     normalizacao='yes'
   else:
     normalizacao='no'

   plt.style.use('ggplot')
   cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
   # Et and Eta indices
   et_index  = [0, 1, 2,3]
   etRange = ['[15, 20]','[20, 30]','[30, 40]','[40, 50000]']

   eta_index = [0, 1, 2, 3, 4,5,6,7,8]
   etaRange = ['[0, 0.6]','[0.6, 0.8]','[0.8, 1.15]','[1.15, 1.37]','[1.37, 1.52]','[1.52, 1.81]','[1.81, 2.01]','[2.01, 2.37]','[2.37, 2.47]']

   #et_index  = [1,2]
   #etRange = ['[20, 30]']

   #eta_index = [1,2]
   #etaRange = ['[0.6, 0.8]']

   #for iet, etrange in zip(et_index, etRange):
   #  for ieta, etarange in zip(eta_index, etaRange):
   iet =  etBinIdx
   etrange = etRange[etBinIdx]
   ieta = etaBinIdx
   etarange = etaRange[etaBinIdx]
       #sgn = data_file['signalPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
       #bkg = data_file['backgroundPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
   #measure=#Normalized_MI,MI,KLdiv,chiSquared,Correlation

   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_MI' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_MI' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_MI' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfBkg.fillna(value=nan, inplace=True)

   allClasses_NMI=dfAll.values.astype(np.float32)
   sgn_NMI=dfSignal.values.astype(np.float32)
   bkg_NMI=dfBkg.values.astype(np.float32)

   # dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # dfAll.fillna(value=nan, inplace=True)
   # dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # #dfSignal.fillna(value=nan, inplace=True)
   # dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # #dfBkg.fillna(value=nan, inplace=True)

   # allClasses_MI=dfAll.values.astype(np.float32)
   # sgn_MI=dfSignal.values.astype(np.float32)
   # bkg_MI=dfBkg.values.astype(np.float32)

   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Wasserstein' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Wasserstein' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Wasserstein' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

   allClasses_Wass=dfAll.values.astype(np.float32)
   sgn_Wass=dfSignal.values.astype(np.float32)
   bkg_Wass=dfBkg.values.astype(np.float32)

   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'KLdiv' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'KLdiv' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'KLdiv' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

   allClasses_KL=dfAll.values.astype(np.float32)
   sgn_KL=dfSignal.values.astype(np.float32)
   bkg_KL=dfBkg.values.astype(np.float32)

   # dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # dfAll.fillna(value=nan, inplace=True)
   # dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # #dfSignal.fillna(value=nan, inplace=True)
   # dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)

   # allClasses_Chi=dfAll.values.astype(np.float32)
   # sgn_Chi=dfSignal.values.astype(np.float32)
   # bkg_Chi=dfBkg.values.astype(np.float32)


   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'kolmogorov-smirnov' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'kolmogorov-smirnov' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'kolmogorov-smirnov' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

   allClasses_KS=dfAll.values.astype(np.float32)
   sgn_KS=dfSignal.values.astype(np.float32)
   bkg_KS=dfBkg.values.astype(np.float32)
   #print 'all',allClasses
   #print 'sgn', sgn
   #print 'bkg', bkg

   fig, ax = plt.subplots(2,2,figsize=(16,10))

   ####Normalized MI

   ax[0,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_NMI, axis=0),yerr=np.std(allClasses_NMI, axis=0), fmt='go-',color='green')
   ax[0,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_NMI, axis=0),yerr=np.std(sgn_NMI, axis=0), fmt='D-', color='cornflowerblue')
   ax[0,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_NMI, axis=0),yerr=np.std(bkg_NMI, axis=0), fmt='ro-')
   ax[0,0].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[0,0].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[0,0].set_title(r'Normalized Mutual Information',fontsize= 20)
   ax[0,0].set_xlabel('#Rings', fontsize='xx-large')
   ax[0,0].set_ylabel('Normalized Mutual Information', fontsize='xx-large')
   #ax[0,0].ylim(ymax=1)
   if log_scale:
     y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
   else:
     y_position = .9 #*np.max([np.mean(sgn_NMI, axis=0), np.mean(bkg_NMI, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[0,0].text(x,y,text, fontsize=15, rotation=90)

   ####MI

   ax[0,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_KS, axis=0),yerr=np.std(allClasses_KS, axis=0), fmt='go-',color='green')
   ax[0,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_KS, axis=0),yerr=np.std(sgn_KS, axis=0), fmt='D-', color='cornflowerblue')
   ax[0,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_KS, axis=0),yerr=np.std(bkg_KS, axis=0), fmt='ro-')
   ax[0,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[0,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[0,1].set_title(r'Kolmogorov Smirnov Distance',fontsize= 20)
   ax[0,1].set_xlabel('#Rings', fontsize='xx-large')
   ax[0,1].set_ylabel('Kolmogorov Smirnov', fontsize='xx-large')
   #ax[0,1].set_yscale('log')
   #ax[0,0].ylim(ymax=1)
   #if log_scale:
   y_position = .9*np.max([np.mean(sgn_KS, axis=0), np.mean(bkg_KS, axis=0)]) + 1e3
   #else:
   #  y_position = .9*np.max([np.mean(sgn_MSE, axis=0), np.mean(bkg_MSE, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[0,1].text(x,y,text, fontsize=15, rotation=90)

   ####Corr

   ax[1,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_Wass, axis=0),yerr=np.std(allClasses_Wass, axis=0), fmt='go-',color='green')
   ax[1,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_Wass, axis=0),yerr=np.std(sgn_Wass, axis=0), fmt='D-', color='cornflowerblue')
   ax[1,0].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_Wass, axis=0),yerr=np.std(bkg_Wass, axis=0), fmt='ro-')
   ax[1,0].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[1,0].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[1,0].set_title(r'Wasserstein Distance',fontsize= 20)
   ax[1,0].set_xlabel('#Rings', fontsize='xx-large')
   ax[1,0].set_ylabel('Wasserstein Distance', fontsize='xx-large')
   ax[1,0].set_yscale('log')
   #ax[0,0].ylim(ymax=1)
   if log_scale:
     y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
   else:
     y_position = .9 #*np.max([np.mean(sgn_Corr, axis=0), np.mean(bkg_Corr, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[1,0].text(x,y,text, fontsize=15, rotation=90)

   ####KLDiv

   ax[1,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(allClasses_KL, axis=0),yerr=np.std(allClasses_KL, axis=0), fmt='go-',color='green')
   ax[1,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(sgn_KL, axis=0),yerr=np.std(sgn_KL, axis=0), fmt='D-', color='cornflowerblue')
   ax[1,1].errorbar(np.arange(allClasses_NMI.shape[1]), np.mean(bkg_KL, axis=0),yerr=np.std(bkg_KL, axis=0), fmt='ro-')
   ax[1,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[1,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[1,1].set_title(r'Kullback-Leibler Divergence',fontsize= 20)
   ax[1,1].set_xlabel('#Rings', fontsize='xx-large')
   ax[1,1].set_ylabel('KL Divergence', fontsize='xx-large')
   #ax[0,0].ylim(ymax=1)
   if log_scale:
     y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
   else:
     y_position = .9*np.max([np.mean(sgn_KL, axis=0), np.mean(bkg_KL, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[1,1].text(x,y,text, fontsize=15, rotation=90)

   ####ChiSquared

 #  ax[1,1].errorbar(np.arange(100), np.mean(allClasses_Chi, axis=0),yerr=np.std(allClasses_Chi, axis=0), fmt='go-',color='green')
 #  ax[1,1].errorbar(np.arange(100), np.mean(sgn_Chi, axis=0),yerr=np.std(sgn_Chi, axis=0), fmt='D-', color='cornflowerblue')
 #  ax[1,1].errorbar(np.arange(100), np.mean(bkg_Chi, axis=0),yerr=np.std(bkg_Chi, axis=0), fmt='ro-')
 #  ax[1,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
 #  for i in [7, 71, 79, 87, 91, 95]:
 #    ax[1,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

 #  ax[1,1].set_title(r'Chi Squared',fontsize= 20)
 #  ax[1,1].set_xlabel('#Rings', fontsize='xx-large')
 #  ax[1,1].set_ylabel('Chi Squared', fontsize='xx-large')
 #  #ax[0,0].ylim(ymax=1)
 #  if log_scale:
 #    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
 #  else:
 #    y_position = .9*np.max([np.mean(sgn_Chi, axis=0), np.mean(bkg_Chi, axis=0)])
 #
 #  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
 #           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
 #          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
 #    ax[1,1].text(x,y,text, fontsize=15, rotation=90)
   plt.tight_layout(rect=[0, 0.03, 1, 0.95])
   if Normed:
     prefix='Normalized '
   else:
     prefix=''
   plt.suptitle(prefix+'Input X Reconstruction - '+model_name+' - '+str(layer)+' - $E_T$={} $\eta$={}'.format(etrange,etarange), fontsize=24)
   plt.savefig(dirout+'/measures_2_'+str(layer)+'_'+model_name+'_'+time+'.png')
   plt.clf()
   plt.close()
   png_files.append(dirout+'/measures_2_'+str(layer)+'_'+model_name+'_'+time+'.png')
   return png_files

def plot_input_reconstruction_diff_measures3(model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False,Normed=False, dirout=None):
   import sqlite3
   import pandas as pd
   from numpy import nan
   #%matplotlib inline
   import matplotlib.pyplot as plt
   png_files=[]
   if Normed:
     normalizacao='yes'
   else:
     normalizacao='no'

   plt.style.use('ggplot')
   cnx = sqlite3.connect('//home/caducovas/run/ringerMLlab.db')
   # Et and Eta indices
   et_index  = [0, 1, 2,3]
   etRange = ['[15, 20]','[20, 30]','[30, 40]','[40, 50000]']

   eta_index = [0, 1, 2, 3, 4,5,6,7,8]
   etaRange = ['[0, 0.6]','[0.6, 0.8]','[0.8, 1.15]','[1.15, 1.37]','[1.37, 1.52]','[1.52, 1.81]','[1.81, 2.01]','[2.01, 2.37]','[2.37, 2.47]']

   #et_index  = [1,2]
   #etRange = ['[20, 30]']

   #eta_index = [1,2]
   #etaRange = ['[0.6, 0.8]']

   #for iet, etrange in zip(et_index, etRange):
   #  for ieta, etarange in zip(eta_index, etaRange):
   iet =  etBinIdx
   etrange = etRange[etBinIdx]
   ieta = etaBinIdx
   etarange = etaRange[etaBinIdx]
       #sgn = data_file['signalPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
       #bkg = data_file['backgroundPatterns_etBin_%i_etaBin_%i' %(iet, ieta)]
   #measure=#Normalized_MI,MI,KLdiv,chiSquared,Correlation

   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MSE' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfBkg.fillna(value=nan, inplace=True)

   allClasses_MSE=dfAll.values.astype(np.float32)
   sgn_MSE=dfSignal.values.astype(np.float32)
   bkg_MSE=dfBkg.values.astype(np.float32)

   # dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # dfAll.fillna(value=nan, inplace=True)
   # dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # #dfSignal.fillna(value=nan, inplace=True)
   # dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'MI' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # #dfBkg.fillna(value=nan, inplace=True)

   # allClasses_MI=dfAll.values.astype(np.float32)
   # sgn_MI=dfSignal.values.astype(np.float32)
   # bkg_MI=dfBkg.values.astype(np.float32)

   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'DeltaEnergy' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'DeltaEnergy' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'DeltaEnergy' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

   allClasses_Del=dfAll.values.astype(np.float32)
   sgn_Del=dfSignal.values.astype(np.float32)
   bkg_Del=dfBkg.values.astype(np.float32)

   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_DeltaEnergy' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_DeltaEnergy' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Normalized_DeltaEnergy' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

   allClasses_NDel=dfAll.values.astype(np.float32)
   sgn_NDel=dfSignal.values.astype(np.float32)
   bkg_NDel=dfBkg.values.astype(np.float32)

   # dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # dfAll.fillna(value=nan, inplace=True)
   # dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)
   # #dfSignal.fillna(value=nan, inplace=True)
   # dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'chiSquared' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   # dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase'],axis=1)

   # allClasses_Chi=dfAll.values.astype(np.float32)
   # sgn_Chi=dfSignal.values.astype(np.float32)
   # bkg_Chi=dfBkg.values.astype(np.float32)


   dfAll = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Correlation' and Class = 'All' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfAll=dfAll.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   dfAll.fillna(value=nan, inplace=True)
   dfSignal = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Correlation' and Class = 'Signal' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"'  and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfSignal=dfSignal.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)
   #dfSignal.fillna(value=nan, inplace=True)
   dfBkg = pd.read_sql_query("SELECT * FROM reconstruction_metrics where time > 201809000000 and Measure = 'Correlation' and Class = 'Background' and Normed='"+str(normalizacao)+"' and layer = '"+str(layer)+"' and Model= '"+model_name+"' and time = '"+time+"'", cnx)
   dfBkg=dfBkg.drop(labels=['id','Class','Layer','Model','time','Measure','Normed','sort','etBinIdx','etaBinIdx','phase','ETotal','PS','EM1','EM2','EM3','EM','HAD1','HAD2','HAD3','HAD'],axis=1)

   allClasses_Corr=dfAll.values.astype(np.float32)
   sgn_Corr=dfSignal.values.astype(np.float32)
   bkg_Corr=dfBkg.values.astype(np.float32)
   #print 'all',allClasses
   #print 'sgn', sgn
   #print 'bkg', bkg

   fig, ax = plt.subplots(2,2,figsize=(16,10))

   ####Normalized MI

   ax[0,0].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(allClasses_MSE, axis=0),yerr=np.std(allClasses_MSE, axis=0), fmt='go-',color='green')
   ax[0,0].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(sgn_MSE, axis=0),yerr=np.std(sgn_MSE, axis=0), fmt='D-', color='cornflowerblue')
   ax[0,0].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(bkg_MSE, axis=0),yerr=np.std(bkg_MSE, axis=0), fmt='ro-')
   ax[0,0].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[0,0].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[0,0].set_title(r'Reconstruction Error',fontsize= 20)
   ax[0,0].set_xlabel('#Rings', fontsize='xx-large')
   ax[0,0].set_ylabel('MSE', fontsize='xx-large')
   ax[0,0].set_yscale('log')
   #ax[0,0].ylim(ymax=1)
   if log_scale:
     y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
   else:
     y_position = .9 #*np.max([np.mean(sgn_MSE, axis=0), np.mean(bkg_MSE, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[0,0].text(x,y,text, fontsize=15, rotation=90)

   ####MI

   ax[0,1].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(allClasses_Corr, axis=0),yerr=np.std(allClasses_Corr, axis=0), fmt='go-',color='green')
   ax[0,1].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(sgn_Corr, axis=0),yerr=np.std(sgn_Corr, axis=0), fmt='D-', color='cornflowerblue')
   ax[0,1].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(bkg_Corr, axis=0),yerr=np.std(bkg_Corr, axis=0), fmt='ro-')
   ax[0,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[0,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[0,1].set_title(r'Correlation Coefficient',fontsize= 20)
   ax[0,1].set_xlabel('#Rings', fontsize='xx-large')
   ax[0,1].set_ylabel('Pearson Correlation coeficient', fontsize='xx-large')
   #ax[0,1].set_yscale('log')
   #ax[0,0].ylim(ymax=1)
   #if log_scale:
   y_position = .9*np.max([np.mean(sgn_Corr, axis=0), np.mean(bkg_Corr, axis=0)]) + 1e3
   #else:
   #  y_position = .9*np.max([np.mean(sgn_MSE, axis=0), np.mean(bkg_MSE, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[0,1].text(x,y,text, fontsize=15, rotation=90)

   ####Corr

   ax[1,0].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(allClasses_Del, axis=0),yerr=np.std(allClasses_Del, axis=0), fmt='go-',color='green')
   ax[1,0].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(sgn_Del, axis=0),yerr=np.std(sgn_Del, axis=0), fmt='D-', color='cornflowerblue')
   ax[1,0].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(bkg_Del, axis=0),yerr=np.std(bkg_Del, axis=0), fmt='ro-')
   ax[1,0].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[1,0].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[1,0].set_title(r'Delta Energy',fontsize= 20)
   ax[1,0].set_xlabel('#Rings', fontsize='xx-large')
   ax[1,0].set_ylabel('Delta Et', fontsize='xx-large')
   #ax[0,0].ylim(ymax=1)
   if log_scale:
     y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
   else:
     y_position = .9 #*np.max([np.mean(sgn_Corr, axis=0), np.mean(bkg_Corr, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[1,0].text(x,y,text, fontsize=15, rotation=90)

   ####KLDiv

   ax[1,1].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(allClasses_NDel, axis=0),yerr=np.std(allClasses_NDel, axis=0), fmt='go-',color='green')
   ax[1,1].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(sgn_NDel, axis=0),yerr=np.std(sgn_NDel, axis=0), fmt='D-', color='cornflowerblue')
   ax[1,1].errorbar(np.arange(allClasses_MSE.shape[1]), np.mean(bkg_NDel, axis=0),yerr=np.std(bkg_NDel, axis=0), fmt='ro-')
   ax[1,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
   for i in [7, 71, 79, 87, 91, 95]:
     ax[1,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

   ax[1,1].set_title(r'Normalized Delta Energy',fontsize= 20)
   ax[1,1].set_xlabel('#Rings', fontsize='xx-large')
   ax[1,1].set_ylabel('Normalized Delta Et', fontsize='xx-large')
   #ax[0,0].ylim(ymax=1)
   if log_scale:
     y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
   else:
     y_position = .9*np.max([np.mean(sgn_NDel, axis=0), np.mean(bkg_NDel, axis=0)])

   for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
            (76,y_position,r'EM2'),(80,y_position,r'EM3'),
           (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
     ax[1,1].text(x,y,text, fontsize=15, rotation=90)

   ####ChiSquared

 #  ax[1,1].errorbar(np.arange(100), np.mean(allClasses_Chi, axis=0),yerr=np.std(allClasses_Chi, axis=0), fmt='go-',color='green')
 #  ax[1,1].errorbar(np.arange(100), np.mean(sgn_Chi, axis=0),yerr=np.std(sgn_Chi, axis=0), fmt='D-', color='cornflowerblue')
 #  ax[1,1].errorbar(np.arange(100), np.mean(bkg_Chi, axis=0),yerr=np.std(bkg_Chi, axis=0), fmt='ro-')
 #  ax[1,1].legend(['All','Signal','Background'], loc='best', fontsize='medium')
 #  for i in [7, 71, 79, 87, 91, 95]:
 #    ax[1,1].axvline(i, color='gray', linestyle='--', linewidth=.8)

 #  ax[1,1].set_title(r'Chi Squared',fontsize= 20)
 #  ax[1,1].set_xlabel('#Rings', fontsize='xx-large')
 #  ax[1,1].set_ylabel('Chi Squared', fontsize='xx-large')
 #  #ax[0,0].ylim(ymax=1)
 #  if log_scale:
 #    y_position = .9#*np.max([np.mean(sgn, axis=0), np.mean(bkg, axis=0)]) + 1e3
 #  else:
 #    y_position = .9*np.max([np.mean(sgn_Chi, axis=0), np.mean(bkg_Chi, axis=0)])
 #
 #  for x,y,text in [(2,y_position,r'PS'), (8,y_position,r'EM1'),
 #           (76,y_position,r'EM2'),(80,y_position,r'EM3'),
 #          (88,y_position,r'HAD1'), (92,y_position,r'HAD2'), (96,y_position,r'HAD3'),]:
 #    ax[1,1].text(x,y,text, fontsize=15, rotation=90)
   plt.tight_layout(rect=[0, 0.03, 1, 0.95])
   if Normed:
     prefix='Normalized '
   else:
     prefix=''
   plt.suptitle(prefix+'Input X Reconstruction - '+model_name+' - '+str(layer)+' - $E_T$={} $\eta$={}'.format(etrange,etarange), fontsize=24)
   plt.savefig(dirout+'/measures_3_'+str(layer)+'_'+model_name+'_'+time+'.png')
   plt.clf()
   plt.close()
   png_files.append(dirout+'/measures_3_'+str(layer)+'_'+model_name+'_'+time+'.png')
   return png_files

def mutualInformation_matrix(signal,measure='MI',sklearn=True,kde=False, n_bins=None):
    #from statsmodels.sandbox.distributions.mv_measures import mutualinfo_kde, mutualinfo_binned
    print signal.shape
    cols,rows = signal.shape
    mat = np.zeros((rows, rows))
    #np.fill_diagonal(mat, 100)
    # Upper diagonal
    for r in range(rows):
        for c in range(r, rows):
            #print r,c
            #if r == c:
            #    continue

            if sklearn:
                try:
                    if measure == 'MI':
                        rr = calc_MI2(signal[:,r]+ 1e-12,signal[:,c]+ 1e-12)
                        score = round(np.sqrt(1. - np.exp(-2 * rr)),4)
                    elif measure == 'KL':
                        score = calc_kl(signal[:,r]+ 1e-12,signal[:,c]+ 1e-12)
                    elif measure == 'Wasserstein':
                        score = wasserstein_distance(signal[:,r],signal[:,c])
                except:
                    score=None
            else:
                if n_bins is None:
                    p_bins = len(np.histogram(p, bins='fd')[0])
                    q_bins = len(np.histogram(q, bins='fd')[0])
                    n_bins = min(p_bins, q_bins)
                elif n_bins == 'auto':
                    qs = np.sort(q)
                    ps = np.sort(p)
                    qbin_sqr = np.sqrt(5./cols)
                    quantiles = np.linspace(0, 1, 1./qbin_sqr)
                    quantile_index = ((cols-1)*quantiles).astype(int)
                    #move edges so that they don't coincide with an observation
                    shift = 1e-6 + np.ones(quantiles.shape)
                    shift[0] -= 2*1e-6
                    q_bins = qs[quantile_index] + shift
                    p_bins = ps[quantile_index] + shift
                    n_bins = min(p_bins, q_bins)

                #mi = mutualinfo_binned(p, q, n_bins)[0]

                fx, binsx = np.histogram(p, bins = p_bins)
                fy, binsy = np.histogram(q, bins = q_bins)
                fyx, binsy, binsx = np.histogram2d(q, p, bins = (binsy, binsx))

                pyx = fyx * 1. / cols
                px = fx * 1. / cols
                py = fy * 1. / cols


                mi_obs = pyx * (np.log(pyx + 1e-12) - np.log(py + 1e-12)[:,None] - np.log(px + 1e-12))
                mi_obs[np.isnan(mi_obs)] = 0
                mi_obs[np.isinf(mi_obs)] = 0
                score = mi_obs.sum()
            import math
            #print mi
            #if math.isnan(float(mi)):
            #    mi = 100
            #elif mi > 1:
            #    mi = 1
            mat[r][c] = score
            mat[c][r] = score
    #print mat
    return mat

def plot_MutualInformation(mixtures=None,reconstruction=None,measure='MI',sae=None,plot_corr=False , KDE = False, nbins = None):
    import seaborn as sb

    if reconstruction is None:
        if plot_corr:
            fig, axs = plt.subplots(1,2, figsize=(40, 18))
        else:
            fig, axs = plt.subplots(figsize=(24, 18))
    else:
        if plot_corr:
            fig, axs = plt.subplots(2,2, figsize=(36, 18))
        else:
            fig, axs = plt.subplots(1,2, figsize=(40, 18))

    if measure == 'KL':
        mat = mutualInformation_matrix(mixtures,measure='KL')
        minval=min(mat.flatten())
        maxval=max(mat.flatten())
    elif measure == 'Wasserstein':
        mat = mutualInformation_matrix(mixtures,measure='Wasserstein')
        minval=min(mat.flatten())
        maxval=max(mat.flatten())

    else:
        minval=0
        maxval=1
    if reconstruction is None:
        if plot_corr:
            sb.heatmap(mutualInformation_matrix(mixtures,measure=measure, kde=KDE, n_bins = nbins), ax=axs[0], annot=False, cmap = 'YlGnBu', vmin = minval, vmax = maxval)
            axs[0].set_title('Mutual Information',fontsize=20)
            #axs[0].set_tick_params(labelsize=20)

            sb.heatmap(np.abs(np.corrcoef(mixtures.T)), annot=False, cmap = 'YlOrRd', ax = axs[1], vmin = 0, vmax =  1)
            axs[1].set_title('Abs Correlation matrix',fontsize=20)
            #axs[1].set_tick_params(labelsize=20)
        else:
            sb.heatmap(mutualInformation_matrix(mixtures,measure=measure, kde=KDE, n_bins = nbins), annot=False, cmap = 'YlGnBu',  vmin = minval, vmax = maxval)
            axs.set_title('Mutual Information',fontsize=20)
    else:
        if plot_corr:
            sb.heatmap(mutualInformation_matrix(mixtures,measure=measure, kde=KDE, n_bins = nbins), ax=axs[0,0], annot=False, cmap = 'YlGnBu',  vmin = minval, vmax = maxval)
            axs[0,0].set_title('Signal Mutual Information')

            sb.heatmap(np.abs(np.corrcoef(mixtures.T)), annot=False, cmap = 'YlOrRd', ax = axs[0,1], vmin = 0, vmax =  1)
            axs[0,1].set_title('Signal Abs Correlation matrix')

            sb.heatmap(mutualInformation_matrix(reconstruction,measure=measure, kde=KDE, n_bins = nbins), ax=axs[1,0], annot=False, cmap = 'YlGnBu',  vmin = minval, vmax = maxval)
            axs[1,0].set_title('Reconstruction Mutual Information')

            sb.heatmap(np.abs(np.corrcoef(reconstruction.T)), annot=False, cmap = 'YlOrRd', ax = axs[1,1], vmin = 0, vmax =  1)
            axs[1,1].set_title('Reconstruction Abs Correlation matrix')

        else:
            sb.heatmap(mutualInformation_matrix(mixtures,measure=measure, kde=KDE, n_bins = nbins),ax=axs[0] , annot=False, cmap = 'YlGnBu',  vmin = minval, vmax = maxval)
            axs[0].set_title('Signal Mutual Information')

            sb.heatmap(mutualInformation_matrix(reconstruction,measure=measure, kde=KDE, n_bins = nbins),ax=axs[1] , annot=False, cmap = 'YlGnBu',  vmin = minval, vmax = maxval)
            axs[1].set_title('Reconstruction Mutual Information')

    return fig

def plot_measures_2d(norm1Par=None,reconstruct=None,model_name="",layer=None,time=None,sort=None,etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
    import matplotlib.pyplot as plt
    import seaborn as sb

    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    #for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
    if isinstance(reconstruct[layer], (tuple, list,)):
        unnorm_reconstruct = []
        for i, cdata in enumerate(reconstruct[layer]):
            #print i,cdata.shape
            unnorm_reconstruct.append( cdata * normlist[i])
        unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
        beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )
        r=unnorm_reconstruct_val_Data
        b=beforenorm_val_Data

    fig = plot_MutualInformation(b,r,measure='MI',plot_corr=True)
    fig.suptitle('MI and Correlation Matrix - Input X Reconstruction - '+model_name+' - '+str(layer), fontsize=24)
    fig.savefig(dirout+'/matrix2d_'+str(layer)+'_'+model_name+'_'+time+'.png')
    fig.clf()
    #plt.close()
    png_files.append(dirout+'/matrix2d_'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def plot_representation_2d(norm1Par=None,code=None,model_name="",layer=None,time=None,sort=None,etBinIdx=None,etaBinIdx=None,log_scale=False, dirout=None):
    import matplotlib.pyplot as plt
    import seaborn as sb

    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    #for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
    if isinstance(code[layer], (tuple, list,)):
        unnorm_code = []
        for i, cdata in enumerate(code[layer]):
            #print i,cdata.shape
            unnorm_code.append( cdata * normlist[i])
        unnorm_code_val_Data = np.concatenate( unnorm_code, axis=0 )
        beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

        code_val_Data = np.concatenate( code[layer], axis=0 )
        afternorm_val_Data = np.concatenate( afternorm, axis=0 )

        r=code_val_Data
        b=afternorm_val_Data

    fig = plot_MutualInformation(b,r,measure='MI',plot_corr=True)
    fig.suptitle('MI and Correlation Matrix - Input X Representation - '+model_name+' - '+str(layer), fontsize=24)
    fig.savefig(dirout+'/codematrix2d_'+str(layer)+'_'+model_name+'_'+time+'.png')
    fig.clf()
    #plt.close()
    png_files.append(dirout+'/codematrix2d_'+str(layer)+'_'+model_name+'_'+time+'.png')
    return png_files

def make_ring_hist(norm1Par=None,reconstruct=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout='//home/caducovas/run/plots/'):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    #for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
    if isinstance(reconstruct[layer], (tuple, list,)):
        unnorm_reconstruct = []
        for i, cdata in enumerate(reconstruct[layer]):
            #print i,cdata.shape
            unnorm_reconstruct.append( cdata * normlist[i])
        unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
        beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

      # reconstruct_val_Data = np.concatenate( reconstruct[layer], axis=0 )
      # afternorm_val_Data = np.concatenate( afternorm, axis=0 )

      # if Normed:
        # input_val_Data=afternorm_val_Data
        # reconstruct_val_Data=reconstruct_val_Data
        # input=afternorm
        # reconstructed=reconstruct[layer]
        # normalizacao='yes'
      # else:
        # input_val_Data=beforenorm_val_Data
        # reconstruct_val_Data=unnorm_reconstruct_val_Data
        # input=beforenorm
        # reconstructed=unnorm_reconstruct
        # normalizacao='no'

    ###All Classes
    r=unnorm_reconstruct_val_Data
    b=beforenorm_val_Data

    diroutAllclasses = dirout+model_name+'_'+time+'/allClasses/'
    if not os.path.exists(diroutAllclasses):
        print 'Creating output folder AllClasses...'
        os.makedirs(diroutAllclasses)
    plt.clf()


    for rings in range(b.shape[1]):
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            rr = calc_MI2(b[:,rings],r[:,rings])
            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
            #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
            mse_score = mean_squared_error(b[:,rings],r[:,rings])
            corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
            at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
            #at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(b[:,rings]),max(r[:,rings]))
        min_value = min(min(b[:,rings]),min(r[:,rings]))
        bins = min( len(np.histogram(b[:,rings],'fd')[0]), len(np.histogram(r[:,rings],'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(b[:,rings], bins=bins_list, alpha=.5, color='b', label='Input Energy')
        plt.hist(r[:,rings], bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
        #plt.axvline(np.max(b[:,rings]), color='b', linestyle='--', linewidth=.8, label='Max Input Energy')
        #plt.axvline(np.max(r[:,rings]), color='r', linestyle='--', linewidth=.8, label='Max Reconstructed Energy')
        plt.title(r'Rings number: '+str(rings+1)+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutAllclasses+'/'+'hist_et{}_eta{}_ring{}.png'.format(etBinIdx, etaBinIdx, rings+1))
        plt.clf()
        plt.close()

    for segment, inputVector,reconstructVector in [('ETotal',b.sum(axis=1),r.sum(axis=1)),('PS',b[:,0:7,].sum(axis=1),r[:,0:7,].sum(axis=1)),('EM1',b[:,8:71,].sum(axis=1),r[:,8:71,].sum(axis=1)),('EM2',b[:,72:79,].sum(axis=1),r[:,72:79,].sum(axis=1)),('EM3',b[:,80:87,].sum(axis=1),r[:,80:87,].sum(axis=1)),('EM',b[:,0:87,].sum(axis=1),r[:,0:87,].sum(axis=1)),('HAD1',b[:,88:91,].sum(axis=1),r[:,88:91,].sum(axis=1)),('HAD2',b[:,92:95,].sum(axis=1),r[:,92:95,].sum(axis=1)),('HAD3',b[:,96:99,].sum(axis=1),r[:,96:99,].sum(axis=1)),('HAD',b[:,88:99,].sum(axis=1),r[:,88:99,].sum(axis=1))]:
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            rr = calc_MI2(inputVector,reconstructVector)
            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            kl_score = round(calc_kl(inputVector,reconstructVector),4)
            #chi_score,chi_pvalue =calc_chisquare(inputVector,reconstructVector)
            mse_score = mean_squared_error(inputVector,reconstructVector)
            corr_score,corr_pvalue= scipy.stats.pearsonr(inputVector,reconstructVector)
            at = AnchoredText('Input \nMean: '+str(round(inputVector.mean(),2))+"\nStd: "+str(round(inputVector.std(),2))+"\nSkw: "+str(round(skew(inputVector),2))+"\nKur: "+str(round(kurtosis(inputVector),2))+"\n\nReconstructed \nMean: "+str(round(reconstructVector.mean(),2))+"\nStd: "+str(round(reconstructVector.std(),2))+"\nSkw: "+str(round(skew(reconstructVector),2))+"\nKur: "+str(round(kurtosis(reconstructVector),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
            #at = AnchoredText('Input \nMean: '+str(round(inputVector.mean(),2))+"\nStd: "+str(round(inputVector.std(),2))+"\nSkw: "+str(round(skew(inputVector),2))+"\nKur: "+str(round(kurtosis(inputVector),2))+"\n\nReconstructed \nMean: "+str(round(reconstructVector.mean(),2))+"\nStd: "+str(round(reconstructVector.std(),2))+"\nSkw: "+str(round(skew(reconstructVector),2))+"\nKur: "+str(round(kurtosis(reconstructVector),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(inputVector),max(reconstructVector))
        min_value = min(min(inputVector),min(reconstructVector))
        bins = min( len(np.histogram(inputVector,'fd')[0]), len(np.histogram(reconstructVector,'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(inputVector, bins=bins_list, alpha=.5, color='b', label='Input Energy')
        plt.hist(reconstructVector, bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
        #plt.axvline(np.max(inputVector), color='b', linestyle='--', linewidth=.8, label='Max Input Energy')
        #plt.axvline(np.max(reconstructVector), color='r', linestyle='--', linewidth=.8, label='Max Reconstructed Energy')
        plt.title(segment+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutAllclasses+'/'+'hist_et{}_eta{}_'+segment+'.png'.format(etBinIdx, etaBinIdx))
        plt.clf()
        plt.close()

    ###Signal
    r=unnorm_reconstruct[0]
    b=beforenorm[0]

    diroutSignal = dirout+model_name+'_'+time+'/Signal/'
    if not os.path.exists(diroutSignal):
        print 'Creating output folder Signal...'
        os.makedirs(diroutSignal)
    for rings in range(b.shape[1]):
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            rr = calc_MI2(b[:,rings],r[:,rings])
            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
            #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
            mse_score = mean_squared_error(b[:,rings],r[:,rings])
            corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
            at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
            #at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(b[:,rings]),max(r[:,rings]))
        min_value = min(min(b[:,rings]),min(r[:,rings]))
        bins = min( len(np.histogram(b[:,rings],'fd')[0]), len(np.histogram(r[:,rings],'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(b[:,rings], bins=bins_list, alpha=.5, color='b', label='Input Energy')
        plt.hist(r[:,rings], bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
        plt.axvline(np.max(b[:,rings]), color='b', linestyle='--', linewidth=.8, label='Max Input Energy')
        plt.axvline(np.max(r[:,rings]), color='r', linestyle='--', linewidth=.8, label='Max Reconstructed Energy')
        plt.title(r'Signal - Rings number: '+str(rings+1)+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutSignal+'/'+'hist_et{}_eta{}_ring{}.png'.format(etBinIdx, etaBinIdx, rings+1))
        plt.clf()
        plt.close()

    for segment, inputVector,reconstructVector in [('ETotal',b.sum(axis=1),r.sum(axis=1)),('PS',b[:,0:7,].sum(axis=1),r[:,0:7,].sum(axis=1)),('EM1',b[:,8:71,].sum(axis=1),r[:,8:71,].sum(axis=1)),('EM2',b[:,72:79,].sum(axis=1),r[:,72:79,].sum(axis=1)),('EM3',b[:,80:87,].sum(axis=1),r[:,80:87,].sum(axis=1)),('EM',b[:,0:87,].sum(axis=1),r[:,0:87,].sum(axis=1)),('HAD1',b[:,88:91,].sum(axis=1),r[:,88:91,].sum(axis=1)),('HAD2',b[:,92:95,].sum(axis=1),r[:,92:95,].sum(axis=1)),('HAD3',b[:,96:99,].sum(axis=1),r[:,96:99,].sum(axis=1)),('HAD',b[:,88:99,].sum(axis=1),r[:,88:99,].sum(axis=1))]:
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            rr = calc_MI2(inputVector,reconstructVector)
            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            kl_score = round(calc_kl(inputVector,reconstructVector),4)
            #chi_score,chi_pvalue =calc_chisquare(inputVector,reconstructVector)
            mse_score = mean_squared_error(inputVector,reconstructVector)
            corr_score,corr_pvalue= scipy.stats.pearsonr(inputVector,reconstructVector)
            at = AnchoredText('Input \nMean: '+str(round(inputVector.mean(),2))+"\nStd: "+str(round(inputVector.std(),2))+"\nSkw: "+str(round(skew(inputVector),2))+"\nKur: "+str(round(kurtosis(inputVector),2))+"\n\nReconstructed \nMean: "+str(round(reconstructVector.mean(),2))+"\nStd: "+str(round(reconstructVector.std(),2))+"\nSkw: "+str(round(skew(reconstructVector),2))+"\nKur: "+str(round(kurtosis(reconstructVector),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
            #at = AnchoredText('Input \nMean: '+str(round(inputVector.mean(),2))+"\nStd: "+str(round(inputVector.std(),2))+"\nSkw: "+str(round(skew(inputVector),2))+"\nKur: "+str(round(kurtosis(inputVector),2))+"\n\nReconstructed \nMean: "+str(round(reconstructVector.mean(),2))+"\nStd: "+str(round(reconstructVector.std(),2))+"\nSkw: "+str(round(skew(reconstructVector),2))+"\nKur: "+str(round(kurtosis(reconstructVector),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(inputVector),max(reconstructVector))
        min_value = min(min(inputVector),min(reconstructVector))
        bins = min( len(np.histogram(inputVector,'fd')[0]), len(np.histogram(reconstructVector,'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(inputVector, bins=bins_list, alpha=.5, color='b', label='Input Energy')
        plt.hist(reconstructVector, bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
        plt.axvline(np.max(inputVector), color='b', linestyle='--', linewidth=.8, label='Max Input Energy')
        plt.axvline(np.max(reconstructVector), color='r', linestyle='--', linewidth=.8, label='Max Reconstructed Energy')
        plt.title(r'Signal - '+segment+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutSignal+'/'+'hist_et{}_eta{}_'+segment+'.png'.format(etBinIdx, etaBinIdx))
        plt.clf()
        plt.close()

    ###Background
    r=unnorm_reconstruct[1]
    b=beforenorm[1]

    diroutBackground = dirout+model_name+'_'+time+'/Background/'
    if not os.path.exists(diroutBackground):
        print 'Creating output folder Background...'
        os.makedirs(diroutBackground)
    for rings in range(b.shape[1]):
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            rr = calc_MI2(b[:,rings],r[:,rings])
            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
            #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
            mse_score = mean_squared_error(b[:,rings],r[:,rings])
            corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
            at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
            #at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(b[:,rings]),max(r[:,rings]))
        min_value = min(min(b[:,rings]),min(r[:,rings]))
        bins = min( len(np.histogram(b[:,rings],'fd')[0]), len(np.histogram(r[:,rings],'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(b[:,rings], bins=bins_list, alpha=.5, color='b', label='Input Energy')
        plt.hist(r[:,rings], bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
        plt.axvline(np.max(b[:,rings]), color='b', linestyle='--', linewidth=.8, label='Max Input Energy')
        plt.axvline(np.max(r[:,rings]), color='r', linestyle='--', linewidth=.8, label='Max Reconstructed Energy')
        plt.title(r'Background - Rings number: '+str(rings+1)+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutBackground+'/'+'hist_et{}_eta{}_ring{}.png'.format(etBinIdx, etaBinIdx, rings+1))
        plt.clf()
        plt.close()

    for segment, inputVector,reconstructVector in [('ETotal',b.sum(axis=1),r.sum(axis=1)),('PS',b[:,0:7,].sum(axis=1),r[:,0:7,].sum(axis=1)),('EM1',b[:,8:71,].sum(axis=1),r[:,8:71,].sum(axis=1)),('EM2',b[:,72:79,].sum(axis=1),r[:,72:79,].sum(axis=1)),('EM3',b[:,80:87,].sum(axis=1),r[:,80:87,].sum(axis=1)),('EM',b[:,0:87,].sum(axis=1),r[:,0:87,].sum(axis=1)),('HAD1',b[:,88:91,].sum(axis=1),r[:,88:91,].sum(axis=1)),('HAD2',b[:,92:95,].sum(axis=1),r[:,92:95,].sum(axis=1)),('HAD3',b[:,96:99,].sum(axis=1),r[:,96:99,].sum(axis=1)),('HAD',b[:,88:99,].sum(axis=1),r[:,88:99,].sum(axis=1))]:
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            rr = calc_MI2(inputVector,reconstructVector)
            mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            kl_score = round(calc_kl(inputVector,reconstructVector),4)
            #chi_score,chi_pvalue =calc_chisquare(inputVector,reconstructVector)
            mse_score = mean_squared_error(inputVector,reconstructVector)
            corr_score,corr_pvalue= scipy.stats.pearsonr(inputVector,reconstructVector)
            at = AnchoredText('Input \nMean: '+str(round(inputVector.mean(),2))+"\nStd: "+str(round(inputVector.std(),2))+"\nSkw: "+str(round(skew(inputVector),2))+"\nKur: "+str(round(kurtosis(inputVector),2))+"\n\nReconstructed \nMean: "+str(round(reconstructVector.mean(),2))+"\nStd: "+str(round(reconstructVector.std(),2))+"\nSkw: "+str(round(skew(reconstructVector),2))+"\nKur: "+str(round(kurtosis(reconstructVector),2))+"\n\nNormalized_MI: "+str(mi_score)+"\nMI: "+str(round(rr,4))+"\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score)+"\nReconstruction Error: "+str(round(mse_score,4)),
            #at = AnchoredText('Input \nMean: '+str(round(inputVector.mean(),2))+"\nStd: "+str(round(inputVector.std(),2))+"\nSkw: "+str(round(skew(inputVector),2))+"\nKur: "+str(round(kurtosis(inputVector),2))+"\n\nReconstructed \nMean: "+str(round(reconstructVector.mean(),2))+"\nStd: "+str(round(reconstructVector.std(),2))+"\nSkw: "+str(round(skew(reconstructVector),2))+"\nKur: "+str(round(kurtosis(reconstructVector),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(inputVector),max(reconstructVector))
        min_value = min(min(inputVector),min(reconstructVector))
        bins = min( len(np.histogram(inputVector,'fd')[0]), len(np.histogram(reconstructVector,'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(inputVector, bins=bins_list, alpha=.5, color='b', label='Input Energy')
        plt.hist(reconstructVector, bins=bins_list, alpha=.5, color='r', label='Reconstructed Energy')
        plt.axvline(np.max(inputVector), color='b', linestyle='--', linewidth=.8, label='Max Input Energy')
        plt.axvline(np.max(reconstructVector), color='r', linestyle='--', linewidth=.8, label='Max Reconstructed Energy')
        plt.title(r'Background - '+segment+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutBackground+'/'+'hist_et{}_eta{}_'+segment+'.png'.format(etBinIdx, etaBinIdx))
        plt.clf()
        plt.close()

    return None

def make_representation_hist(norm1Par=None,code=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout='//home/caducovas/run/plots/'):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    #for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
    # if isinstance(reconstruct[layer], (tuple, list,)):
        # unnorm_reconstruct = []
        # for i, cdata in enumerate(reconstruct[layer]):
            # #print i,cdata.shape
            # unnorm_reconstruct.append( cdata * normlist[i])
        # unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
        # beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

      # reconstruct_val_Data = np.concatenate( reconstruct[layer], axis=0 )
      # afternorm_val_Data = np.concatenate( afternorm, axis=0 )

      # if Normed:
        # input_val_Data=afternorm_val_Data
        # reconstruct_val_Data=reconstruct_val_Data
        # input=afternorm
        # reconstructed=reconstruct[layer]
        # normalizacao='yes'
      # else:
        # input_val_Data=beforenorm_val_Data
        # reconstruct_val_Data=unnorm_reconstruct_val_Data
        # input=beforenorm
        # reconstructed=unnorm_reconstruct
        # normalizacao='no'

    ###All Classes
    b=code[layer][0] ###signal
    r=code[layer][1] ###background

    diroutAllclasses = dirout+model_name+'_'+time+'/Representation/'
    if not os.path.exists(diroutAllclasses):
        print 'Creating output folder AllClasses...'
        os.makedirs(diroutAllclasses)
    plt.clf()


    for rings in range(b.shape[1]):
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            #rr = calc_MI2(b[:,rings],r[:,rings])
            #mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            #kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
            #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
            #mse_score = mean_squared_error(b[:,rings],r[:,rings])
            #corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
            at = AnchoredText('Signal \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nBackground \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2)),
            #at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        except:
            print "Error on ring: "+str(rings+1)

        max_value = max(max(b[:,rings]),max(r[:,rings]))
        min_value = min(min(b[:,rings]),min(r[:,rings]))
        bins = min( len(np.histogram(b[:,rings],'fd')[0]), len(np.histogram(r[:,rings],'fd')[0]))
        bins_list = np.linspace(min_value, max_value, num=bins)

        plt.hist(b[:,rings], bins=bins_list, alpha=.5, color='b', label='Signal Energy')
        plt.hist(r[:,rings], bins=bins_list, alpha=.5, color='r', label='Background Energy')
        plt.axvline(np.max(b[:,rings]), color='b', linestyle='--', linewidth=.8, label='Max Signal Energy')
        plt.axvline(np.max(r[:,rings]), color='r', linestyle='--', linewidth=.8, label='Max Background Energy')
        plt.title(r'Rings number: '+str(rings+1)+' distribution - '+model_name+' - '+str(layer),fontsize=25)
        plt.xlabel('Energy [MeV]', fontsize=20)
        plt.tick_params(labelsize = 15)
        plt.legend(loc='best', fontsize='medium')
        #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
        plt.savefig(diroutAllclasses+'/'+'hist_et{}_eta{}_ring{}.png'.format(etBinIdx, etaBinIdx, rings+1))
        plt.clf()
        plt.close()

    return None

def make_ReconstructionErro_hist(norm1Par=None,reconstructErrVector=None,model_name=None,layer=None,time=None, etBinIdx=None,etaBinIdx=None,log_scale=False, dirout='//home/caducovas/run/plots/'):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    beforenorm = norm1Par[0]
    normlist = norm1Par[1]
    afternorm = norm1Par[2]
    png_files=[]

    #for layer in reconstruct.keys():
        #print 'LAYER: '+str(layer)
        #for nsort in reconstruct[layer].keys():
        #print "Sort: "+str(nsort)
    # if isinstance(reconstruct[layer], (tuple, list,)):
        # unnorm_reconstruct = []
        # for i, cdata in enumerate(reconstruct[layer]):
            # #print i,cdata.shape
            # unnorm_reconstruct.append( cdata * normlist[i])
        # unnorm_reconstruct_val_Data = np.concatenate( unnorm_reconstruct, axis=0 )
        # beforenorm_val_Data = np.concatenate( beforenorm, axis=0 )

      # reconstruct_val_Data = np.concatenate( reconstruct[layer], axis=0 )
      # afternorm_val_Data = np.concatenate( afternorm, axis=0 )

      # if Normed:
        # input_val_Data=afternorm_val_Data
        # reconstruct_val_Data=reconstruct_val_Data
        # input=afternorm
        # reconstructed=reconstruct[layer]
        # normalizacao='yes'
      # else:
        # input_val_Data=beforenorm_val_Data
        # reconstruct_val_Data=unnorm_reconstruct_val_Data
        # input=beforenorm
        # reconstructed=unnorm_reconstruct
        # normalizacao='no'

    ###All Classes
    b=reconstructErrVector[0] ###signal
    r=reconstructErrVector[1] ###background

    diroutAllclasses = dirout+model_name+'_'+time+'/ReconstructionError/'
    if not os.path.exists(diroutAllclasses):
        print 'Creating output folder ReconstructionError...'
        os.makedirs(diroutAllclasses)
    plt.clf()


    for rings in range(b.shape[1]):
        fig, ax = plt.subplots(figsize=(16,10))
        try:
            #rr = calc_MI2(b[:,rings],r[:,rings])
            #mi_score = 100*round(np.sqrt(1. - np.exp(-2 * rr)),4)
            #kl_score = round(calc_kl(b[:,rings],r[:,rings]),4)
            #chi_score,chi_pvalue =calc_chisquare(b[:,rings],r[:,rings])
            #mse_score = mean_squared_error(b[:,rings],r[:,rings])
            #corr_score,corr_pvalue= scipy.stats.pearsonr(b[:,rings],r[:,rings])
            at = AnchoredText('Signal \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nBackground \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2)),
            #at = AnchoredText('Input \nMean: '+str(round(b[:,rings].mean(),2))+"\nStd: "+str(round(b[:,rings].std(),2))+"\nSkw: "+str(round(skew(b[:,rings]),2))+"\nKur: "+str(round(kurtosis(b[:,rings]),2))+"\n\nReconstructed \nMean: "+str(round(r[:,rings].mean(),2))+"\nStd: "+str(round(r[:,rings].std(),2))+"\nSkw: "+str(round(skew(r[:,rings]),2))+"\nKur: "+str(round(kurtosis(r[:,rings]),2))+"\n\nCorrelation: "+str(100*round(corr_score,4))+"\nKL Div: "+str(kl_score),
            prop=dict(size=12), frameon=True,
            loc='center right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)

            max_value = max(max(np.log(b[:,rings])),max(np.log(r[:,rings])))
            min_value = min(min(np.log(b[:,rings])),min(np.log(r[:,rings])))
            bins = min( len(np.histogram(np.log(b[:,rings]),'fd')[0]), len(np.histogram(np.log(r[:,rings]),'fd')[0]))
            bins_list = np.linspace(min_value, max_value, num=bins)

            plt.hist(np.log(b[:,rings]), bins=bins_list, alpha=.5, color='b', label='Signal Reconstruction Error')
            plt.hist(np.log(r[:,rings]), bins=bins_list, alpha=.5, color='r', label='Background Reconstruction Error')
            #plt.axvline(np.max(b[:,rings]), color='b', linestyle='--', linewidth=.8, label='Max Signal Reconstruction Error')
            #plt.axvline(np.max(r[:,rings]), color='r', linestyle='--', linewidth=.8, label='Max Background Reconstruction Error')
            plt.title(r'Rings number: '+str(rings+1)+' Reconstruction Error - '+model_name+' - '+str(layer),fontsize=25)
            plt.xlabel('Reconstruction Error (10^)', fontsize=20)
            plt.tick_params(labelsize = 15)
            plt.legend(loc='best', fontsize='medium')
            #plt.savefig(save_path_pdf+'/'+'hist_et{}_eta{}_ring{}.pdf'.format(iet, ieta, iring))
            plt.savefig(diroutAllclasses+'/'+'reconstructionError_et{}_eta{}_ring{}.png'.format(etBinIdx, etaBinIdx, rings+1))
            plt.clf()
            plt.close()
        except:
            print "Error on ring: "+str(rings+1)

    return None
