import numpy 				 as np
import matplotlib.pyplot 	 as plt
from scipy 					 import interp

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics 		 import (f1_score,
									accuracy_score,
									roc_auc_score,
									precision_score,
									recall_score,
									classification_report,
									roc_curve,
									auc)


fig, ax = plt.subplots(1, figsize=(15,10))

def cross_val_analysis(n_split=10, classifier=None, x=None, y=None, model_name="", plot=False):
	'''
		Classification and ROC analysis
		Run classifier with cross-validation and plot ROC curves

		Returns:
			metrics_: Dict of metrics
	'''
	kf = StratifiedKFold(n_splits=n_split)
	kf.get_n_splits(x)

	tprs = []
	fpr_ = []
	tpr_ = []
	aucs = []
	accuracy_  = []
	f1_score_  = []
	precision_ = []
	recall_    = []
	roc_auc_   = []

	metrics_ = {}
	mean_fpr = np.linspace(0, 1, 100)

	i = 0
	#start_time = time.time()
	for train, val in kf.split(x,y):
		model   = classifier.fit(x.iloc[train, :], y[train])
		pred_   = model.predict(x.iloc[val, :])
		probas_ = model.predict_proba(x.iloc[val, :])

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

		if plot:
			plt.plot(fpr, tpr, lw=1, alpha=0.3,
					 label='ROC fold %d (AUC = %0.2f)' % (i, 100*roc_auc))

		i += 1
	if plot:
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
			 	 label='Luck', alpha=.8)

	# Store average and std metrics in dict
	# metrics_['model']			= model_name
	metrics_['accuracy']		= np.mean(accuracy_)
	metrics_['accuracy_std']	= np.std(accuracy_)
	metrics_['precision']		= np.mean(precision_)
	metrics_['precision_std']	= np.std(precision_)
	metrics_['recall']			= np.mean(recall_)
	metrics_['recall_std']		= np.std(recall_)
	metrics_['auc']				= np.mean(roc_auc_)
	metrics_['auc_std']			= np.std(roc_auc_)
	metrics_['f1']				= np.mean(f1_score_)
	metrics_['f1_std']			= np.std(f1_score_)
	#metrics_['fpr']			= np.mean(fpr_)
	#metrics_['fpr_std']		= np.std(fpr_)
	#metrics_['tpr']			= np.mean(tpr_)
	#metrics_['tpr_std']		= np.std(tpr_)

	if plot:
		mean_tpr 	 = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc 	 = auc(mean_fpr, mean_tpr)
		std_auc 	 = np.std(aucs)
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

	return metrics_

def cross_val_analysis_nn(n_split=10, classifier=None, x=None, y=None, model_name="",
						  patience=30, train_verbose=2, n_epochs=500):
	'''
		Classification and ROC analysis
		Run classifier with cross-validation and plot ROC curves
	'''
	import keras.callbacks 		 as callbacks

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
