import numpy                as np
import pandas               as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score

import dirs
import defines              as defs
from load_dataset           import load_dataset
from preproc                import preproc, dimension_reduction
from utils                  import show_class_splits, print_metrics
from cross_val_analysis     import cross_val_analysis

from analysis_functions             import (gaussian_naive_bayes,
                                            log_reg, ridge_log_reg,
                                            perceptron,
                                            nearest_neighbours,
                                            decision_tree, random_forest, ada_boost,
                                            linear_discriminant_analysis, quadratic_discriminant_analysis)

# np.set_printoptions(precision=4)

print("\n\n---- Loading and Preprocessing ----")

dataDf, labels = load_dataset(dirs.dataset, randomState=defs.standardSample,
                                fracPos=defs.fracPos, fracNeg=defs.fracNeg)#numPos=numPos, numNeg=numNeg)
dataDf = preproc(dataDf, verbose=False)

testSize = round(dataDf.shape[0]*defs.fracTest)
trainDf, testDf, y_train, y_test = train_test_split(dataDf, labels, test_size=testSize,
                                                    random_state=defs.standardSample)

print("\nTrain data loaded with following class distributions:")
show_class_splits(y_train)
print("\nTest data loaded with following class distributions:")
show_class_splits(y_test)

# 'Principal Components Analysis'
# '   useful to reduce dataset dimensionality'
# compactDf = dimension_reduction(dataDf, keepComp=60)

print("\n\n---- Classification ----\n")

'Logistic Regression with L2 Regularization'
# TODO: Testar LogisticRegressionCV, que encontra o C otimo
logPenalty = 1/100

print("\nLogistic Regression with L2 Regularization")
start = time.perf_counter()
_, logReg = ridge_log_reg(trainDf, y_train, testDf, y_test, reg=logPenalty, class_weight='balanced')

metricsLogReg = cross_val_analysis(n_split=10, classifier=logReg, x=trainDf, y=y_train,
                                    model_name="Logistic Regression", plot=False)

elapsed = time.perf_counter() - start
print("Regularization paramenter (smaller is stronger): \n", logPenalty)
print("Elapsed: {:.2f}s".format(elapsed))
print_metrics(metricsLogReg)
