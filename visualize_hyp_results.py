import numpy                as np
import pandas               as pd
# import sklearn              as sk
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score

import dirs
import defines              as defs

from utils                  import load_results
from vis_functions          import plot_hyp

modelName = "AdaBoost"
cvResultsDf, predictions = load_results(modelName)
plot_hyp(cvResultsDf, modelName)

modelName = "Decision Tree"
cvResultsDf, predictions = load_results(modelName)
plot_hyp(cvResultsDf, modelName)

modelName = "Nearest Neighbors"
cvResultsDf, predictions = load_results(modelName)
plot_hyp(cvResultsDf, modelName)

modelName = "Random Forest"
cvResultsDf, predictions = load_results(modelName)
plot_hyp(cvResultsDf, modelName)
