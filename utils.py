import numpy        as np
import pandas       as pd

import defines      as defs
import dirs

np.set_printoptions(precision=2)

def sp_score(labels, predictions):
    from sklearn.metrics    import confusion_matrix

    vn, fp, fn, vp = confusion_matrix(labels, predictions).ravel()

    S = vp/(vp + fn) # Recall
    E = vn/(vn + fp) # Specificity

    sp_score = np.sqrt(np.sqrt(S*E)*(S + E)/2)

    return sp_score


def get_best_thresh(labels, score, plot=False):
    from sklearn.metrics    import roc_curve, f1_score

    fpr, tpr, thresholds = roc_curve(labels, score[:, 1], pos_label=defs.posCode, drop_intermediate=True)

    spList = []
    for thresh in thresholds:
        # If sample has score >= threshold, it is classified as positive
        predictions = np.where(score[:, 1] >= thresh, defs.posCode, defs.negCode)

        spList.append(sp_score(labels, predictions))    # Select based on SP score
        # spList.append(f1_score(labels, predictions))    # Select based on F1 score (yields worse results, investigate why )

    bestThresh = thresholds[np.argmax(spList)]
    print("\nThreshold: {:.2f}".format(bestThresh))

    return bestThresh


def show_class_splits(labels):
    entriesPos = np.sum(labels == defs.posCode)
    entriesNeg = np.sum(labels == defs.negCode)
    total = len(labels)

    print("Positive class: {:5.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:5.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    return defs.success


def print_metrics(metricsDict):
    for key in metricsDict.keys():
        if isinstance(metricsDict[key][0], float):
            print("{:15}: {:.2f}".format(key, metricsDict[key][0]))
        else:
            print("{:15}: {}".format(key, metricsDict[key][0]))

    return defs.success


def report_performance(labels, predictions, elapsed=0, modelName="", report=True, save=True):
    from sklearn.metrics         import (f1_score, accuracy_score, roc_auc_score,
                                         precision_score, recall_score, confusion_matrix)
    from collections    import OrderedDict

    # metrics = OrderedDict()
    # metricsDf = pd.DataFrame([])
    metrics = dict()

    if modelName == "":
        modelName = "unnamed"

    metrics['Model']		= [modelName]
    metrics['Elapsed']      = [elapsed]
    metrics['accuracy']		= [accuracy_score(labels, predictions, normalize=True)]
    metrics['f1']     		= [f1_score(labels, predictions)]
    metrics['auc']			= [roc_auc_score(labels, predictions)]
    metrics['precision']	= [precision_score(labels, predictions)]
    metrics['recall']		= [recall_score(labels, predictions)]

    # metricsDf.from_dict(metrics)

    conf_matrix = confusion_matrix(labels, predictions, labels=[defs.posCode, defs.negCode])

    if report is True:
        print_metrics(metrics)
        print("\nConfusion Matrix:")
        print(conf_matrix)

    # if save is True:
        # modelName = modelName.replace(" ", "_")
        # # Save metrics to Excel file
        # metricsDf = pd.DataFrame.from_dict(metrics, orient='columns')
        #
        # filePath = dirs.report+"metrics_"+modelName+".xlsx"
        # metricsDf.to_excel(filePath, index=False, float_format="%.2f")

        # # Save confusion matrix to Excel file
        # confDf = pd.DataFrame(conf_matrix, index=['True1', 'True2'], columns=['Predicted1', 'Predicted2'])
        #
        # filePath = dirs.report+"conf_matrix_"+modelName+".xlsx"
        # confDf.to_excel(filePath, index=True)


    return metrics, conf_matrix


def save_excel(metricsTest, metricsCV):
    import os

    try:
        os.makedirs(dirs.report)
    except OSError:
        pass

    modelName = metricsTest["Model"][0].replace(" ", "_")#.split("_")[:-1]
    metricsTest.pop("Model")
    metricsTest.pop("Elapsed")

    df = pd.DataFrame()

    for key in metricsCV.keys():
        # Logic to print mean +- std in one field
        if len(key.split("_")) >= 2:
            name  = "_".join(key.split("_")[:-1])
            compl = key.split("_")[-1]
            if compl == "std":
                df.loc[name, "CV"] = "{:.2f} +- {:.2f}".format(metricsCV[name], metricsCV[key])

    for key in metricsTest.keys():
        # Save floats in a scale [0, 100] for better visualization
        df.loc[key, "Teste"] = metricsTest[key][0]*100

    # Save metrics to Excel file
    filePath = dirs.report+"metrics_"+modelName+".xlsx"
    df.to_excel(filePath, index=True, float_format="%.2f")

    return defs.success


def save_results(cvResults, predictions, modelName):
    import os
    import dirs

    try:
        os.makedirs(dirs.results)
    except OSError:
        pass

    if not(cvResults is None):
        cvResultsPath = dirs.results+modelName.replace(" ", "_")+"_cv_results.csv"
        resultsDf = pd.DataFrame(cvResults)
        resultsDf.to_csv(cvResultsPath)
    else:
        resultsDf = None

    if not(cvResults is None):
        predPath      = dirs.results+modelName.replace(" ", "_")+"_best_pred.npy"
        np.save(predPath,  predictions)

    return resultsDf, predictions


def load_results(modelName):
    import dirs

    cvResultsPath = dirs.results+modelName.replace(" ", "_")+"_cv_results.csv"
    predPath      = dirs.results+modelName.replace(" ", "_")+"_best_pred.npy"

    resultsDf   = pd.DataFrame(pd.read_csv(cvResultsPath))
    predictions = np.load(predPath)

    return resultsDf, predictions


# def get_class_weights(y):
#     '''
#         Returns dict of class count
#     '''
#     classWeights = dict()
#     classWeights[defs.classPos] = np.sum(y == defs.classPos)
#     classWeights[defs.classNeg] = np.sum(y == defs.classNeg)
#     return classWeights
