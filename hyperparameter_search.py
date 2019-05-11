import numpy                as np
import pandas               as pd
import time
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV

import dirs
import defines              as defs
from load_dataset           import load_dataset
from preproc                import preproc, dimension_reduction
from utils                  import show_class_splits, report_performance, save_results
# from cross_val_analysis     import cross_val_analysis

# from analysis_functions             import (gaussian_naive_bayes,
#                                             log_reg, ridge_log_reg,
#                                             perceptron,
#                                             nearest_neighbours,
#                                             decision_tree, random_forest, ada_boost,
#                                             linear_discriminant_analysis, quadratic_discriminant_analysis)

def hyp_logistic_regression(x_train, y_train, x_test, num_iter=10):
    '''
        Perform Hyperparameter search for Logistic Regression model on train and
        validation sets, evaluate best estimator on test set.

        Obs: It is usually not necessary to apply regularization on linear models,
        rendering this search moot.

        Returns test set predictions
    '''
    from scipy.stats             import expon
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.linear_model    import LogisticRegression


    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=100,
                                class_weight='balanced', n_jobs=1)

    params = {'C': expon()}
    hypModel = RandomizedSearchCV(model, params, n_iter=num_iter, scoring='f1', cv=10,
                                n_jobs=-1, error_score='raise', verbose=0)

    hypModel.fit(trainDf, y_train)

    print("\nBest parameters:", hypModel.best_params_)

    predictions = hypModel.predict(testDf)

    return predictions

def hyp_knn(x_train, y_train, x_test):
    '''
        Perform Hyperparameter search for Nearest Neighbours classifier on train and
        validation sets, evaluate best estimator on test set.

        Returns test set predictions
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors       import KNeighborsClassifier

    model = KNeighborsClassifier( n_neighbors=5, algorithm='ball_tree', weights='uniform',
                                p=2, metric='minkowski', n_jobs=-1)

    params = {'n_neighbors': list(range(3, 30, 1))}
    hypModel = GridSearchCV(model, params, scoring='f1', cv=10, n_jobs=-1, error_score='raise',
                            verbose=2, return_train_score=True)

    hypModel.fit(trainDf, y_train)

    print("\nBest parameters:", hypModel.best_params_)

    predictions = hypModel.predict(testDf)

    return predictions, hypModel.cv_results_

def hyp_decision_tree(x_train, y_train, x_test):
    '''
        Perform Hyperparameter search for Decision Tree on train and
        validation sets, evaluate best estimator on test set.

        Returns test set predictions
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree            import DecisionTreeClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = DecisionTreeClassifier(class_weight="balanced", criterion='entropy', max_depth=15, min_samples_leaf=5)

    params = {'max_depth': list(range(2, 30, 1))}
    hypModel = GridSearchCV(model, params, scoring='f1', cv=10, n_jobs=4, error_score='raise',
                            verbose=2, return_train_score=True)

    hypModel.fit(trainDf, y_train)

    print("\nBest parameters:", hypModel.best_params_)

    predictions = hypModel.predict(testDf)

    return predictions, hypModel.cv_results_

def hyp_random_forest(x_train, y_train, x_test):
    '''
        Perform Hyperparameter search for Decision Tree on train and
        validation sets, evaluate best estimator on test set.

        Returns test set predictions
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble        import RandomForestClassifier


    model = RandomForestClassifier(n_estimators=50, class_weight='balanced', criterion='entropy',
                                    max_depth=15, min_samples_leaf=5, n_jobs=-1)

    params = {'n_estimators': list(range(2, 80, 2))}
    hypModel = GridSearchCV(model, params, scoring='f1', cv=10, n_jobs=4, error_score='raise',
                            verbose=2, return_train_score=True)

    hypModel.fit(trainDf, y_train)

    print("\nBest parameters:", hypModel.best_params_)

    predictions = hypModel.predict(testDf)

    return predictions, hypModel.cv_results_

def hyp_ada_boost(x_train, y_train, x_test):
    '''
        Perform Hyperparameter search for AdaBoost ensemble on train and
        validation sets, evaluate best estimator on test set.

        Returns test set predictions
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble        import AdaBoostClassifier

    # tree = DecisionTreeClassifier(class_weight="balanced", criterion='entropy',
    #                                         max_depth=15, min_samples_leaf=5)

    model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

    params = {'n_estimators': list(range(2, 80, 2))}
    hypModel = GridSearchCV(model, params, scoring='f1', cv=10, n_jobs=4, error_score='raise',
                            verbose=2, return_train_score=True)

    hypModel.fit(trainDf, y_train)

    print("\nBest parameters:", hypModel.best_params_)

    predictions = hypModel.predict(testDf)

    return predictions, hypModel.cv_results_


if __name__ == "__main__":
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

    print("\n\n---- Hyperparameter search ----\n")


    #
    # 'Logistic Regression with L2 Regularization'
    # # TODO: Testar LogisticRegressionCV, que encontra o C otimo
    # modelName = "Logistic Regression with L2 Regularization"
    # numIter = 10
    #
    # print("\n", modelName)
    #
    # start   = time.perf_counter()
    # bestPred = hyp_logistic_regression(trainDf, y_train, testDf, num_iter=10)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest = report_performance(y_test, bestPred, elapsed=elapsed, model_name=modelName)
    # print("")

    # 'Nearest Neighbors'
    # modelName = "Nearest Neighbors"
    # print("\n", modelName)
    #
    # start   = time.perf_counter()
    # bestPred, cvResults = hyp_knn(trainDf, y_train, testDf)
    # elapsed = time.perf_counter() - start
    #
    # metricsPercepTest = report_performance(y_test, bestPred, elapsed=elapsed, model_name=modelName)
    #
    # save_results(cvResults, bestPred, modelName)
    # print("")
    #
    # 'Decision Tree'
    # modelName = "Decision Tree"
    # print("\n", modelName)
    #
    # start   = time.perf_counter()
    # bestPred, cvResults = hyp_decision_tree(trainDf, y_train, testDf)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest = report_performance(y_test, bestPred, elapsed=elapsed, model_name=modelName)
    #
    # save_results(cvResults, bestPred, modelName)
    # print("")

    # 'Random Forest'
    # modelName = "Random Forest"
    # print("\n", modelName)
    #
    # start   = time.perf_counter()
    # bestPred, cvResults = hyp_random_forest(trainDf, y_train, testDf)
    # elapsed = time.perf_counter() - start
    #
    # metricsTest = report_performance(y_test, bestPred, elapsed=elapsed, model_name=modelName)
    #
    # resultsDf, _ = save_results(cvResults, bestPred, modelName)
    # plot_hyp(resultsDf, modelName)
    # print("")

    'AdaBoost'
    modelName = "AdaBoost"
    print("\n", modelName)

    start   = time.perf_counter()
    bestPred, cvResults = hyp_ada_boost(trainDf, y_train, testDf)
    elapsed = time.perf_counter() - start

    metricsTest = report_performance(y_test, bestPred, elapsed=elapsed, model_name=modelName)

    resultsDf, _ = save_results(cvResults, bestPred, modelName)
    plot_hyp(resultsDf, modelName)
    print("")
