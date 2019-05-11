import numpy                    as np
import pandas                   as pd

import dirs
import defines                  as defs

from utils                      import get_best_thresh
from vis_functions              import plot_roc_curve

# from sklearn.model_selection    import cross_val_score
from cross_val_analysis         import cross_val_analysis

def log_reg(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train Logistic Regression classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        thres: Class discrimination threshold
    '''
    from sklearn.linear_model   import LogisticRegression
    modelName = "Logistic Regression"

    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', n_jobs=1, max_iter=100)

    #metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)
    metricsCV = 0
    model.fit(x_train, y_train)#, weights) # TODO: Add class weights

    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        # predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)

        plot_roc_curve(y_test, probTest, modelName="Logistic Regression")
    else:
        predTest    = model.predict(x_test)
        # predTrain   = model.predict(x_train)

    return predTest, metricsCV, model

def gaussian_naive_bayes(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train Naive Bayes classifier on x_train and predict on x_test
        x_train, x_test: DataFrames of shape (data x features)
    '''
    from sklearn.naive_bayes    import GaussianNB

    model = GaussianNB(priors=None)
    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)

        plot_roc_curve(y_test, probTest, modelName="Naive Bayes")
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def decision_tree(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train a Decision Tree classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.tree           import DecisionTreeClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=15,
                                    min_samples_leaf=5)

    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def random_forest(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train an ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.ensemble       import RandomForestClassifier

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = RandomForestClassifier(n_estimators=50, class_weight='balanced', criterion='entropy',
                                    max_depth=15, min_samples_leaf=5, n_jobs=-1)

    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def ada_boost(x_train, y_train, x_test, y_test, compute_threshold=False):
    '''
        Train an AdaBoost ensemble of Decision Trees on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape num_samples x num_features.
    '''
    from sklearn.ensemble       import AdaBoostClassifier
    from sklearn.tree           import DecisionTreeClassifier
    from sklearn.linear_model   import LogisticRegression

    # estimator = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=7, min_samples_leaf=5)
    estimator = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', n_jobs=1, max_iter=100)
    model = AdaBoostClassifier(base_estimator=estimator, n_estimators=60, learning_rate=1.0)

    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        # probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)

        plot_roc_curve(y_test, probTest, modelName="AdaBoost")
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def linear_discriminant_analysis(x_train, y_train, x_test, y_test, n_components=2, compute_threshold=True):
    '''
        Train Linear Discriminant Analysis (LDA) classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        n_components: Number of components (< n_classes - 1) for dimensionality reduction.
    '''
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = LinearDiscriminantAnalysis(priors=None, n_components=n_components)
    #X_r2 = model.fit(x_train, y_train).transform(X)
    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def quadratic_discriminant_analysis(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train Quadratic Discriminant Analysis (LDA) classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
        n_components: Number of components (< n_classes - 1) for dimensionality reduction.
    '''
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = QuadraticDiscriminantAnalysis()
    #X_r2 = model.fit(x_train, y_train).transform(X)
    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def nearest_neighbours(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train K-Nearest Neighbours classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.neighbors     import KNeighborsClassifier

    # TODO: Experiment with 'weights' parameter
    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = KNeighborsClassifier( n_neighbors=3, algorithm='ball_tree', weights='uniform',
                                p=2, metric='minkowski', n_jobs=-1)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)


    model.fit(x_train, y_train)#, weights) # TODO: Add class weights


    if compute_threshold is True:
        probTest  = model.predict_proba(x_test)
        probTrain = model.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = model.predict(x_test)

    return predTest, metricsCV, model

def perceptron(x_train, y_train, x_test, y_test):
    '''
        Train a Perceptron classifier with linear activation function on x_train
        and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.linear_model   import Perceptron

    classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    model = Perceptron(shuffle=True, n_jobs=-1, class_weight=classWeights, max_iter=1000, tol=1e-3)

    # print("\nParameters initialization:")
    # print(model.coef_)

    # metricsCV = cross_val_analysis(classifier=model, x=x_train, y=y_train, plot=False)
    metricsCV = dict()


    model.fit(x_train, y_train)

    predTest = model.predict(x_test)

    return predTest, metricsCV, model

def svm(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train SVM classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.neighbors     import KNeighborsClassifier
    from sklearn.svm import SVC
    # TODO: Experiment with 'weights' parameter
    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)

    # print("\nParameters initialization:")
    # print(percep.coef_)

    svm.fit(x_train, y_train)#, weights) # TODO: Add class weights


    if compute_threshold is True:
        probTest  = svm.predict_proba(x_test)
        probTrain = svm.predict_proba(x_train)

        bestThresh = get_best_thresh(y_train, probTrain)

        predTest    = np.where(probTest[:, 1] >= bestThresh, defs.posCode, defs.negCode)
        predTrain   = np.where(probTrain[:, 1] >= bestThresh, defs.posCode, defs.negCode)
    else:
        predTest    = svm.predict(x_test)
        predTrain   = svm.predict(x_train)

    return predTest, predTrain, svm

def linear_svm(x_train, y_train, x_test, y_test, compute_threshold=True):
    '''
        Train linear SVM classifier on x_train and predict on x_test.

        x_train, x_test: DataFrames of shape data x features.
    '''
    from sklearn.neighbors     import KNeighborsClassifier
    from sklearn.svm import SVC
    # TODO: Experiment with 'weights' parameter
    # classWeights = {defs.posCode: 0.5, defs.negCode: 0.5}
    svm = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
                    verbose=0)

    # print("\nParameters initialization:")
    # print(percep.coef_)

