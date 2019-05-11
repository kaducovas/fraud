import numpy                as np
import pandas               as pd
import seaborn              as sns
import os

import matplotlib.pyplot    as plt
import matplotlib           as mlp
from mpl_toolkits.mplot3d   import Axes3D

import defines              as defs
import dirs

mlp.rcParams['font.size'] = 24

def plot_roc_curve(labels, score, modelName="", show=False, save=True):
    from sklearn.metrics    import roc_curve, f1_score
    from utils              import sp_score

    fig = plt.figure(figsize=(24, 18))

    fpr, tpr, thresholds = roc_curve(labels, score[:, 1], pos_label=defs.posCode, drop_intermediate=True)

    spList = []
    for thresh in thresholds:
        # If sample has score >= threshold, it is classified as positive
        predictions = np.where(score[:, 1] >= thresh, defs.posCode, defs.negCode)

        spList.append(sp_score(labels, predictions))    # Select based on SP score
        # spList.append(f1_score(labels, predictions))    # Select based on F1 score (yields worse results, investigate why )

    bestThresh = thresholds[np.argmax(spList)]

    plt.plot(fpr, tpr, label='ROC', linewidth='4')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Classificador aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Probabilidade de Falso Positivo',fontsize= 'large')
    plt.ylabel('Probabilidade de Deteccao',fontsize= 'large')
    plt.title('Curva ROC '+modelName, fontsize= 'xx-large')
    plt.legend(loc="lower right")

    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
                        wspace=None, hspace=None)


    if show is True:
        plt.show()

    if save is True:
        try:
            os.makedirs(dirs.figures)
        except OSError:
            pass

        fig.savefig(dirs.figures+"ROC_Curve_"+modelName.replace(" ", "_")+".pdf",
                    orientation='portrait', bbox_inches='tight')
        fig.savefig(dirs.figures+"ROC_Curve_"+modelName.replace(" ", "_")+".png",
                    orientation='portrait', bbox_inches='tight')


    return fig

def plot_conf_matrix(labels, predictions, modelName="", show=False, save=True, normalize=True):
    import os
    from sklearn.metrics    import confusion_matrix

    if modelName == "":
        modelName = "unnamed"

    try:
        os.makedirs(dirs.report)
    except OSError:
        pass

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
    confDf = pd.DataFrame(confusion_matrix(labels, predictions, labels=[defs.posCode, defs.negCode]),
                         index=['True H', 'True E'], columns=['Pred. H', 'Pred. E'])

    ## Plot confusion matrix
    # Auto adjust colorbar limits
    vmin = None
    vmax = None
    fmt = ''

    axs[0].set_title("Matriz de Confusao")
    ax = sns.heatmap(confDf, ax=axs[0], annot=True, cmap='Blues', vmin=vmin, vmax=vmax, square=True, cbar=False,
                    fmt=fmt, annot_kws={'verticalalignment': 'center', 'horizontalalignment': 'center'})


    # Set up border labels

    ## Plot normalized confusion matrix

    # Normalize each row of conf matrix
    confDf.iloc[0, :] = confDf.iloc[0, :]/confDf.iloc[0, :].sum()
    confDf.iloc[1, :] = confDf.iloc[1, :]/confDf.iloc[1, :].sum()

    # Set colorbar limits to range [0, 1]
    vmin = 0.0
    vmax = 1.0
    fmt = '.2f'

    axs[1].set_title("Normalizada")
    ax = sns.heatmap(confDf, ax=axs[1], annot=True, cmap='Blues', vmin=vmin, vmax=vmax, square=True, cbar=False,
                    fmt=fmt, annot_kws={'verticalalignment': 'center', 'horizontalalignment': 'center'})

    axs[1].get_yaxis().set_visible(False)


    plt.subplots_adjust(left=0.02, bottom=0.08, right=0.90, top=0.95,
                        wspace=None, hspace=None)
    plt.suptitle("{}".format(modelName))

    if show is True:
        plt.show()

    if save is True:
        fig.savefig(dirs.report+"Conf_matrix_"+modelName.replace(" ", "_")+".png", orientation='portrait', bbox_inches='tight')

    return ax

def plot_hyp(resultsDf, modelName, save=True, show=False):
    '''
        Plot hyperparameter search plots: F1 score as function of parameter variation.
        Restricted to unidimensional search spaces (search over only one parameter).
    '''
    import os
    import ast

    try:
        os.makedirs(dirs.report)
    except OSError:
        pass

    # Convert the dict expressed in string form to dict object
    params = list(ast.literal_eval(resultsDf.loc[0, 'params']).keys())

    if len(params) == 1:
        paramName = params[0]

        x = resultsDf["param_"+paramName]

        train = resultsDf["mean_train_score"]
        test  = resultsDf["mean_test_score"]

        trainStd = resultsDf["std_train_score"]
        testStd  = resultsDf["std_test_score"]

        fig = plt.figure(figsize=(28, 15))

        # Plot train score
        plt.errorbar(x, train, yerr=trainStd,  fmt='.-', color='xkcd:tomato', markersize=8,
                    markerfacecolor='xkcd:fire engine red', markeredgecolor='xkcd:tomato',
                    ecolor="xkcd:grey green", label="Train Score")

        # Plot test score
        plt.errorbar(x, test, yerr=testStd,  fmt='.-', color='xkcd:dark blue', markersize=8,
                    markerfacecolor='xkcd:night blue', markeredgecolor='xkcd:dark blue',
                    ecolor="xkcd:grey green", label="Val Score")
        plt.legend(fontsize='small', loc='best')

        plt.ylim(ymin=0.0, ymax=1.0)             # Should be adjusted for each search space
        plt.xlim(xmin=x.min()-1, xmax=x.max()+1) # as needed

        plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
                            wspace=None, hspace=None)

        plt.title("{} parameter search: {}".format(modelName, paramName))
        plt.xlabel(paramName)
        plt.ylabel("F1 Score", fontsize=28)

        if show is True:
            plt.show()

        if save is True:
            # Save plots
            fig.savefig(dirs.report+"hyp_"+modelName.replace(" ", "_")+".pdf", orientation='portrait', bbox_inches='tight')
            fig.savefig(dirs.report+"hyp_"+modelName.replace(" ", "_")+".png", orientation='portrait', bbox_inches='tight')
    return defs.success

def projection_plot(inputDf, labels, save=True, show=False):
    '''
        inputDf is an observations by features DataFrame
        labels is an observations by 1 DataFrame of [+1, -1] labels
    '''
    try:
        os.makedirs(dirs.figures)
    except OSError:
        pass

    features = inputDf.shape[1]
    # features = 10

    posData = inputDf[labels == defs.posCode]  # Last DataFrame columns are labels:
    negData = inputDf[labels == defs.negCode]  # used to filter data by class

    fig, axs = plt.subplots(nrows=features, ncols=features, figsize=(12,10))#, squeeze=False, tight_layout=True)

    # print("\nMax: ", features)
    for row in range(features):
        for col in range(features):
            if row <= col:
                if row == col:
                    # Plot histogram of feature i
                    axs[row,col].hist(inputDf.iloc[:, col].values, bins='auto', color='xkcd:dull blue')  # axis.hist() method doesn't work with DataFrame
                else:
                    # Plot projection X_i by X_j
                    axs[row,col].plot(posData.iloc[:, row], posData.iloc[:, col], '.', alpha=0.3, markerfacecolor='xkcd:fire engine red', markeredgecolor='xkcd:tomato')
                    axs[row,col].plot(negData.iloc[:, row], negData.iloc[:, col], '.', alpha=0.3, markerfacecolor='xkcd:night blue', markeredgecolor='xkcd:dark blue')

            # Hide axis labels
            axs[row,col].get_xaxis().set_visible(False)
            axs[row,col].get_yaxis().set_visible(False)
            axs[row,col].set_yticklabels([])
            axs[row,col].set_xticklabels([])

            # Set up border labels
            if col == 0:
                axs[row,col].set_ylabel("X{}".format(row))
                axs[row,col].get_yaxis().set_visible(True)
            if row == (features-1):
                axs[row,col].set_xlabel("X{}".format(col))
                axs[row,col].get_xaxis().set_visible(True)

    # plt.title("Projection Plot of {}".format(features))
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
                        wspace=None, hspace=None)

    fig.set_size_inches(20, 20)

    if show is True:
        plt.show()

    if save is True:
        # Save plots
        fig.savefig(dirs.figures+"Projection_Plot_{}_features".format(features)+".pdf",
                    orientation='portrait', bbox_inches='tight')
        fig.savefig(dirs.figures+"Projection_Plot_{}_features".format(features)+".png",
                    orientation='portrait', bbox_inches='tight')

    return axs, fig


def plot_boxplot(inputDf, labels, save=True, show=False):
    '''
        Plot Boxplot
    '''
    try:
        os.makedirs(dirs.figures)
    except OSError:
        pass

    dataDf = pd.concat([inputDf, pd.DataFrame(labels, columns=['labels'])])

    # Boxplot
    ax = sns.boxplot(data=inputDf)

    ax.set_title("Boxplot of {} features".format(inputDf.shape[1]))

    fig = plt.gcf()

    # ax.tick_params(axis='x', labelsize='x-small')
    ax.set_ylim(top=100)
    ax.get_xaxis().set_visible(False)

    fig.set_size_inches(28, 15)
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    wspace=None, hspace=None)

    if show is True:
        plt.show()

    if save is True:
        # Save plots
        fig.savefig(dirs.figures+"Boxplot"+".pdf", orientation='portrait', bbox_inches='tight')
        fig.savefig(dirs.figures+"Boxplot"+".png", orientation='portrait', bbox_inches='tight')

    return ax


def eigen_plot(inputDf, labels, save=True, show=False):
    '''
        Plot cumulative components cumulative and individual contribution to signal energy.
    '''
    try:
        os.makedirs(dirs.figures)
    except OSError:
        pass

    # Get Principal Components ratio
    eignVals =  np.linalg.svd(inputDf, compute_uv=False)
    eignVals = eignVals/eignVals.sum()

    features = len(eignVals)
    index    = list(range(features))

    # Compute cumulative contributions
    sumVals = np.zeros(features)
    for i in range(features):
        sumVals[i] = np.sum(eignVals[:i])

    sumVals = sumVals[1:]
    sumVals = np.append(sumVals, 1.0)

    # Plot barplot of Principal Components ratios
    ax = sns.barplot(x=index, y=eignVals, palette="Blues_d")

    # Plot lineplot of cumulative contributions
    ax.plot(index, sumVals, 'b-')

    ax.set_title("{} Principal values".format(features))
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.get_xaxis().set_visible(False)

    fig = plt.gcf()
    fig.set_size_inches(25, 15)
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    wspace=None, hspace=None)

    if show is True:
        plt.show()

    if save is True:
        # Save plots
        plt.savefig(dirs.figures+"Principal_components"+".pdf", orientation='portrait', bbox_inches='tight')
        plt.savefig(dirs.figures+"Principal_components"+".png", orientation='portrait', bbox_inches='tight')
    return ax


def plot_3d(compactDf, labels, save=True, show=False):
    '''
        Plot projection of first 3 principal components in a 3D plot.
    '''
    try:
        os.makedirs(dirs.figures)
    except OSError:
        pass

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    posData = compactDf[labels == defs.posCode]
    negData = compactDf[labels == defs.negCode]

    ax.scatter(posData.iloc[:, 0], posData.iloc[:, 1], posData.iloc[:, 2], c='xkcd:fire engine red', alpha=0.3)
    ax.scatter(negData.iloc[:, 0], negData.iloc[:, 1], negData.iloc[:, 2], c='xkcd:twilight blue', alpha=0.3)

    ax.set_title("3D Scatter Plot")

    fig = plt.gcf()
    fig.set_size_inches(28, 28)
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    wspace=None, hspace=None)

    if show is True:
        plt.show()

    if save is True:
        # Save plots
        fig.savefig(dirs.figures+"Plot_3D"+".pdf", orientation='portrait', bbox_inches='tight')
        fig.savefig(dirs.figures+"Plot_3D"+".png", orientation='portrait', bbox_inches='tight')

    return fig, ax


def corr_matrix_plot(inputDf, save=True, show=False):
    '''
        inputDf is an observations by features DataFrame
    '''
    try:
        os.makedirs(dirs.figures)
    except OSError:
        pass


    corr = inputDf.corr()
    print("Correlation 1x1: ", corr[0,0])

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(13, 11))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap='viridis', vmax=.3, center=0,
                square=True, linewidths=0.00, cbar_kws={"shrink": .5})

    ax.set_title("Correlation plot of {} features".format(len(corr)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig = plt.gcf()
    fig.set_size_inches(25, 25)
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    wspace=None, hspace=None)


    if show is True:
        plt.show()

    if save is True:
        # Save plots
        fig.savefig(dirs.figures+"Correlation_Matrix_{}_features".format(len(corr))+".pdf",
                    orientation='portrait', bbox_inches='tight')
        fig.savefig(dirs.figures+"Correlation_Matrix_{}_features".format(len(corr))+".png",
                    orientation='portrait', bbox_inches='tight')

    return fig, ax

from operator import itemgetter

def format_as_table(data,
                    keys,
                    header=None,
                    sort_by_key=None,
                    sort_order_reverse=False):
    """
    Takes a list of dictionaries, formats the data, and returns
    the formatted data as a text table.

    Required Parameters:
        data - Data to process (list of dictionaries). (Type: List)
        keys - List of keys in the dictionary. (Type: List)

    Optional Parameters:
        header - The table header. (Type: List)
        sort_by_key - The key to sort by. (Type: String)
        sort_order_reverse - Default sort order is ascending, if
            True sort order will change to descending. (Type: Boolean)
    """
    # Sort the data if a sort key is specified (default sort order
    # is ascending)
    if sort_by_key:
        data = sorted(data,
                      key=itemgetter(sort_by_key),
                      reverse=sort_order_reverse)

    # If header is not empty, add header to data
    if header:
        # Get the length of each header and create a divider based
        # on that length
        header_divider = []
        for name in header:
            header_divider.append('-' * len(name))

        # Create a list of dictionary from the keys and the header and
        # insert it at the beginning of the list. Do the same for the
        # divider and insert below the header.
        header_divider = dict(zip(keys, header_divider))
        data.insert(0, header_divider)
        header = dict(zip(keys, header))
        data.insert(0, header)

    column_widths = []
    for key in keys:
        column_widths.append(max(len(str(column[key])) for column in data))

    # Create a tuple pair of key and the associated column width for it
    key_width_pair = zip(keys, column_widths)

    format = ('%-*s ' * len(keys)).strip() + '\n'
    formatted_data = ''
    for element in data:
        data_to_format = []
        # Create a tuple that will be used for the formatting in
        # width, value format
        for pair in key_width_pair:
            data_to_format.append(pair[1])
            data_to_format.append(element[pair[0]])
        formatted_data += format % tuple(data_to_format)
    return formatted_data
