import numpy                as np
import pandas               as pd

from sklearn.decomposition  import PCA

def preproc(dataDf, verbose=False):
    '''
        Display basic dataset information, clean up and preprocess the data.
            1. Remove zero-values features
            2. Perform Z-score normalization
            3. Display dataset statistics (if verbose is True)
    '''
    if verbose:
        print("\nData shape: ", dataDf.shape)

    # Number of zero values per feature
    # zeroValues = (dataDf == 0).sum(axis=0)
    allZeros   = (dataDf == 0).all(axis=0).sum()

    ## Features containing only zeros will be dropped
    dataDf = dataDf.loc[:, (dataDf != 0).any(axis=0)]
    dataDf = dataDf.reset_index(drop=True)
    print("\n{} features containing only zeros have been dropped from data.".format(allZeros))
    if verbose:
        print("\nNew data shape: ", dataDf.shape)

    ## Z-score normalization
    mean = dataDf.mean(axis=0)
    std  = dataDf.std (axis=0)
    dataDf = (dataDf - mean)/std

    ## Basic Sample Statistics
    if verbose:
        print("\nMax:\n", dataDf.max   (axis=0).max())
        print("\nMin:\n", dataDf.min   (axis=0).min())
        print("\nMean:\n",dataDf.mean  (axis=0).mean())
        print("\nMed:\n", dataDf.median(axis=0).mean())
        print("\nVar:\n", dataDf.std   (axis=0).mean())
        print("\nStd:\n", dataDf.var   (axis=0).mean())

    return dataDf


def dimension_reduction(dataDf, keepComp=0):
    '''
        dataDf:   is an observations by features DataFrame
        keepComp: Number of components to keep
    '''
    if keepComp <= 0:
        keepComp = dataDf.shape[1]

    dataPCA = PCA(n_components=None)

    xCompact = dataPCA.fit_transform(dataDf)

    explVar = dataPCA.explained_variance_ratio_
    # print("\nExplained variance:\n", explVar)
    print("\nN components:", dataPCA.n_components_)
    print("\nPrincipal components to keep: ", keepComp)

    # # Save compact data as the reference DataFrame
    # yCol = dataDf.iloc[:, -1].values.reshape((dataDf.iloc[:, -1].shape[0], 1))
    # compactDf = pd.DataFrame(np.concatenate((xCompact[:, :keepComp], yCol), axis=1))
    compactDf = pd.DataFrame(xCompact[:, :keepComp])
    print("\nCompact data: ", np.shape(compactDf))
    return compactDf

# if __name__ == "__main__":
#     preproc()
