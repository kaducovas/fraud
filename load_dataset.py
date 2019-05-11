import numpy                as np
import pandas               as pd

import defines

def load_dataset(path, fracPos=1.0, fracNeg=1.0, numPos=0, numNeg=0, randomState=None):
    # Load raw data
    data = np.load(path)

    # Save each class
    classNeg = pd.DataFrame(data[data.keys()[0]])   # Hadrons
    classPos = pd.DataFrame(data[data.keys()[1]])   # Eletrons

    print("\nOriginal data:")
    print("Positive examples: {:>9d}".format(classPos.shape[0]))
    print("Negative examples: {:>9d}".format(classNeg.shape[0]))

    # Sample a fraction of the data
    # doing this before anything else avoids memory issues
    # features = classPos.shape[1]
    totPos   = classPos.shape[0]
    totNeg   = classNeg.shape[0]

    if (numPos == 0) and (numNeg == 0):
        classPos = classPos.sample(frac=fracPos, axis=0, random_state=randomState)
        classNeg = classNeg.sample(frac=fracNeg, axis=0, random_state=randomState)
    else:
        classPos = classPos.sample(n=numPos, axis=0)
        classNeg = classNeg.sample(n=numNeg, axis=0)

    # if testSize > 0:
    #     # Create test set with testSize elements per class
    #     testPos = classPos[:testSize]
    #     testNeg = classNeg[:testSize]
    #
    #     # Test set and labels
    #     testDf     = pd.concat((testPos, testNeg), axis=0)
    #     testLabels = np.concatenate((np.ones(testSize),     # Positive examples
    #                                 -np.ones(testSize)))    # Negative examples
    #
    #     # Remove test set from data
    #     classPos = classPos[testSize:]
    #     classNeg = classNeg[testSize:]
    # else:
    #     # Create empty variables for compatibility
    #     testDf     = pd.DataFrame([])
    #     testLabels = np.array([])

    entriesPos = classPos.shape[0]
    entriesNeg = classNeg.shape[0]
    total = entriesPos + entriesNeg

    # Create Label vectors
    yPos  = np.ones(entriesPos, dtype=int)*defines.posCode  # [+1, -1] representation has zero mean, which
    yNeg  = np.ones(entriesNeg, dtype=int)*defines.negCode  # can lead to faster gradient convergence

    # Concatenate class labels
    labels = np.concatenate((yPos, yNeg))

    print("\n\nData loaded with following class distribution: ")
    print("Positive class: {:5.2f} %, {} entries ".format(entriesPos/total*100, entriesPos))
    print("Negative class: {:5.2f} %, {} entries ".format(entriesNeg/total*100, entriesNeg))
    print("Total:          {} entries".format(total))

    # Save dataset in a DataFrame
    dataDf = pd.concat((classPos, classNeg), axis=0)

    return dataDf, labels
