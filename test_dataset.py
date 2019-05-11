import numpy                as np
# import seaborn              as sns
import matplotlib.pyplot    as plt
# import sklearn              as skl
from sklearn.decomposition  import PCA
from mpl_toolkits.mplot3d   import Axes3D

import dirs

from keras.datasets         import mnist

# Dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

print("x shape: ", np.shape(x_train))
print("y shape: ", np.shape(y_train))
print("\nx max:", x_train.max())
print("\nx min:", x_train.min())

# Flatten input
x = np.reshape(x_train, (np.shape(x_train)[0], -1))

# Z-score normalization - for images: compute statistics over the entire dataset instead of featurewise
x_mean = np.mean(x) # Mean over every flattened input
x_std  = np.std(x)  # Std over every flattened input

x = np.divide(x - x_mean, x_std)

## PCA
Data_pca = PCA(n_components=None)
x_new = Data_pca.fit_transform(x, y_train)

expl_var = Data_pca.explained_variance_ratio_
print("\nExplained variance:\n", expl_var)
print("\nN components:\n", Data_pca.n_components_)

print("\nPrincipal components: ", np.shape(Data_pca.components_))

## Plot data
# plt.plot(x_new[:, 0], x_new[:, 1], 'b.')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_new[:, 0], x_new[:, 1], x_new[:, 2], 'b')
plt.show()
