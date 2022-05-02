#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors       import KNeighborsClassifier


from no_pipeline import Dataset
filename = 'Classification music/GenreClassData_30s.txt'
feat = ['spectral_centroid_mean', 'mfcc_1_mean', 'spectral_rolloff_mean', 'spectral_contrast_var']
# genre_data = Dataset(filename, 5, feat, None)
# genre_data.scale()


# ## LOOK AT HISTOGRAM AND ERROR RATE
# genre_data.hist(2,2,True)
# er = genre_data.classify()
# print(er)

## LOOK AT CORRELATION BETWEEN FEATURES
test_features = ['rmse_mean', 'spectral_bandwidth_mean', 'spectral_contrast_var','chroma_stft_12_std']
for f in test_features:
    feat[3] = f
    genre_data = Dataset(filename, 5, feat, None)
    genre_data.scale()
    genre_data.train_data.index = list(range(len(genre_data.train_data.index))) # to avoid a pairplot error
    sns.pairplot(genre_data.train_data)

## Cross validate
kfold = KFold(shuffle=True, random_state=1)
best_feature = ''
best_score = 0
genre_data = Dataset(filename, 5, feat, None)
genre_data.scale()
X = genre_data.train_data.values
Y = genre_data.train_labels.values
for f in test_features:
    feat[3] = f
    knn = KNeighborsClassifier()
    score = np.mean(cross_val_score(knn,X,Y,cv = kfold))
    if score > best_score:
        best_score = score
        best_feature = f

print("Best feature: ", best_feature)
print("score", best_score)
feat[3] = best_feature
genre_data = Dataset(filename, 5, feat, None)
genre_data.scale()
er = genre_data.classify(method='knn',conf_matrix=True)
print("Error rate using",best_feature,"as fourth feature: ", er)

plt.show()