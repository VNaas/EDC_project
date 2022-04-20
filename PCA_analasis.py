#!/usr/bin/env python3


from json.encoder import py_encode_basestring_ascii
from unittest import case
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D
from no_pipeline import Dataset


features = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'tempo']
genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, 'min-max',features)
genre_data.hist()
genre_data.scale()
print("Scaled data. printing head")
print(genre_data.train_data.head)
error_rate = genre_data.classify(True)
print(error_rate)
genre_data.three_feature_plot()

plt.show()

genre_data.do_pca(3)
genre_data.scree_plot()
genre_data.classify(True)
genre_data.plot_train_data_pca()


# genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']

# fig = plt.figure(figsize=(12, 9))
# ax = Axes3D(fig)
# for i in range(10):
#     samples = genre_data.train_data[genre_data.train_data.index == i]
#     x = samples.loc[:,features[0]]
#     y = samples.loc[:,features[1]]
#     z = samples.loc[:,features[2]]
#     ax.scatter(x,y,z, label=genres[i])
# ax.legend()

# plt.bar(x=range(1,len(per_var)+1), height = per_var, tick_label = labels)
# plt.ylabel('Percentage of Explaied Vairance')
# plt.xlabel('Principle Component')
# plt.title('Scree Plot')
# plt.show()

# pca_df = pd.DataFrame(pca_data, columns = labels, index = genre_data.train_labels['GenreID'])

# print(pca_df.head())


# for grp_name, grp_idx in pca_df.groupby('GenreID').groups.items():
#     y = pca_df.iloc[grp_idx,1]
#     print(y)
#     x = pca_df.iloc[grp_idx,0].values[:]
#     z = pca_df.iloc[grp_idx,2].values[:]
#     ax.scatter(x,y,z, label=grp_name)
# ax.legend()



plt.show()