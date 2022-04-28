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

gen = ['pop', 'metal', 'disco','classical']

feat = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'tempo']
genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, gen)
genre_data.hist()
genre_data.scale()
# print("Scaled data. printing head")
# print(genre_data.train_data.head)
# error_rate = genre_data.classify(True)
# print(error_rate)
# genre_data.three_feature_plot()

# plt.show()

genre_data.do_pca(3)
genre_data.scree_plot()
genre_data.classify(True)
genre_data.plot_train_data_pca()


genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']


plt.show()