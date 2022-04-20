#!/usr/bin/env python3


from unittest import case
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay


class Knn:
    """Class for classifying samples using the
        knn-method"""

    def __init__(self,filename, k, normalization_method = 'min-max', features = None, pca_components = None):
        self.k = k
        data_frame = pd.read_csv(filename, sep='\t')
        # self.features = features
        steps = []
        if features == None:
            features = list(data_frame.columns.values)[2:65]
        
        data_set = data_frame[features]
        labels = data_frame[['GenreID']]

        self.train_data, self.test_data, self.train_labels, self.test_labels = \
            train_test_split(data_set, labels, shuffle = False, train_size = 0.8)



        if normalization_method == 'min-max':
            steps.append(('min-max',MinMaxScaler())) 
        
        elif normalization_method == 'z-score':
            steps.append(('z-score',StandardScaler()))
        
        if pca_components != None:
            steps.append(('PCA',PCA(n_components=pca_components)))
        
        steps.append(('classifier', KNeighborsClassifier(n_neighbors = k)))

        self.pipeline = Pipeline(steps)

    def error_rate(self):
        return (1 - self.pipeline.score(self.test_data, self.test_labels))

    def conf_matrix(self):
        genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']
        ConfusionMatrixDisplay.from_estimator(self.pipeline, self.test_data, self.test_labels, display_labels = genres)
    
    def plot_PCA(self, pca_components):
        pca = PCA(n_components = pca_components)
        pca.fit(self.train_data, self.train_labels)
        pca_data = pca.transform(self.train_data)
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(pca_data[:,0],pca_data[:,1],pca_data[:,2])
        


    


features = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'tempo']
my_knn = Knn('Classification music/GenreClassData_30s.txt', 5, 'min-max',None, pca_components = 30)
my_knn.pipeline.fit(my_knn.train_data, my_knn.train_labels)
print(my_knn.error_rate())
my_knn.conf_matrix()

plt.show()
