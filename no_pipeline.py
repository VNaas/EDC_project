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

class Dataset:
    """Class for classifying samples using the
        knn-method"""

    def __init__(self,filename, k, normalization_method = 'min-max', features = None, genres = None):
        # TODO Should be able to initialize with a set of genres,
        # And  make the data set only out of these. See how it's
        # done in the first line in the for loop of three_feature_plot()
        # It might help
        self.k = k
        data_frame = pd.read_csv(filename, sep='\t')
        print(data_frame.head())
        # self.features = features
        if features == None:
            features = list(data_frame.columns.values)[2:65]

        self.features = features
        self.data_set = data_frame[features]
        self.labels = data_frame[['GenreID']]
        self.genredict = {  'pop':0, 'metal':1,'disco':2,\
            'blues':3, 'reggae':4, 'classical':5,\
            'rock':6, 'hip_hop':7, 'country':8,'jazz':9}

        self.train_data, self.test_data, self.train_labels, self.test_labels = \
            train_test_split(self.data_set, self.labels, shuffle = False, train_size = 0.8)
        
        self.pca = PCA()
        # self.train_data_df = pd.DataFrame(self.train_data,\
        #      columns = features, index = self.train_labels['GenreID'])
        self.train_data.index = self.train_labels['GenreID']
        print("Initiated dataset ocject. Training data head: \n ",\
            self.train_data.head)
    

        print("hei")

    def hist(self):
        self.train_data.hist()

    def scale(self, normalization_method = 'min-max'):
        """
        Trains a scaling object to the training set and uses it to
        transform the training and test sets.
        The transform methods of the scaler classes return arrays.
        Therefore we must make the scaled arrays first and then
        update the values of the data frames using df.loc
        """
        if normalization_method == 'min-max':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        scaled_train_data = scaler.fit_transform(self.train_data)
        self.train_data.loc[:,:] = scaled_train_data

        scaled_test_data = scaler.transform(self.test_data)
        self.test_data.loc[:,:] = scaled_test_data
        print("Scaled data: \n", self.train_data.head)

    def three_feature_plot(self):
        """
        Makes a 3-Dimensional scatter plot of a 3D feature space.
        Note that the function does not call plt.show()
        """
        if len(self.features) == 3:
            genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']
            fig = plt.figure(figsize=(12, 9))
            ax = Axes3D(fig)
            for i in range(10):
                samples = self.train_data[self.train_data.index == i]
                x = samples.loc[:,self.features[0]]
                y = samples.loc[:,self.features[1]]
                z = samples.loc[:,self.features[2]]
                ax.scatter(x,y,z, label=genres[i])
            ax.axes.set_xlabel(self.features[0])
            ax.axes.set_ylabel(self.features[1])
            ax.axes.set_zlabel(self.features[2])
            ax.legend()
            ax.set_title("Scatter plot of three features")

    def classify(self, conf_matrix = False):
        """
        Returns: Errot rate
        Trains a KNN-classifier using the training data and 
        tests it on the test set. If conf_matrix is true, it will also
        create a confusion matrix.
        """
        classifier = KNeighborsClassifier(n_neighbors = self.k)
        classifier.fit(self.train_data,self.train_labels)
        genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']        
        if conf_matrix: 
            ConfusionMatrixDisplay.from_estimator(classifier, self.test_data, self.test_labels, display_labels = genres)
        error_rate = (1 - classifier.score(self.test_data, self.test_labels))
        return error_rate

    def do_pca(self, n = 3):
        """
        Performs principle component analysis on the data set, possibly
        reducing the feature space down to n components. Note that n
        must be smaller than or equal to the number of features in the 
        data set.
        """
        # TODO This should create new data frame objects since
        # dimentions might change
        pca = PCA(n_components = n)
        pca.fit(self.train_data)
        pca_train_data =  pca.fit_transform(self.train_data)
        pca_test_data = pca.transform(self.test_data)
        self.train_data.loc[:,:] = pca_train_data
        self.test_data.loc[:,:] = pca_test_data
        self.pca = pca
    
    def plot_train_data_pca(self):
        """
        Creates a 3D scatter plot from the three principle components.
        Note that the function does not call plt.show()
        """
        genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']

        if self.test_data.shape[1] == 3:
            self.train_data.columns = ['PC1', 'PC2', 'PC3']
            fig = plt.figure(figsize=(12, 9))
            ax = Axes3D(fig)
            for i in range(10):     #Might change to length of a list of a self.genres array for using fewer genres
                samples = self.train_data[self.train_data.index == i]
                ax.scatter(samples.loc[:,'PC1'],samples.loc[:,'PC2'],samples.loc[:,'PC3'], label=genres[i])
            ax.legend()
            ax.axes.set_xlabel('PC1')
            ax.axes.set_ylabel('PC2')
            ax.axes.set_zlabel('PC3')
            ax.set_title('PCA analysis')

        else: print('Data is not 3-dimensional')
    
    def scree_plot(self):
        """
        A Scree plot Shows how much of the variance (information)
        in a data set is represented by each principle component.
        This is a good indicator of which principle components 
        should be used to represent your data.
        """
        fig = plt.figure(figsize=(12, 9))
        per_var = np.round(self.pca.explained_variance_ratio_ * 100, decimals = 1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        plt.bar(x=range(1,len(per_var)+1), height = per_var, tick_label = labels)
        plt.ylabel('Percentage of Explaied Vairance')
        plt.xlabel('Principle Component')
        plt.title('Scree Plot')


genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, 'min-max',None)

#er = 1
#for i in range(1,63):   
#    genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, 'min-max',None)
#    genre_data.scale('min-max')
# 
#    genre_data.pca(i)
#    er_i = genre_data.classify(False)
#    if er_i <= er:
#        er = er_i
#        best_classification = i
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(genre_data.train_data[:,0], genre_data.train_data[:,1], genre_data.train_data[:,2])
# fig = plt.figure(figsize=(12, 9))
# ax = Axes3D(fig)
# for grp_name, grp_idx in genre_data.train_data.groupby('grp').groups.items():
#     y = genre_data.train_data.iloc[grp_idx,1]
#     x = genre_data.train_data.iloc[grp_idx,0]
#     z = genre_data.train_data.iloc[grp_idx,2]
#     ax.scatter(x,y,z, label=grp_name)

#print(er)
# plt.show()
#genre_data.plot_pca()
#print("hei")
#genre_data.classify()