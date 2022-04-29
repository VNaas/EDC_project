#!/usr/bin/env python3


from cProfile import label
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

    def __init__(self,filename, k, features = None, genres = None):
        # TODO Should be able to initialize with a set of genres,
        # And  make the data set only out of these. See how it's
        # done in the first line in the for loop of three_feature_plot()
        # It might help
        self.k = k
        data_frame = pd.read_csv(filename, sep='\t')
        train_data = data_frame.loc[data_frame['Type'] == 'Train']
        test_data = data_frame.loc[data_frame['Type'] == 'Test']
        if features == None:
            features = list(data_frame.columns.values)[2:65]

        ## Extract only wanted features
        self.features = features
        self.train_data = train_data[features]
        self.train_labels = train_data[['GenreID']]
        self.train_data.index = self.train_labels['GenreID']

        self.test_data = test_data[features]
        self.test_labels = test_data[['GenreID']]
        self.test_data.index = self.test_labels['GenreID']

        ## Extract only wanted genres
        genredict = {  'pop':0, 'metal':1,'disco':2,\
            'blues':3, 'reggae':4, 'classical':5,\
            'rock':6, 'hip_hop':7, 'country':8,'jazz':9}
        if genres == None:
            self.genres=[None]*len(genredict)
            self.genreIDs = [None]*len(self.genres)   
            for key in genredict:    
                self.genreIDs[genredict[key]]=genredict[key]
                self.genres[genredict[key]] = key
        else:
            self.genres = genres
            self.genreIDs = [None]*len(genres)
            for i in range(len(genres)):
                self.genreIDs[i] = genredict[genres[i]]

        frames = []
        for i in range(len(self.genreIDs)):
            frames.append(self.train_data[self.train_data.index == self.genreIDs[i]])
        self.train_data = pd.concat(frames)
        self.train_labels = self.train_data.index


        frames = []
        for i in range(len(self.genreIDs)):
            frames.append(self.test_data[self.test_data.index == self.genreIDs[i]])
        self.test_data = pd.concat(frames)
        self.test_labels = self.test_data.index
        
        self.pca = PCA()

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

    def hist(self,m,n, legend = False):
        """
        Makes a 2x2 subplot where cells are features and the overlapping
        histograms are genres.
        """
        if m*n >= len(self.features):
            plt.subplots(m,n)
            all_params = {}
            for i in range(len(self.genres)):
                samples = self.train_data[self.train_data.index == self.genreIDs[i]]
                my_label = self.genres[i]
                j = 1
                all_params[self.genres[i]] = {}
                for feature in self.features:
                    all_params[self.genres[i]][feature] = {} 
                    plt.subplot(m,n,j)
                    data = samples.loc[:,feature]
                    mean = data.mean()
                    var = data.var()
                    all_params[self.genres[i]][feature]['mean'] = mean
                    all_params[self.genres[i]][feature]['var'] = var
                    plt.hist(data, bins = 40, alpha = 0.5, label = my_label)
                    plt.title(feature)
                    j += 1
                    if legend: 
                        plt.legend(loc = 'upper left')
            return all_params



    def three_feature_plot(self):
        """
        Makes a 3-Dimensional scatter plot of a 3D feature space.
        Note that the function does not call plt.show()
        """
        if len(self.features) == 3:
            #genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hip-hop','country','jazz']
            fig = plt.figure(figsize=(12, 9))
            ax = Axes3D(fig)
            for i in range(len(self.genres)):
                samples = self.train_data[self.train_data.index == self.genreIDs[i]]
                x = samples.loc[:,self.features[0]]
                y = samples.loc[:,self.features[1]]
                z = samples.loc[:,self.features[2]]
                ax.scatter(x,y,z, label=self.genres[i])
            ax.axes.set_xlabel(self.features[0])
            ax.axes.set_ylabel(self.features[1])
            ax.axes.set_zlabel(self.features[2])
            ax.legend()
            ax.set_title("Scatter plot of three features")
        else:
            print("Did not plot 2D because of too many/little features")

    def two_feature_plot(self):
        """
        Makes a 2-Dimensional scatter plot of a 2D feature space.
        Note that the function does not call plt.show()
        """
        if len(self.features) == 2:
            # fig = plt.figure(figsize=(12, 9))

            for i in range(10):
                samples = self.train_data[self.train_data.index == i]
                x = samples.loc[:,self.features[0]]
                y = samples.loc[:,self.features[1]]
                plt.scatter(x.values,y.values)
            plt.title("Scatter plot of two features")
            plt.xlabel(self.features[0])
            plt.ylabel(self.features[1])
            plt.legend(self.genres)
            
        else:
            print("Did not plot 2D because of too many/little features")
            

    def classify(self, conf_matrix = False):
        """
        Returns: Errot rate
        Trains a KNN-classifier using the training data and 
        tests it on the test set. If conf_matrix is true, it will also
        create a confusion matrix.
        """
        classifier = KNeighborsClassifier(n_neighbors = self.k)
        classifier.fit(self.train_data,self.train_labels)
        if conf_matrix: 
            ConfusionMatrixDisplay.from_estimator(classifier, self.test_data, self.test_labels, display_labels = self.genres)
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
        self.train_data = pd.DataFrame(pca_train_data, index = self.train_data.index)
        self.test_data = pd.DataFrame(pca_test_data, index = self.test_data.index)
        self.pca = pca
        self.features = self.train_data.columns.values
    
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


