#!/usr/bin/env python3

from unittest import case
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D
from no_pipeline import Dataset
from sklearn.svm import SVC
from sklearn.model_selection import KFold

genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)
genre_data.scale('z-score')
control = 1- genre_data.classify('SVM')
pca_scores = []


kfold = KFold(shuffle=True, random_state=1)
for i in range(1,64):
    genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)

    genre_data.scale(normalization_method='z-score')

    genre_data.do_pca(i)
    X = genre_data.train_data.values
    Y = genre_data.train_labels.values
    svc = SVC()
    score = np.mean(cross_val_score(svc,X,Y,cv=kfold))
    pca_scores.append(score)

PCs = np.argmax(pca_scores) + 1
best_mean_score = np.amax(pca_scores)
print("Optimal no of PCs: ", PCs)
print("Best mean score: ", best_mean_score)
print("Control: ", control)

genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)
genre_data.scale('z-score')
genre_data.do_pca(PCs)
control_pca = 1- genre_data.classify('SVM')
print("Control using optimum number of PCs: ", control_pca)
