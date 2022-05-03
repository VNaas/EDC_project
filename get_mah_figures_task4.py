import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from no_pipeline import Dataset
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib 
matplotlib.rc('axes', titlesize=18) 
matplotlib.rc('axes', labelsize=18)
matplotlib.rc('xtick', labelsize = 14)
matplotlib.rc('ytick', labelsize = 14)
matplotlib.rc('legend', fontsize = 14)



filename = 'Classification music/GenreClassData_30s.txt'
genre_data = Dataset(filename, 5, None, None)
genre_data.scale('z-score')
control = 1- genre_data.classify('SVM')
pca_scores_knn = np.array([])
pca_scores_svc = np.array([])

kfold = KFold(shuffle=True, random_state=1)

for i in range(1,64):
    genre_data = Dataset(filename, 5, None, None)
    genre_data.scale(normalization_method='z-score')
    genre_data.do_pca(i)
    X = genre_data.train_data.values
    Y = genre_data.train_labels.values
    knn = KNeighborsClassifier()
    svc = SVC()
    score_svc = np.mean(cross_val_score(svc,X,Y,cv=kfold))
    score_knn = np.mean(cross_val_score(knn,X,Y,cv=kfold))
    pca_scores_knn = np.append(pca_scores_knn, score_knn)
    pca_scores_svc = np.append(pca_scores_svc, score_svc)


PCs = np.argmax(pca_scores_knn) + 1
best_mean_score = np.amax(pca_scores_knn)
# print("Optimal no of PCs: ", PCs)
# print("Best mean score: ", best_mean_score)
# print("Control, using all features: ", control)

# genre_data = Dataset(filename, 5, None, None)
# genre_data.scale('z-score')
# genre_data.do_pca(PCs)
# control_pca_knn = 1- genre_data.classify('knn')

# print("Score using optimum number of PCs: ", control_pca_knn)
# print("Error rate using optimal number of PCs: ",1-control_pca_knn)
fig2 = plt.figure(figsize=(15,9))
plt.title("Error Rate Per Principle Component")
plt.xlabel("Principle components")
plt.ylabel("Error rate")
plt.plot(1-pca_scores_knn, label = "k-NN")
plt.plot(1-pca_scores_svc, label = "SVC")
plt.legend(loc = 'upper right')
plt.show()