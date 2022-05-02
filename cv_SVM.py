#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from no_pipeline import Dataset
from sklearn.model_selection import KFold
from sklearn.svm import SVC


genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)
genre_data.scale('z-score')
genre_data.do_pca(45)
X_train, X_test, y_train, y_test = train_test_split(genre_data.train_data.values, genre_data.train_labels.values, test_size=0.2,random_state=1)
svc = SVC()
svc.fit(X_train,y_train)

best_score = np.mean(cross_val_score(svc,X_test,y_test, cv=5))
control = best_score


kernels = ['rbf','linear']
Cs = [0.001,0.1,1,10,100,1000,1000]
gammas = [0.001,0.001,0.01,0.1,1,10,100,1000, 'scale','auto']

best_kernel = ''
best_C = 1
best_gamma = 1
genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)
genre_data.scale('z-score')
genre_data.do_pca(45)
for kern in kernels:
    for c in Cs:
        for gam in gammas:
            kfold = KFold(shuffle=True, random_state=1)
            #X_train, X_test, y_train, y_test = train_test_split(genre_data.train_data.values, genre_data.train_labels.values, test_size=0.2,random_state=1)
            if kern == 'rbf':
                svc = SVC(kernel = kern, C = c, gamma = gam)
            else:
                svc = SVC(kernel = kern,C=c)
            X = genre_data.train_data.values
            Y = genre_data.train_labels.values
            score = np.mean(cross_val_score(svc,X,Y,cv=kfold))
            if score > best_score:
                best_kernel, best_C, best_gamma =  kern, c, gam
                best_score = score

print("Best kernel function: ", best_kernel)
print("Best C: ", best_C)
if best_kernel == 'rbf':
    print("Best gamma: ", best_gamma)
print("Best mean score: ", best_score)
print("Control: ", control)

genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)
genre_data.do_pca(45)
genre_data.scale('z-score')
er = genre_data.classify('SVM', conf_matrix=True, kernel=best_kernel, C=best_C, gamma=best_gamma)
print("Error rate using best parameters: ", er)
plt.show()