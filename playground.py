#!/usr/bin/env python3
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt


blue_x = np.array([0.2,0.22,0.1,0.222,0.18,0.19,0.24,0.23,0.21,0.185])+0.1
red_x = np.array([0.5,0.49,0.51,0.511,0.48,0.53,0.488,0.43,0.45,0.532])

blue_y = np.random.uniform(0,1,10)
red_y = np.random.uniform(0,1,10)
#blue_y = np.random.normal(0.5,0.2,10,seed = 1)
#red_y = np.random.normal(0.2,0.2,10,seed = 1)

plt.subplot(1,2,1)
plt.scatter(blue_x,np.zeros(10),c='b')
plt.scatter(red_x,np.zeros(10),c='r')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Feature X")

plt.subplot(1,2,2)
plt.scatter(blue_x,blue_y,c='b')
plt.scatter(red_x,red_y,c='r')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.show()
