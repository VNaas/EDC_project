#!/usr/bin/env python3

from no_pipeline import Dataset
from matplotlib import pyplot as plt


min_er = 1
components = 63
ties = []
for i in range(1,64):
    genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, None, None)
    genre_data.scale()
    genre_data.do_pca(i)
    er = genre_data.classify(method = 'SVM')
    print("Error rate using SVM: ", er)
    if  er < min_er:
        ties = []
        components = i
        min_er = er
    if er == min_er:
        ties.append(i)
plt.show()

print("optimal:", ties)
print("Error rate: ", min_er)