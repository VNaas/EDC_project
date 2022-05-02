#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from no_pipeline import Dataset

gen = ['pop', 'metal', 'disco','classical']
feat = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean']
genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, feat, gen)
genre_data.scale()

er = genre_data.classify('knn',True)
genre_data.three_feature_plot()
print("Error rate when removing Tempo:" , er)
plt.show()