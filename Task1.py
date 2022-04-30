
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from no_pipeline import Dataset
filename = 'https://raw.githubusercontent.com/VNaas/EDC_project/main/Classification%20music/GenreClassData_30s.txt'
feat = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'tempo']
genre_data = Dataset(filename, 5, feat, None)

genre_data.scale()
error_rate = genre_data.classify(True)
print(error_rate)
plt.show()