
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from no_pipeline import Dataset

feat = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'tempo']
genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, feat, None)

error_rate = genre_data.classify(True)
print(error_rate)
plt.show()