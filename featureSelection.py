#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

from no_pipeline import Dataset
filename = 'https://raw.githubusercontent.com/VNaas/EDC_project/main/Classification%20music/GenreClassData_30s.txt'
gen = ['pop', 'metal', 'disco','classical']
feat = ['spectral_centroid_mean', 'mfcc_1_mean', 'spectral_rolloff_mean', 'spectral_contrast_var']
genre_data = Dataset(filename, 5, feat, None)
genre_data.scale()


## LOOK AT HISTOGRAM AND ERROR RATE
genre_data.hist(2,2,True)
er = genre_data.classify(True)
print(er)

## LOOK AT CORRELATION BETWEEN FEATURES
test_features = ['rmse_mean', 'spectral_bandwidth_mean', 'spectral_contrast_var','chroma_stft_12_std']
for f in test_features:
    feat[3] = f
    genre_data = Dataset(filename, 5, feat, None)
    genre_data.scale()
    er = genre_data.classify()
    print("Error rate using " + f + " as fourth feature:\n\t", er)
    genre_data.train_data.index = list(range(len(genre_data.train_data.index))) # to avoid a pairplot error
    sns.pairplot(genre_data.train_data)

##CALCULATE VARIANCES
# pd.set_option("display.max_colwidth", None,"display.max_rows",None)

# description = genre_data.train_data.describe(include='all')
# variances = genre_data.train_data.var()
# display(variances)
# display(description)


plt.show()