#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from no_pipeline import Dataset

gen = ['pop', 'metal', 'disco','classical']
feat = ['spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'tempo']
genre_data = Dataset('Classification music/GenreClassData_30s.txt', 5, feat, gen)
genre_data.scale()
all_params = genre_data.hist2x2()

for genre in all_params:
    print(genre+":")
    for feature in genre:
        print("\t"+feature)
        #print("\t\tmean:",feature['mean'] )

plt.show()