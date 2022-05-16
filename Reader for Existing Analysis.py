
# import necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import gc
from tkinter import filedialog
from tkinter import *
import cv2
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile as io
import sklearn
from sklearn import mixture
import scipy
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import math 
from scipy.spatial import distance as dist
from Utility Functions import * 

gc.enable()



"""
class_number_list = [9, 10, 14, 17, 24, 22, 32]
class_number_list = [8, 9, 13, 16, 23, 21, 31]
class_name_list = ["AlTiYSi - Large Oxides", "CrNiTi - Metal?", "AlTiY - Halos", "AlSiY - Small Common", "Y rich phase", "Si Oxides", "FeSiAl - Larger Oxides"]

class_number_list = [0]
class_name_list = ["test"]

image = segmentation
measurements_dataframe = segmented_images_measurements(segmentation, class_number_list, class_name_list)
measurements_dataframe.to_csv('Measurements.csv', index=False)



fig, (ax1) = plt.subplots(1, 1, figsize=(size, size), dpi=300)
plt.hist(measurements_dataframe['Area (px)'])
plt.savefig("TEST.png") 







measurements_dataframe['Area (px)'].quantile(.10)




"""












"""

# Load model selected 
model_src = filedialog.askdirectory( title = "Select existing model directory")
print("Model directory chosen")
try:
    os.chdir(model_src)
    files = os.listdir()
    for file in files: 
        if ('_Means.npy' in file) or ('_means.npy' in file):
            means = np.load(file)
        elif ('_Covariances.npy' in file) or ('_covariances.npy' in file):
            covar = np.load(file)
        elif ('_Weights.npy' in file) or ('_weights.npy' in file):
            weights = np.load(file)
        
    DPGMM = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    DPGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    DPGMM.weights_ = weights
    DPGMM.means_ = means
    DPGMM.covariances_ = covar
    components = len(DPGMM.weights_)
    uncertainty = np.load('Log_Probabilities.npy')
    
    # Return up one directory level 
    os.chdir( os.path.dirname(os.getcwd()) )
    print("SUCCESS")
except: 
    print("ERROR")
    pass 


"""
