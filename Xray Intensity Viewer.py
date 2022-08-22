



##############################################################################################################
# import necessary packages 
##############################################################################################################



import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import gc
from tkinter import filedialog
from tkinter import *
import cv2
import random
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile as io
#import sklearn
from sklearn import mixture
#import scipy
import seaborn as sns
#from scipy.cluster.hierarchy import dendrogram
import math

from pathlib import Path
import h5py
import glob
import hyperspy.api as hs
from matplotlib.ticker import (MultipleLocator)
import matplotlib.ticker as tck
import matplotlib as mpl
from tqdm import tqdm
import collections 
import statistics

import matplotlib
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




##############################################################################################################
# plotting, garbage collection, and tkinter settings 
##############################################################################################################
gc.enable()
#plt.ioff()
plt.ion()
#%matplotlib qt


root = Tk()
root.withdraw()
root.attributes('-topmost',1)
#sns.set(font_scale = 3)
#sns.set_style("white")

# Get user to select montage file(s) 
montage_list = []
while True: 
    montage_paths = filedialog.askopenfilenames( title = "Select montage files(s). Press 'cancel' to stop adding folders")
    if montage_paths != '':
        for path in montage_paths:
            montage_list.append(path)
    else:
        break
    
output_src = filedialog.askdirectory( title = "Select output directory")
os.chdir(os.path.join(output_src))
 
for montage_path in montage_list: 
    with h5py.File(montage_path, 'r+') as file:     
        
        try:
            data = file['EDS']['Xray Intensity'][...]
        except: 
            data = np.sum( file['EDS']['Xray Spectrum'][...], axis = 2 )
        
        if len(data.shape) == 3:
            data = np.sum(data, axis = 2) 
        
        fig, ( ax1 ) = plt.subplots(1, 1, figsize = ( 40,40 ) , dpi = 400)
        fig.suptitle( os.path.basename(montage_path) )
            
        plt.imshow(data, cmap = 'gist_ncar')
        plt.savefig("Xray Intensity" + str(os.path.basename(montage_path).split(".")[0]) + ".png")    
        plt.close(fig)
        gc.collect()

print("Completed graphing all selected files")
print(" ")












