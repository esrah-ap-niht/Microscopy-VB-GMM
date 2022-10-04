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

# Get user to select analysis file 
analysis_path = filedialog.askopenfilename( title = "Select input analysis file", filetypes=[("H5 files", ".h5")] )
analysis_file = os.path.abspath( analysis_path )

# Get user to select montage file(s) 
montage_list = []
while True: 
    montage_paths = filedialog.askopenfilenames( title = "Select montage files(s). Press 'cancel' to stop adding folders", filetypes=[("H5 files", ".h5")] )
    if montage_paths != '':
        for path in montage_paths:
            montage_list.append(path)
    else:
        break

with h5py.File(analysis_file, 'r+') as file:     
    
    # Get the montages in the analysis file 
    try:
        montages = list( file['Montages'].keys() ) 
    except: 
        pass     

    # Get the unique classes in the segmentation(s) 
    try: 
        classes = []
        for montage in montages: 
            new_classes = np.unique( file['Montages'][montage]['Segmentation'][...] )
            
            for new in new_classes: 
                if new in classes: 
                    pass 
                else: 
                    classes.append(new) 
    except: 
        pass 
    
    # Get input from user on which class(es) to measure 
    classes_to_analyze = []
    print("For each of the following classes, enter 'y' to measure the class")
    print("Otherwise leave the entry field blank and press 'enter' ")
    print(" ")
    for cla in classes: 
        print(cla)
        use = input(  )
        print(" ")
        
        if use == 'y': 
            classes_to_analyze.append(cla) 

print("Enter the minimum number of pixels a feature must be equal to or larger than to be included in measurements")
min_pixels = float( input() )
print(" ")


resolutions = [] 
print("Retreiving um/pixel resolutions")
for analysis_montage in montages: 
    for montage_path in montage_list: 
        
        if os.path.basename(montage_path).replace("Montage ", "").replace(".h5", "") == analysis_montage: 
            with h5py.File(montage_path, 'r+') as file:     
                resolution = np.unique( file['Metadata']['EDS X Step Size (um)'][...] )
                
                if len(resolution) != 1: 
                    raise Exception( "More than one resolution scale detected" ) 
                else: 
                    resolution = float( resolution[0].decode('utf-8') ) 
                print(analysis_montage)
                print(resolution)
                print(" ")
                resolutions.append(resolution)

cols = ['Montage', 'Class ID', 'Area (um^2)', 'Perimeter (um)', 'Aspect Ratio', 'Equivalent Diameter (um)']
table = pd.DataFrame(columns = cols)
table.to_csv(str(analysis_file) + ".csv", mode='w', index=False, header=True)
    
with h5py.File(analysis_file, 'r+') as file:     
    for i, montage in enumerate( montages): 
        try: 
            segmentation = file['Montages'][montage]['Segmentation'][...]
            for cla in classes_to_analyze: 
                print("Measuring Montage " + str(montage) + " Class " + str(cla))
                mask = np.zeros( shape = segmentation.shape, dtype = np.uint8)
                
                mask[segmentation == cla] = 255 
                
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            
                for cnt in tqdm(contours): 
                    area = cv2.contourArea(cnt) * (resolutions[i]*resolutions[i]) # area
                    
                    if cv2.contourArea(cnt) >= min_pixels: 
                        perimeter = cv2.arcLength(cnt,True) * resolutions[i] # perimeter 
                        
                        x,y,w,h = cv2.boundingRect(cnt)
                        aspect_ratio = float(w)/h # aspect ratio 
                        
                        equi_diameter = np.sqrt(4*area/np.pi) # Equivalent Diameter
                        
                        temp = pd.DataFrame( [[montage, cla, area, perimeter, aspect_ratio, equi_diameter]], columns = cols)
            
                        table = pd.concat( [table, temp ], axis = 0, ignore_index = True, sort = False )
                        
                        if table.shape[0] >= 10_000: 
                            table.to_csv(str(analysis_file) + ".csv", mode='a', index=False, header=False)
                            table = pd.DataFrame(columns = cols)
                
        except: 
            pass 
        
table.to_csv(str(analysis_file) + ".csv", mode='a', index=False, header=False)

        










