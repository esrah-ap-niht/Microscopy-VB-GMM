
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
import h5py
import os
import numpy as np
import hyperspy.api as hs
import sys
import statistics 
import math 
from tqdm import tqdm 

# plotting, garbage collection, and tkinter settings 
gc.enable()
plt.ioff()
root = Tk()
root.withdraw()
root.attributes('-topmost',1)
sns.set(font_scale = 3)
sns.set_style("white")

# define recursive print statement. This lists all contents of an HDF5 file 
def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.attrs.keys():
            print (sep+'\t','-',key,':',obj.attrs[key])
        
        for key in obj.keys():
            print (sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        #if obj.shape == (1,) or obj.shape == (2,): 
        #    print(sep+'\t','-',':',obj[...])
        for key in obj.attrs.keys():
            print (sep+'\t','-',key,':',obj.attrs[key])
    
def h5_dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])
         
       
def main():
    # Get user to select HDF5 file
    file_path = filedialog.askopenfilename( title = "Select HDF5 File", filetypes=[("H5 files", ".h5"), ("H5 files", ".h5oina")])
    
    # check that the file is inded HDF5 or another compatable format 
    try: 
        file = h5py.File(os.path.join(file_path), "r")
        
        # Recursively print all contents of the file
        h5_dump(file_path)
        file.close()
    except OSError: 
        print("Error. Selected file type or name is incompatable with HDF5 format")
        sys.exit()
    else: 
        print("Succes. Selected file type is compatable with HDF5 format")
    
if __name__ == "__main__":
    main()







