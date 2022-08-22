
##############################################################################################################
# import necessary packages 
##############################################################################################################
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
import cv2

from pathlib import Path
import h5py
import glob
import hyperspy.api as hs
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)
import matplotlib.ticker as tck
import matplotlib as mpl
from tqdm import tqdm
import collections 

##############################################################################################################
# plotting, garbage collection, and tkinter settings 
##############################################################################################################
gc.enable()
plt.ioff()
root = Tk()
root.withdraw()
root.attributes('-topmost',1)
sns.set(font_scale = 3)
sns.set_style("white")




# Conversion of covariance matrix to correlation matrix (i.e. normalization)
def correlation_from_covariance( covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# Create linkage matrix and plot the dendrogram. Used for visuals to determine what classes are most similar
def plot_dendrogram( model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Cluster Label")
    plt.ylabel("Dissimilarity Scale")
    return linkage_matrix
    





def plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, bk_grd, montage_path, metadata, analysis_file): 
    global x_min
    global y_min
    size = 40
    graphing_dpi = 300
    color_fixed = 'bwr_r'
    color_floating = 'gist_ncar_r'
    fontsize = 60
    ratio = 2 
    background = np.array(background, dtype = np.float32)
    uniques = range(len(Weights))
    analysis = Path(analysis_file).stem
    
    # Verfiy that the background image and segmentation are of the same shape. 
    # If not, resize the background image to have the same shape as the segmentation 
    # Note: The CV2 Resize function has backwards shape (y,x)
    if ( background.shape[0] != segmentation.shape[0] ) or ( background.shape[1] != segmentation.shape[1] ):
        background = cv2.resize(background, dsize = ( segmentation.shape[1], segmentation.shape[0] ), interpolation = cv2.INTER_CUBIC )
        
    
    try: 
        montage = Path(montage_path).stem.replace("Montage ", "")
    except NameError:
        montage = "Unlabeled"
        
    try:
        with h5py.File(analysis_file, 'r+') as file: 
            analysis_kev = file['Channel KeV Peaks'][...]
            analysis_kev = [ round(x, 2) for x in analysis_kev]
    except: 
        pass 
    
    try:
        with h5py.File(analysis_file, 'r+') as file: 
            display_shells = file['Autodetected Peak Labels'][...]
    except: 
        pass     
         
    try: 
        display_shells = [ str(round(analysis_kev[q], 2)) + " " + display_shells[q].decode('utf-8') for q in range(len(analysis_kev))]
    except: 
        display_shells = [round(analysis_kev[q], 2) for q in range(len(analysis_kev))]
        
        
    try :
        with h5py.File(montage_path) as f:
            headers = list( f['Metadata'].keys() )
            metadata = pd.DataFrame(columns = headers)
            
            for key in headers: 
                metadata[key] = f['Metadata'][ str(key) ][...]
                
        # Get the um/px scale 
        x_scale = []
        y_scale = [] 
        for i, value in enumerate( metadata['EDS X Step Size (um)'] ): 
            try: 
                value = float( value.decode('utf-8') )
                test = metadata['Montage Label'][i].decode('utf-8') == montage
                if test: 
                    x_scale.append(value) 
            except: 
                pass 
                
        for i, value in enumerate( metadata['EDS Y Step Size (um)'] ): 
            try: 
                value = float( value.decode('utf-8') ) 
                test = metadata['Montage Label'][i].decode('utf-8') == montage
                if test: 
                    y_scale.append(value)         
            except: 
                pass 
            
        # remove nan if present 
        x_scale = [x for x in x_scale if np.isnan(x) == False]
        y_scale = [x for x in y_scale if np.isnan(x) == False]
        
        x_scale = np.unique(x_scale)
        y_scale = np.unique(y_scale)
        
        # verify that only one value for the resolution exists for a given montage 
        if (len(x_scale) > 1) or (len(y_scale) > 1): 
            raise Exception("Multiple unique values found for um/px resolution: " + str(x_scale) + ", " + str(y_scale) ) 
        else: 
            x_scale = x_scale[0]
            y_scale = y_scale[0]
            
        x_list =  []
        y_list = []
        for i, value in enumerate( metadata['EDS Stage X Position (mm)'] ): 
            try: 
                value = float( value.decode('utf-8') )
                test = metadata['Montage Label'][i].decode('utf-8') == montage
                if test: 
                    x_list.append(value) 
            except: 
                pass 
            
        for i, value in enumerate( metadata['EDS Stage Y Position (mm)'] ): 
            try: 
                value = float( value.decode('utf-8') )
                test = metadata['Montage Label'][i].decode('utf-8') == montage
                if test: 
                    y_list.append(value) 
            except: 
                pass 
                
        # find minimum coodinates so that we can create an appropriately sized array 
        x_min = int( math.ceil(min(x_list)*1000/x_scale))
        y_min = int( math.ceil(min(y_list)*1000/y_scale))
      
                  
        def forward_x(x): 
            return (x + x_min)/1000
            
        def reverse_x(x): 
            return (x - x_min)/1000
        
        def forward_y(y):
            return (y + y_min)/1000
        
        def reverse_y(y):
            return (y - y_min)/1000
    except: 
        pass 
           

        
    # Create agglomerative hierarchial model and save results 
    fig, ax = plt.subplots(figsize=(size, size))
    heir = sklearn.cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage = 'single')
    clustering = heir.fit(Means)  
    linkage_matrix = plot_dendrogram(clustering, truncate_mode='level', p=100)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(str(analysis) + " Consolidation Dendrogram" + ".png")
    plt.close(fig)
    gc.collect()
    
    # Prepare HAM linkage matrix for exporting 
    # Remove all non-first-level linkages 
    linkage_matrix = np.delete(linkage_matrix,3,1)
    
    # Reformat to three columns 
    linkage_matrix = np.concatenate( (np.delete(linkage_matrix, 1, 1), np.delete(linkage_matrix, 0, 1) ), axis = 0)
    
    index_to_delete = []
    for i, row in enumerate(linkage_matrix[:,0]):
        if (row in uniques) == False:
            index_to_delete.append(i)
    
    linkage_matrix = np.delete(linkage_matrix, np.array(index_to_delete, dtype = np.int16), 0)
    linkage_matrix = pd.DataFrame( data = linkage_matrix, columns = ["Class ID", "Dissimilarity Scale"])
    linkage_matrix = linkage_matrix.sort_values(by=['Class ID'])
    linkage_matrix['Class Weights'] = Weights * 100.0
    
    area = []
    for row in uniques:        
        area.append( np.sum(segmentation == row) / segmentation.size * 100 )
    
    linkage_matrix['Area Fraction'] = area 
    linkage_matrix['Training/Testing Ratio'] = linkage_matrix['Class Weights'] / linkage_matrix['Area Fraction']
    
    try: 
        for i in range( Means.shape[1] ):
               linkage_matrix['KeV' + str( display_shells[i] ) ] = Means[:,i]   
    except: 
        for i in range( Means.shape[1] ):
            linkage_matrix['Energy Bin: ' + str( i ) ] = Means[:,i]
    linkage_matrix.to_excel(str(montage) + " Class Data.xlsx", index = False)

    ########
    # Create absolute uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    # if available, plot the background image as well
    try:
        plt.imshow(background)
        plot = plt.imshow(uncertainty, cmap = color_fixed, vmin = -40, vmax = 0, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = color_fixed, vmin = -40, vmax = 0 )
        pass 
    
    # we wish to provide users with information both about the stage location and px/um scaling
    # It is not strictly necessary to provide a scale bar, because if the stage coordinates are provided, users can easily get the scaling
    
    # Turn off all tick marks and labels for the images 
    plt.xticks([])
    plt.yticks([])
  
    try: 
        # add secondary axis labels for the stage locations     
                
        secondx_ax = ax.secondary_xaxis('bottom', functions = (forward_x, reverse_x))
        secondx_ax.set_xlabel('X Stage Location (mm)')
        secondx_ax.tick_params(labelsize=30)
        secondx_ax.tick_params(which='major', width = 5, length=15)
            
        secondy_ax = ax.secondary_yaxis('left',  functions = (forward_y, reverse_y))
        secondy_ax.set_ylabel('Y Stage Location (mm)')
        secondy_ax.tick_params(labelsize=30)
        secondy_ax.tick_params(which='major', width = 5, length=15)
    except NameError: 
        pass 
    
    # add figure title 
    plt.title(str(montage) + " Fixed Scale Log Liklihoods", fontsize = fontsize)
            
    # auto-fit padding and spacing, then save figure and clear memory 
    plt.tight_layout()
    plt.savefig(str(montage) + " Fixed Scale Log Liklihoods Without Colorbar" + str(bk_grd) + ".png")
    
    # add color bar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    
    plt.savefig(str(montage) + " Fixed Scale Log Liklihoods " + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
    
    ########
    # Create relative uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    # if available, plot the background image as well
    try:
        plt.imshow(background)
        plot = plt.imshow(uncertainty, cmap = color_floating, vmax = 0, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = color_floating, vmax = 0 )
        pass 
    
    # we wish to provide users with information both about the stage location and px/um scaling
    # It is not strictly necessary to provide a scale bar, because if the stage coordinates are provided, users can easily get the scaling
    
    # Turn off all tick marks and labels for the images 
    plt.xticks([])
    plt.yticks([])
  
    try: 
        # add secondary axis labels for the stage locations             
        secondx_ax = ax.secondary_xaxis('bottom', functions = (forward_x, reverse_x))
        secondx_ax.set_xlabel('X Stage Location (mm)')
        secondx_ax.tick_params(labelsize=30)
        secondx_ax.tick_params(which='major', width = 5, length=15)
            
        secondy_ax = ax.secondary_yaxis('left',  functions = (forward_y, reverse_y))
        secondy_ax.set_ylabel('Y Stage Location (mm)')
        secondy_ax.tick_params(labelsize=30)
        secondy_ax.tick_params(which='major', width = 5, length=15)
    except NameError: 
        pass 
    
    # add figure title 
    plt.title(str(montage) + " Floating Scale Log Liklihoods", fontsize = fontsize)
    
    plt.tight_layout()
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods Without Colorbar" + str(bk_grd) + ".png")
    
    # add color bar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    
    # auto-fit padding and spacing, then save figure and clear memory 
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods with Colorbar" + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
         
    ########
    # Create boxed relative uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    # if available, plot the background image as well
    temp = uncertainty.copy() 
    temp[temp >= -25] = 0   
    temp[temp < -25] = 255   
    temp = np.array(temp, dtype = np.uint8)
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del temp 
    
    temp = np.zeros(dtype = np.uint8, shape = (uncertainty.shape[0], uncertainty.shape[1]) )
    #temp = np.ma.array(temp)
    
    
    for i, cnt in enumerate(contours): 
        x,y,w,h = cv2.boundingRect(cnt)
        
        if (hierarchy[0][i][3] == -1) and (cv2.contourArea(cnt) > 4):
            cv2.rectangle(temp, (x,y), (x+w,y+h), (50), 3)
    
    temp = np.ma.masked_where(temp[:,:] == 0, temp)
    
    
    try:
        plt.imshow(background)

        plot = plt.imshow(uncertainty, cmap = color_floating, vmax = 0, alpha = 0.60 )
        plt.imshow(temp, cmap = 'Reds_r', vmax = 255, vmin = 0)

    except NameError:
        plot = plt.imshow(uncertainty, cmap = color_floating, vmax = 0 )
        pass 
    
    # we wish to provide users with information both about the stage location and px/um scaling
    # It is not strictly necessary to provide a scale bar, because if the stage coordinates are provided, users can easily get the scaling
    
    # Turn off all tick marks and labels for the images 
    plt.xticks([])
    plt.yticks([])
  
    try: 
        # add secondary axis labels for the stage locations             
        secondx_ax = ax.secondary_xaxis('bottom', functions = (forward_x, reverse_x))
        secondx_ax.set_xlabel('X Stage Location (mm)')
        secondx_ax.tick_params(labelsize=30)
        secondx_ax.tick_params(which='major', width = 5, length=15)
            
        secondy_ax = ax.secondary_yaxis('left',  functions = (forward_y, reverse_y))
        secondy_ax.set_ylabel('Y Stage Location (mm)')
        secondy_ax.tick_params(labelsize=30)
        secondy_ax.tick_params(which='major', width = 5, length=15)
    except NameError: 
        pass 
    
    # add figure title 
    plt.title(str(montage) + " Floating Scale Log Liklihoods", fontsize = fontsize)
    
    plt.tight_layout()
    
    # add color bar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    
    # auto-fit padding and spacing, then save figure and clear memory 
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods with Boxes " + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
    
    ########
    # Save semantic segmentation map 
    fig, ax = plt.subplots(figsize = (size, size), dpi=graphing_dpi) 
    ax = plt.subplot()
    cmap = plt.get_cmap('gist_ncar', np.max(segmentation)-np.min(segmentation)+1)
    plt.gca().invert_yaxis()
    try:
        plt.imshow(background)
        plot = plt.imshow(segmentation, cmap=cmap, alpha = 0.7, vmin = np.min(segmentation)-.5, vmax = np.max(segmentation)+.5)
    except NameError:
        plot = plt.imshow(segmentation, cmap=cmap, alpha = 1, vmin = np.min(segmentation)-.5, vmax = np.max(segmentation)+.5)
        pass 
    plt.xticks([])
    plt.yticks([])
    
    try: 
        # add secondary axis labels for the stage locations             
        secondx_ax = ax.secondary_xaxis('bottom', functions = (forward_x, reverse_x))
        secondx_ax.set_xlabel('X Stage Location (mm)')
        secondx_ax.tick_params(labelsize=30)
        secondx_ax.tick_params(which='major', width = 5, length=15)
            
        secondy_ax = ax.secondary_yaxis('left',  functions = (forward_y, reverse_y))
        secondy_ax.set_ylabel('Y Stage Location (mm)')
        secondy_ax.tick_params(labelsize=30)
        secondy_ax.tick_params(which='major', width = 5, length=15)
    except NameError: 
        pass 
    
    plt.title(str(montage) + " Class Segmentation", fontsize = fontsize*(0.7))
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(str(montage) + " Class Segmentation Without Colorbar "  + str(bk_grd) + ".png", dpi=graphing_dpi)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax, ticks=np.arange(np.min(segmentation),np.max(segmentation)+1))
    cbar.ax.tick_params(labelsize=25) 

    plt.savefig(str(montage) + " Class Segmentation with Colorbar "  + str(bk_grd) + ".png", dpi=graphing_dpi)
    plt.close(fig)
    gc.collect()       

    # Create binary class maps and save 
    w = (Weights).argsort()
    precisions_cholesky = np.linalg.cholesky(np.linalg.inv(Covariance))
    
    for ind2 in np.unique(segmentation):
        
        gc.collect()
        
        class_map = np.ones(shape = segmentation.shape, dtype = np.uint8 )
        class_map[segmentation != ind2] = 0
        dilation = cv2.dilate(class_map, np.ones((3,3), 'uint8'), iterations = 3)
        dilation = dilation * 255
        class_map = class_map * 255
        class_map =  np.ma.masked_where(segmentation != ind2, class_map, copy = True )
              
        
        fontsize = 30
        fig, ax1 = plt.subplots(figsize = (size, size), dpi=graphing_dpi) 
        #gs = fig.add_gridspec( ncols = 6, nrows = 3)
        
        #ax1 = fig.add_subplot(gs[0:3, 0:3])
        #ax1 = fig.add_subplot()
        ax1.set_title('Class Map: ' + str(ind2), fontsize = 15)
        
        plt.sca(ax1)
        plt.tick_params(left = True)
        plt.tick_params(bottom = True)
        
        plot = plt.imshow( dilation, cmap = 'Blues', alpha = 1)     
        plot = plt.imshow( class_map, cmap = 'autumn', alpha = 1)  
        
        plt.yticks([])
        plt.xticks([])
        
        try: 
            # add secondary axis labels for the stage locations             
            secondx_ax = ax1.secondary_xaxis('bottom', functions = (forward_x, reverse_x))
            secondx_ax.set_xlabel('X Stage Location (mm)')
            secondx_ax.tick_params(labelsize=30)
            secondx_ax.tick_params(which='major', width = 5, length=15)
                
            secondy_ax = ax1.secondary_yaxis('left',  functions = (forward_y, reverse_y))
            secondy_ax.set_ylabel('Y Stage Location (mm)')
            secondy_ax.tick_params(labelsize=30)
            secondy_ax.tick_params(which='major', width = 5, length=15)
        except NameError: 
            pass 
        
        """
        divider = make_axes_locatable(ax1)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #cb = plt.colorbar(plot, cax=cax)   
        #cb.ax.tick_params(labelsize=fontsize) 
        #cb.remove()
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.figure.axes[-1].yaxis.label.set_size(10)    # colorbar label size 
        ax1.figure.axes[-1].xaxis.label.set_size(10)    # colorbar label size 
        #plt.xlabel("Scale (um)", fontsize = 12)
        
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        ax1.set_aspect(aspect = segmentation.shape[0]/segmentation.shape[1], adjustable = 'box')
        """
        try:
            plt.sca(ax1)
            plot = plt.imshow(background, alpha = 0.3, cmap = 'gray_r')
        except NameError:
            pass 
                
        plt.tight_layout()
       
        
        plt.savefig(str(montage) + " Class " + str(ind2) + " " + str(bk_grd) + ".png")
        gc.collect()

    try: 
        # Create gridview to indicate where each tile comes from 
        fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
        
        blank = background.copy()
        """
        if (blank.dtype == np.float32) or (blank.dtype == np.int32) or (blank.dtype == np.uint32):
            blank = blank/(2^32)
            blank= blank*255.0
        """    
            
            
        
        try: 
            blank = cv2.cvtColor(np.array(blank, dtype = np.uint8),cv2.COLOR_GRAY2RGB)
        except:
            pass 
        
        start_x = []
        for value in metadata['EDS Stage X Position (mm)']:
            start_x.append(value.decode('utf-8'))
        start_x = np.min( np.array( start_x , dtype = float) )
        
        start_y = []
        for value in metadata['EDS Stage Y Position (mm)']:
            start_y.append(value.decode('utf-8'))
        start_y = np.min( np.array( start_y , dtype = float) )
        
        
        for index, row in metadata.iterrows():
            #print(index)
            x = int( (float( row['EDS Stage X Position (mm)'].decode('utf-8') ) - start_x) * 1_000.0/ float(row['EDS X Step Size (um)'].decode('utf-8') ) )
            y = int( (float( row['EDS Stage Y Position (mm)'].decode('utf-8') ) - start_y) * 1_000.0/ float(row['EDS Y Step Size (um)'].decode('utf-8') ) )
            x_range = int( row['EDS Number of X Cells'].decode('utf-8') )
            y_range = int( row['EDS Number of Y Cells'].decode('utf-8') )
            cmp = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) 
            title = str( row['EDS Field'].decode('utf-8') )
            
            cv2.rectangle(blank, (x,y), (x+x_range, y+y_range) , cmp, 10 )
            cv2.putText(blank, title, (x + int(x_range/10), y + int(y_range/2) ), cv2.FONT_HERSHEY_TRIPLEX, 3, cmp, 10)
        
        plt.imshow(blank)
       
        plt.xticks([])
        plt.yticks([])
        
        plt.title(str(montage) + " Gridview", fontsize = fontsize)
        
        plt.tight_layout()
        plt.savefig(str(montage) + " Gridview " + str(bk_grd) + ".png")
        plt.close(fig)
        gc.collect()

    except: 
        print("Gridview Error")
        pass 

    try: 
        # Create gridview to indicate where each tile comes from 
        fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
        
        blank = np.zeros( shape = (segmentation.shape[0], segmentation.shape[1]) , dtype = np.uint8)
        blank2 = np.zeros( shape = (segmentation.shape[0], segmentation.shape[1]) , dtype = np.uint8)
        
        start_x = []
        for value in metadata['EDS Stage X Position (mm)']:
            start_x.append(value.decode('utf-8'))
        start_x = np.min( np.array( start_x , dtype = float) )
        
        start_y = []
        for value in metadata['EDS Stage Y Position (mm)']:
            start_y.append(value.decode('utf-8'))
        start_y = np.min( np.array( start_y , dtype = float) )
        
        
        for index, row in metadata.iterrows():
            #print(index)
            x = int( (float( row['EDS Stage X Position (mm)'].decode('utf-8') ) - start_x) * 1_000.0/ float(row['EDS X Step Size (um)'].decode('utf-8') ) )
            y = int( (float( row['EDS Stage Y Position (mm)'].decode('utf-8') ) - start_y) * 1_000.0/ float(row['EDS Y Step Size (um)'].decode('utf-8') ) )
            x_range = int( row['EDS Number of X Cells'].decode('utf-8') )
            y_range = int( row['EDS Number of Y Cells'].decode('utf-8') )
            #cmp = random.randint(0, 255)
            cmp = 255
            title = str( row['EDS Field'].decode('utf-8') )
            
            cv2.rectangle(blank, (x,y), (x+x_range, y+y_range) , cmp, 10 )
            cv2.putText(blank, title, (x + int(x_range/10), y + int(y_range/2) ), cv2.FONT_HERSHEY_TRIPLEX, 3, cmp, 10)
            cv2.putText(blank2, title, (x + int(x_range/10), y + int(y_range/2) ), cv2.FONT_HERSHEY_TRIPLEX, 3, (255,255,255), 16)
        
        alpha = np.zeros( shape = segmentation.shape, dtype = np.uint8 )
        alpha[ blank > 0 ] = 1
        alpha = np.ma.masked_where(alpha == 0, alpha)
        
        alpha2 = np.zeros( shape = segmentation.shape, dtype = np.uint8 )
        alpha2[ blank2 > 0 ] = 1
        alpha2 = np.ma.masked_where(alpha2 == 0, alpha2)
        
        cmap = plt.get_cmap('gist_ncar', np.max(segmentation)-np.min(segmentation)+1)

        try:
            plt.imshow(background)
            plot = plt.imshow(segmentation, cmap=cmap, alpha = 0.7, vmin = np.min(segmentation)-.5, vmax = np.max(segmentation)+.5)
        except NameError:
            plot = plt.imshow(segmentation, cmap=cmap, alpha = 1, vmin = np.min(segmentation)-.5, vmax = np.max(segmentation)+.5)
            pass 
        
        plt.imshow(blank2, cmap = 'binary', alpha = alpha2)
        plt.imshow(blank, cmap = 'gray', alpha = alpha)

        plt.xticks([])
        plt.yticks([])
        plt.title(str(montage) + " Gridview", fontsize = fontsize)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax, ticks=np.arange(np.min(segmentation),np.max(segmentation)+1))
        cbar.ax.tick_params(labelsize=25) 
    
        
        plt.tight_layout()
        plt.savefig(str(montage) + " Gridview " + str("Segmentation") + ".png")
        plt.close(fig)
        gc.collect()

    except: 
        print("Gridview Error")
        pass 





def plot_model(Means, Covariance, Weights, uniques, analysis_file, segmentation): 
    size = 40
    graphing_dpi = 400
    color = 'brg'
    fontsize = 60
    ratio = 2 
   
    analysis = Path(analysis_file).stem
    components = len(Weights)

    # Save cluster weight graph
    fig, ax = plt.subplots(figsize=(size, size))
    plot_w = np.arange(components) + 1
    ax.bar(plot_w - 0.5, np.sort(Weights)[::-1], width=1., lw=0);
    ax.set_xlim(0.5, components);
    ax.set_xlabel('Dirichlet Process Components');
    ax.set_ylabel('Posterior expected mixture weight');
    fig.suptitle("Mixture Weight per Class" ,fontsize=20 )
    plt.tight_layout()
    plt.savefig( str(analysis) + " Mixture Weights" + ".png", dpi=graphing_dpi)
    plt.close(fig)
    gc.collect()
    
    precisions_cholesky = np.linalg.cholesky(np.linalg.inv(Covariance))
    try:
        with h5py.File(analysis_file, 'r+') as file: 
            analysis_kev = file['Channel KeV Peaks'][...]
            analysis_kev = [ round(x, 2) for x in analysis_kev]
    except: 
        pass 
    
    try:
        with h5py.File(analysis_file, 'r+') as file: 
            display_shells = file['Autodetected Peak Labels'][...]
    except: 
        pass     
         
    try: 
        display_shells = [ str(round(analysis_kev[q], 2)) + " " + display_shells[q].decode('utf-8') for q in range(len(analysis_kev))]
    except: 
        display_shells = [round(analysis_kev[q], 2) for q in range(len(analysis_kev))]
        
            
    for i in range(Covariance.shape[0] ):
        fig = plt.figure(constrained_layout=False, figsize=(10,5), dpi=200)

        ax2 = fig.add_subplot()
        ax2.set_title("Class " + str(i) + " Correlation Matrix", fontsize = 15)
        
        
        plt.sca(ax2)
        stdev = np.sqrt(np.abs(Covariance[i]))
        stdev[Covariance[i] < 0.0] = -1 * stdev[Covariance[i] < 0.0] 
        
        
        try:     
            sns.heatmap(correlation_from_covariance(Covariance[i]), xticklabels = display_shells, yticklabels = display_shells, vmin = 0, vmax = 1, linewidths=1, linecolor = 'white', cmap = color, mask = np.triu(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'vertical'})
        except NameError: 
            sns.heatmap(correlation_from_covariance(Covariance[i]), vmin = 0, vmax = 1, linewidths=1, linecolor = 'white', cmap = color, mask = np.triu(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'vertical'})
    
        ax2.tick_params(axis='x', pad=5)

        ax2.set_yticklabels( ax2.get_yticklabels(), rotation=0)
        ax2.set_xticklabels( ax2.get_xticklabels(), rotation=90)
            
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(6)
            
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)               # colorbar tick size 
        ax2.figure.axes[-1].yaxis.label.set_size(10)    # colorbar label size 
        ax2.figure.axes[-1].xaxis.label.set_size(10)    # colorbar label size 
        plt.xlabel("Electron Shell (KeV)", fontsize = 12)
        plt.ylabel("Electron Shell (KeV)", fontsize = 12)
        
        plt.tight_layout()
        plt.savefig("Class " + str(i) + " Correlation Matrix.png")
        plt.close(fig)
        gc.collect()
        
                           

def main():
    # Get user to select analysis file 
    analysis_path = filedialog.askopenfilename( title = "Select input analysis file", filetypes=[("H5 files", ".h5")])
    analysis_file = os.path.abspath( analysis_path )

    # Get user to select montage directories 
    montage_src = []
    while True: 
        get_path = filedialog.askdirectory( title = "Select montage directory(s). Press 'cancel' to stop adding folders")
        if get_path != '':
            montage_src.append(get_path)
        else:
            break 
         
    montage_list = []
    for src in montage_src:
        for file in glob.glob(src + '/**/*', recursive=True):
            if (Path(file).suffix) in ['.h5'] and ('Montage ' in Path(file).stem ):
                montage_list.append(file)
                
    # Get user to select output directory 
    output_src = filedialog.askdirectory( title = "Select output directory")
    os.chdir(os.path.join(output_src))
    with h5py.File(analysis_file, 'r+') as file: 
        unique_montages = list( file['Montages'].keys() )
        
    for montage in unique_montages: 
        with h5py.File(analysis_file, 'r+') as file:     
            
            # find and load the segmentation map 
            try:      
                segmentation = file['Montages'][str(montage)]['Segmentation'][...]
            except: 
                raise Exception("Error: Segmentation not found for montage: " + str(montage) )
            
            # Find and load the final uncertainy map 
            try:      
                keys = file['Montages'][str(montage)].keys() 
                keys2 = []
                for key in keys: 
                    if "Uncertainty" in key: 
                        key = key.replace("Uncertainty", "")
                        key = key.replace("Generation", "")
                        key = key.strip()

                        keys2.append( int(key) ) 
                        
                uncertainty = file['Montages'][str(montage)]['Generation ' + str( max( keys2 ) ) + " Uncertainty"][...]
            except: 
                raise Exception("Error: Segmentation not found for montage: " + str(montage) )
        
    print("All montage segmentations and uncertainty maps successfully located")
    print("")
       
    if len(montage_list) > 0:
        available_backgrounds = []
        
        for montage in unique_montages:
            for montage_path in montage_list: 
                
                if montage in montage_path: 
                    with h5py.File(montage_path, 'r+') as file: 
                        for folder in ['Electron Image', 'EDS', 'EBSD']: 
                            
                            try: 
                                
                                keys = list( file[str(folder)].keys() )
                                for background in keys: 
                                    
                                    if (background != 'Xray Spectrum') and ((background in available_backgrounds) == False) and ( len(file[str(folder)][background].shape) == 2):
                                        available_backgrounds.append(background) 
                            except: 
                                pass 
                            
        if len(available_backgrounds) > 0:
            print("The following background channels are available to be overlaid on AI outputs")
            print("For each background, indicate with 'y' or 'n' whether the background should be used in graphing:")
            print("")
            for background in available_backgrounds: 
                print(str(background) )
            
            print("#####")
            print("")
            
            use_background = []
            for background in available_backgrounds: 
                print(str(background) )
                use_background.append( input("") )
                print("")
        else: 
           pass 
           
        ########      
        
        for montage in unique_montages: 
            for montage_path in montage_list: 
                if montage in montage_path: 
                    with h5py.File(montage_path, 'r+') as montage_file:    
                        with h5py.File(analysis_file, 'r+') as file: 
                               
                            
                            
                            # find and load the segmentation map 
                            try:      
                                segmentation = file['Montages'][str(montage)]['Segmentation'][...]
                                segmentation = np.array(segmentation, dtype = np.uint8 )
                            except: 
                                raise Exception("Error: Segmentation not found for montage: " + str(montage) )
                            
                            # Find and load the final uncertainy map 
                            try:      
                                keys = file['Montages'][str(montage)].keys() 
                                keys2 = []
                                for key in keys: 
                                    if "Uncertainty" in key: 
                                        key = key.replace("Uncertainty", "")
                                        key = key.replace("Generation", "")
                                        key = key.strip()
                
                                        keys2.append( int(key) ) 
                                        
                                uncertainty = file['Montages'][str(montage)]['Generation ' + str( max( keys2 ) ) + " Uncertainty"][...]
                                uncertainty = np.array(uncertainty, dtype = np.float32 )
    
                            except: 
                                raise Exception("Error: Uncertainty not found for montage: " + str(montage) )
    
                                
                            Means = file['GMM Parameters']['Means'][...]
                            Covariance = file['GMM Parameters']['Covariance'][...]
                            Weights = file['GMM Parameters']['Weights'][...]
                            uniques = range(len(Weights))
                            
                            metadata = pd.DataFrame()
                            for meta in montage_file['Metadata'].keys():
                                try:
                                    metadata[meta] = montage_file['Metadata'][meta][...]
                                except ValueError:
                                    pass
                            
                            plot_model(Means, Covariance, Weights, uniques, analysis_file, segmentation)
                            
                    
                            if len(available_backgrounds) > 0: 
                                for i, bk_grd in enumerate(available_backgrounds): 
                                    if use_background[i] == 'y': 
                                        for folder in ['Electron Image', 'EDS', 'EBSD']: 
                                            try:     
                                                keys = list( montage_file[str(folder)].keys() )
                                                for key in keys: 
                                                    if (bk_grd == key): 
                                                        background = montage_file[folder][key][...]
                                                        background = np.array(background, dtype = np.float32 )
                                                        plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, bk_grd, montage_path, metadata, analysis_file)                          
                                            except:
                                                print("GMM Graphing Error")
                                                pass 
                                            
                            else:
                                background = file['Montages'][str(montage)]['Channels'][...]
                                background = np.sum(background, axis = 2 )
                                plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, "Spectral Intensity", montage, metadata, analysis_file)                      
    
    else: 
        available_backgrounds = []                     
        with h5py.File(analysis_file, 'r+') as file: 
            # find and load the segmentation map 
            try:      
                segmentation = file['Montages'][str(montage)]['Segmentation'][...]
                segmentation = np.array(segmentation, dtype = np.uint8 )
            except: 
                raise Exception("Error: Segmentation not found for montage: " + str(montage) )
            
            # Find and load the final uncertainy map 
            try:      
                keys = file['Montages'][str(montage)].keys() 
                keys2 = []
                for key in keys: 
                    if "Uncertainty" in key: 
                        key = key.replace("Uncertainty", "")
                        key = key.replace("Generation", "")
                        key = key.strip()

                        keys2.append( int(key) ) 
                        
                uncertainty = file['Montages'][str(montage)]['Generation ' + str( max( keys2 ) ) + " Uncertainty"][...]
                uncertainty = np.array(uncertainty, dtype = np.float32 )

            except: 
                raise Exception("Error: Uncertainty not found for montage: " + str(montage) )

                
            Means = file['GMM Parameters']['Means'][...]
            Covariance = file['GMM Parameters']['Covariance'][...]
            Weights = file['GMM Parameters']['Weights'][...]
            uniques = range(len(Weights))
            
            plot_model(Means, Covariance, Weights, uniques, analysis_file, segmentation)
            
            if len(available_backgrounds) > 0: 
                for i, bk_grd in enumerate(available_backgrounds): 
                    if use_background[i] == 'y': 
                        for folder in ['Electron Image', 'EDS', 'EBSD']: 
                            try:     
                                keys = list( montage_file[str(folder)].keys() )
                                for key in keys: 
                                    
                                    if (bk_grd == key): 
                                        background = montage_file[folder][key][...]
                                        background = np.array(background, dtype = np.float32 )
                                        plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, bk_grd, montage, None, analysis_file)                          
                            except:
                                pass 
                            
            else:
                background = file['Montages'][str(montage)]['Channels'][...]
                background = np.sum(background, axis = 2 )
                plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, "Spectral Intensity", montage, None, analysis_file)         
                    
        
        
        
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    main()


                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
            



