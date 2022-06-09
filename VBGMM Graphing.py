
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

from pathlib import Path
import h5py
import glob
import hyperspy.api as hs
from matplotlib.ticker import (MultipleLocator)
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

  
def forward_x(x): 
    return (x + x_min)/1000
    
def reverse_x(x): 
    return (x - x_min)/1000

def forward_y(y):
    return (y + y_min)/1000

def reverse_y(y):
    return (y - y_min)/1000

    


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
    




"""

print("Enter the resolution of final outputs in dots per inch (DPI)")
graphing_dpi = int(input(''))
print("")

  
        try :
            with h5py.File(analysis_file) as f:
                headers = list( f['Metadata'].keys() )
                metadata = pd.DataFrame(columns = headers)
                
            for key in headers: 
                metadata[key] = load_h5(analysis_file, 'Metadata/' + str(key) )
            
        except: 
            metadata_marker = False
            for file in file_list:
                if 'Metadata.csv' in file:
                    metadata = pd.read_csv(file)
                    '''
                    metadata = pd.read_csv('Metadata.csv')
                    '''
                    metadata_marker = True
                    break
            if metadata_marker == False:

                

# If an existing analysis file exists, get its file path
# otherwise, create a new analysis H5 file 
if 'Analysis.h5' in file_list_stems: 
    analysis_file = file_list[ file_list_stems == 'Analysis.h5' ]
    #analysis_file = h5py.File(analysis_file) 
else: 
    analysis_file = os.path.join(output_src, 'Analysis.h5')
    save_h5(analysis_file, _, _, _)
    #analysis_file = h5py.File(analysis_file) 



    
    
    print("Autodetected EDS Starting Channel is: " + str(channel_offset) + " eV")
    print("Enter optional replacement value in eV and press ENTER, otherwise leave field blank and press ENTER to keep autodetected value")
    new = input()
    if new != "": 
        channel_offset = float(new)
    print("")
        
    print("Autodetected EDS Channel Width is: " + str(channel_width) + " eV")
    print("Enter optional replacement value in eV and press ENTER, otherwise leave field blank and press ENTER to keep autodetected value")
    new = input()
    if new != "": 
        channel_width = new
    print("")
    
    search_width = 3
    element_search_half_distance = 0.2 # +- distance to search for x-ray band matches from peaks. Units are KeV 




if ('Final_Model_Weights.npy' in file_list_stems) and ('Final_Model_Means.npy' in file_list_stems) and('Final_Model_Covariances.npy' in file_list_stems):
    os.chdir(os.path.join(output_src, 'Training', 'Training Data', 'Analysis' ))
    
    print("Previous VBGMM Model Detected")
    print("Loading model and skipping training")
    covariances = np.load("Final_Model_Covariances.npy")
    means = np.load("Final_Model_Means.npy")
    weights = np.load("Final_Model_Weights.npy")
    precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covariances))

    DPGMM = mixture.GaussianMixture(n_components = len(weights) )   
    DPGMM.weights_ = np.asarray(weights)
    DPGMM.means_ = np.asarray(means)
    DPGMM.covariances_ = np.asarray(covariances)
    precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covariances))
    DPGMM.precisions_cholesky_ = precisions_cholesky     

else:
    
    for w, montage in enumerate(unique_montages): 
        
        
        shape = ( array.shape[0], array.shape[1], len(appended_peaks))
        array = np.array( array, dtype = np.float64)
    
        for ind, loc in enumerate(appended_peaks):  
            image = array[:, :, ind]
            image = np.array(image, dtype = np.float64)
            # Apply chosen preprocessing filter         
            if filter_mode == 1: 
                image = cv2.medianBlur(image,filter_size)
            elif filter_mode == 2:
                image = cv2.GaussianBlur(image, (filter_size,filter_size),0 ) 
            elif filter_mode == 3:
                image = cv2.bilateralFilter(image,filter_size,75,75)
            
            # Add filtered image to dataset 
            array[:,:,ind] = image
            
            
        gc.collect()

    
            
            DPGMM = mixture.GaussianMixture(n_components = len(weights) )
            
            #if generation == 0 
            DPGMM.weights_ = np.asarray(weights)
            DPGMM.means_ = np.asarray(means)
            DPGMM.covariances_ = np.asarray(covar)
            precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covar))
            DPGMM.precisions_cholesky_ = precisions_cholesky 
            
            
            
            
            # Create absolute uncertainty intensity map
            fig, ax = plt.subplots(figsize=(size, size), dpi=300)
            try:
                plt.imshow(background)
                plot = plt.imshow(uncertainty, cmap = color, vmin = autodidectic_threshold, vmax = 0, alpha = 0.7 )
            except NameError:
                plot = plt.imshow(uncertainty, cmap = color, vmin = autodidectic_threshold, vmax = 0 )
                pass 
            plt.xticks([])
            plt.yticks([])
            plt.title(str(montage) + " Fixed Scale Log Liklihoods", fontsize = fontsize)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.ax.tick_params(labelsize=fontsize*(0.7))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(str(montage) + " Fixed Scale Log Liklihoods" + ".png")
            plt.close(fig)
            gc.collect()
            
                # Create absolute uncertainty intensity map
            fig, ax = plt.subplots(figsize=(size, size), dpi=300)
            try:
                plt.imshow(background)
                plot = plt.imshow(uncertainty, cmap = color, vmin = max(-50, autodidectic_threshold), vmax = 0, alpha = 0.7 )
            except NameError:
                plot = plt.imshow(uncertainty, cmap = color, vmin = max(-50, autodidectic_threshold), vmax = 0 )
                pass 
            plt.xticks([])
            plt.yticks([])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(str(montage) + " Fixed Scale Log Liklihoods Without Colorbar" + ".png")
            plt.close(fig)
            gc.collect()
            
            
            # Create relative uncertainty intensity map
            fig, ax = plt.subplots(figsize=(size, size), dpi=300)
            try:
                plt.imshow(background)
                plot = plt.imshow(uncertainty, cmap = color, alpha = 0.7 )
            except NameError:
                plot = plt.imshow(uncertainty, cmap = color )
                pass 
            plt.xticks([])
            plt.yticks([])
            plt.title(str(montage) + " Floating Scale Log Liklihoods", fontsize = fontsize)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.ax.tick_params(labelsize=fontsize*(0.7))
            #plt.title(file)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(str(montage) + " Floating Scale Log Liklihoods with Colorbar" + ".png")
            plt.close(fig)
            gc.collect()
            
            
            # Create relative uncertainty intensity map
            fig, ax = plt.subplots(figsize=(size, size), dpi=300)
            try:
                plt.imshow(background)
                plot = plt.imshow(uncertainty, cmap = color, alpha = 0.7 )
            except NameError:
                plot = plt.imshow(uncertainty, cmap = color )
                pass 
            plt.xticks([])
            plt.yticks([])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(str(montage) + " Floating Scale Log Liklihoods Without Colorbar" + ".png")
            plt.close(fig)
            gc.collect()
            
            
            fig, ax = plt.subplots(figsize=(size, size))
            ax = plt.subplot()
            sns.kdeplot( data = random.sample(list(uncertainty.flatten()), 100_000) , 
                        lw = 6, 
                        log_scale = (False, False) )
            plt.xlabel("Log Liklihood Score")
            plt.ylabel("Relative Frequency")
            plt.title(str(montage) + " Histograms of Log Probabilities")
            plt.tight_layout()
            plt.savefig(str(montage) + " Histograms of Log Liklihood Score" + ".png", dpi=300)
            plt.close(fig)
            gc.collect()
           
            # Save cluster weight graph
            fig, ax = plt.subplots(figsize=(size, size))
            plot_w = np.arange(components) + 1
            ax.bar(plot_w - 0.5, np.sort(DPGMM.weights_)[::-1], width=1., lw=0);
            ax.set_xlim(0.5, components);
            ax.set_xlabel('Dirichlet Process Components');
            ax.set_ylabel('Posterior expected mixture weight');
            fig.suptitle("Mixture Weight per Class" ,fontsize=20 )
            plt.tight_layout()
            plt.savefig("Mixture Weights" + ".png", dpi=300)
            plt.close(fig)
            gc.collect()
            
            
        check = True 
        for file in file_list: 
            if ("['EDS', 'XRay']" in file) and (montage in file):        
                check = False
                array = np.load(file)
                gc.collect()
                break 
            
        if check: 
            try: 
                os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            except FileExistsError:
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
                   
            analysis_ev = np.load('appended_peaks.npy')
            sum_spectrum = np.load('sum_spectrum.npy')
            highest_spectrum = np.load('highest_spectrum.npy')        
            
            # Stitch and save xray datacubes 
            print("Stitching datacube for: " + str(montage) )
            eds_binning_width = 3
            array = microscopy_parsers.oxford_montage_stitcher(montage, metadata, ['EDS', 'XRay'], file_list, appended_peaks, eds_binning_width)
            
            np.save("['EDS', 'XRay']", array, allow_pickle = False)
            gc.collect()
    
        print("Total number of xray peaks detected in " + str(montage) + ": " + str( len(appended_peaks) ) )
    
        fig, ax = plt.subplots(1, 1, dpi = 200)
        plt.plot(sum_spectrum)
        plt.scatter( x = appended_peaks, y = np.zeros( len(appended_peaks) ) , c = 'green')
        plt.title( montage )
        plt.savefig("Sum of Spectrum.png") 
        
        fig, ax = plt.subplots(1, 1, dpi = 200)
        plt.plot(highest_spectrum)
        plt.scatter( x = appended_peaks, y = np.zeros( len(appended_peaks) ) , c = 'green')
        plt.title( montage )
        plt.savefig("Highest Count Spectrum.png")      
        
        
        
        
        
        
        
    auto_suggest_filters = ['Ka']
    element_filters = [] 
    distance = 3
    
    for montage in unique_montages:
        
    
        try: 
            os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
        except FileExistsError:
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            
        try: 
            appended_peaks = np.load("appended_peaks.npy")
            appended_peaks = np.array(appended_peaks.flatten())
            sum_spectrum = np.load("sum_spectrum.npy")
            sum_spectrum = np.array(sum_spectrum.flatten())
            highest_spectrum = np.load("highest_spectrum.npy")
            highest_spectrum = np.array(highest_spectrum.flatten())
            
            temp = list( appended_peaks.copy())
        except: 
            pass 
            
            
        for file in file_list: 
            if ("display_shells" in file) and (montage in file):
                display_shells = np.load(file)
            
            else: 
                while True:      
                        
                    if len(temp) == 0:
                        break 
                    
                    
                    for i, peak in enumerate(temp):  
                        while True: 
                            
                            new_filters = input("Change auto suggest filters: ").split(",")
                            
                            for command in new_filters: 
                                if command.startswith('+'): 
                                    auto_suggest_filters.append(command.split('+')[1] )
                                    
                                elif command.startswith('-'): 
                                    try: 
                                        auto_suggest_filters.remove(command.split('-')[1] )
                                    except ValueError:
                                        print( str( command.split('-')[1] ) + "not found in auto suggest filters")
                                        
                                elif command == '': 
                                    pass
                                else: 
                                    print("Error: " + str(command) + " - commands must begin with + or -")
                                 
                            max_x = peak
                            fig, ( ax2, ax3, ax4) = plt.subplots(3, 1, figsize = (30,30 ) , dpi = 200)
                            fig.suptitle('Potential Peak at ' + str( round( (channel_offset + channel_width*peak)/1_000.0, 2 )) + "KeV")
                           
                            plt.sca(ax2)
                            plt.scatter(x = (channel_offset + channel_width*np.asarray(temp))/1_000.0, y = highest_spectrum[temp], s = 400)
                
                            plt.scatter(x = (channel_offset + channel_width*max_x)/1_000.0, y = highest_spectrum[ max_x ], s = 500, color = 'red')
                            plt.plot((channel_offset + channel_width*np.linspace(0,1024, num=1024))/1_000.0, highest_spectrum, color = "Green")
                            plt.title("Sum of Spectrum of Entire Dataset - Suggested Peaks")
                            plt.tight_layout()
                            plt.xlabel("KeV")
                            plt.ylabel("X-Ray Counts")
                            
                            ax2.tick_params(which = 'major', length = 25, bottom = True, left = True)
                            ax2.tick_params(which = 'minor', axis = 'x', length = 15, bottom = True, left = True)
                            
                            ax2.minorticks_on()
                            
                            ax2.xaxis.set_major_locator(MultipleLocator(5))
                            ax2.xaxis.set_minor_locator(tck.AutoMinorLocator())
                                 
                            
                            
                            plt.sca(ax3)
                            plt.scatter(x = (channel_offset + channel_width*np.asarray(temp))/1_000.0, y = sum_spectrum[temp], s = 400)
                
                            plt.scatter(x = (channel_offset + channel_width*max_x)/1_000.0, y = sum_spectrum[ max_x ], s = 500, color = 'red')
                            plt.plot((channel_offset + channel_width*np.linspace(0,1024, num=1024))/1_000.0, sum_spectrum, color = "Green")
             
                         
                            plt.title("Sum of Spectrum of Entire Dataset - Suggested Peaks")
                            plt.tight_layout()
                            plt.xlabel("KeV")
                            plt.ylabel("X-Ray Counts")
                            
                            ax3.tick_params(which = 'major', length = 25, bottom = True, left = True)
                            ax3.tick_params(which = 'minor', axis = 'x', length = 15, bottom = True, left = True)
                            
                            ax3.minorticks_on()
                            
                            ax3.xaxis.set_major_locator(MultipleLocator(5))
                            ax3.xaxis.set_minor_locator(tck.AutoMinorLocator())
                            
                            auto_suggest_elements = hs.eds.get_xray_lines_near_energy((channel_offset + channel_width*max_x)/1_000.0, only_lines = auto_suggest_filters, width = 0.5)
                            print(auto_suggest_elements)
                            for z, element in enumerate(auto_suggest_elements):     
                                ax3.annotate(str(element ),
                                        xy = ( 0.9, (len(auto_suggest_elements) + 1 - z )/ ( len(auto_suggest_elements) + 1 ) - 0.02),
                                        xycoords='data',
                                        #xytext=( 15, max(group)*(len(elements) - z)), 
                                        textcoords='axes fraction',
                                        horizontalalignment='left', 
                                        verticalalignment='top')
                                
                          
                    
                            plt.sca(ax4)
                            plt.scatter(x = (channel_offset + channel_width*max_x)/1_000.0, y = sum_spectrum[ max_x ], s = 500, color = 'red')
                            plt.plot((channel_offset + channel_width*np.linspace(0,1024, num=1024))/1_000.0, sum_spectrum, color = "Green")
                            plt.title("Sum of Spectrum of Entire Dataset - User Specified Peaks")
                            
                            for element in element_filters:
                                print(str(element) ) 
                                keys = eval ('hs.material.elements.' + str(element) + '.Atomic_properties.Xray_lines.keys()' ) 
                            
                                for key in keys:
                                    
                                    try: 
                                        #print( hs.material.elements.Fe.Atomic_properties.Xray_lines.get_item(keys[0]).get_item('energy (keV)') )
                                        print("   " + str(key))
                                        print( "   " + str(eval( "hs.material.elements." + str(element) + ".Atomic_properties.Xray_lines.get_item('" + key + "').get_item('energy (keV)')" )))
                                        eV = eval( "hs.material.elements." + str(element) + ".Atomic_properties.Xray_lines.get_item('" + key + "').get_item('energy (keV)')" )
                                        eV2 = round( ( eV*1000.0 - channel_offset)/ channel_width)
                                        y = sum_spectrum[eV2]
                                        ax4.annotate( str(element) + "_" + str(key) , xy=(eV, y + 0.25*np.max(sum_spectrum)),  
                                                     xycoords='data',
                                                     textcoords = 'data',
                                                     horizontalalignment='right', 
                                                     verticalalignment='top'
                                                     )
                                        plt.bar(x = eV, 
                                                height = y,
                                                width = 0.075,
                                                color = 'red')
                            
                                    except IndexError: 
                                        pass 
                                
                            plt.tight_layout()
                            plt.xlabel("KeV")
                            plt.ylabel("X-Ray Counts")
                            
                            ax4.tick_params(which = 'major', length = 25, bottom = True, left = True)
                            ax4.tick_params(which = 'minor', axis = 'x', length = 15, bottom = True, left = True)
                            
                            ax4.minorticks_on()
                            
                            ax4.xaxis.set_major_locator(MultipleLocator(5))
                            ax4.xaxis.set_minor_locator(tck.AutoMinorLocator())
                            plt.show()
                            
                            
                            print("Existing element list:")
                            print(str(element_filters))
                            print("")
                            print("Add and/or remove one or more elements (e.g. +Mn, -Al, +Cr). Seperate elements with commas")
                            print("Enter 'z' to remove peak from review list")
                            print("Enter 'end' to continue to next peak but not remove from review list")
                            new_command = input("")
                            
                            if new_command == 'end': 
                                print("Continuing to next peak")
                                break 
                            elif new_command == 'z': 
                                temp.remove(peak)
                                break
                            else: 
                                new_command = new_command.split(',')
                                for command in new_command: 
                                    
                                    try: 
                                        command = command.replace(" ", "")
                                    except:
                                        pass 
                                    
                                    if command.startswith('+'): 
                                        try:
                                            element_filters.append(command.replace('+', ''))
                                        except ValueError: 
                                            print(str(command) + " not found in list of elements")
                                        
                                    elif command.startswith('-'): 
                                        try:
                                            element_filters.remove(command.replace('-', ''))
                                        except ValueError: 
                                            print(str(command) + " not found in list of elements")
                                            
                                    else: 
                                        print("Error: " + str(command) + " - commands must begin with + or -")
                                
                            print( "Specified Peaks: ")
                            
                            
                            for element in element_filters:
                                print("   " + str(element) ) 
                            
                 
                    
             
            
     
    
    element_filters.sort()
    energy_locs = list((channel_offset + channel_width*np.asarray(appended_peaks))/1_000.0)
    
    element_energies = []
    element_shells = []
    display_shells = ["" for n in range(len(appended_peaks) )]
    
    for element in element_filters:
        keys = eval ('hs.material.elements.' + str(element) + '.Atomic_properties.Xray_lines.keys()' ) 
    
        for key in keys:
            energy = eval( "hs.material.elements." + str(element) + ".Atomic_properties.Xray_lines.get_item('" + key + "').get_item('energy (keV)')" )
            element_energies.append(energy) 
            element_shells.append(str(element) + "_" + str(key))
            
            
    for az, energy in enumerate(energy_locs): 
        
        diff = np.abs( np.asarray(element_energies) - energy) 
        loc = np.argmin(diff)
        
        
        
        if (np.min(diff) < 0.05) and (display_shells[az] == "") :
            
            display_shells[az] = element_shells[loc] 
        
        elif (np.min(diff) < 0.05) and (display_shells[az] != ""): 
            old_value = ( np.abs(energy - element_energies[loc]) > np.abs(energy - element_energies[ np.argwhere( np.asarray(element_shells) == display_shells[az] ) ] ))
            
            if old_value > np.minimum(diff):
                display_shells[az] = element_shells[loc] 
    
    
    order = np.argsort(np.asarray(display_shells ))
    display_shells.sort()
    temp_appended_peaks = [] 
    
    
    for value in order: 
        
    
    
    
    test = appended_peaks.copy()
    
    test = list(test.flatten()) 
    test.sort(key = list(order.flatten()) )
    
    (key = order)
    
    sorted(test , key = order.flatten() )
    
    
    
        
    for montage in unique_montages:
        
        try: 
                os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
        except FileExistsError:
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
       
        np.save("appended_peaks", appended_peaks, allow_pickle = False)
        np.save("display_shells", display_shells, allow_pickle = False)
        gc.collect()
        
            
    # If the same greyscale backgrounds are available for all montages, input from user which to use 
    if len(available_specifications) > 0:
        
        print("Select the background grayscale to use:")
        for i, spec in enumerate(available_specifications):
            print(str(i) + " " + str(spec)) 
        background = int(input("Enter the number of the background to use: "))
        background = available_specifications[background]
        background = np.load( (os.path.join(output_src, 'Data', 'Montages', str(montage), str(background) + '.npy' )))
    
    else:
        print("No single grayscale background image available for all montages")
        print("Using X-Ray count intensities as artificial grayscale background instead")
        background = None
    
    #display_shells = np.load('display_shells.npy')
    
    
    energy_locs = list((channel_offset + channel_width*np.asarray(appended_peaks))/1_000.0)
    
    temp = []
    for i in range(len(display_shells)):
        if str(display_shells[i]) != '':
            temp.append(str(display_shells[i]) + ": " + str( round(energy_locs[i], 3) ) + "KeV")
        else: 
            temp.append( str( round(energy_locs[i], 3) ) + "KeV")
            
    display_shells = temp.copy()
    
    
    
    
    
"""

def plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, bk_grd, montage_path): 
    size = 40
    graphing_dpi = 400
    color = 'seismic_r'
    fontsize = 60
    ratio = 2 
    background = np.array(background, dtype = np.float32)
    try: 
        montage = Path(montage_path).stem.replace("Montage ", "")
    except NameError:
        montage = "Unlabeled"
        
        
    # Create absolute uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    try:
        plt.imshow(background)
        plot = plt.imshow(uncertainty, cmap = color, vmin = -20, vmax = 0, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = color, vmin = -20, vmax = 0 )
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.title(str(montage) + " Fixed Scale Log Liklihoods", fontsize = fontsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(str(montage) + " Fixed Scale Log Liklihoods " + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
    
    
    # Create absolute uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    try:
        plt.imshow(background)
        plot = plt.imshow(uncertainty, cmap = color, vmin = -20, vmax = 0, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = color, vmin = -20, vmax = 0 )
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(str(montage) + " Fixed Scale Log Liklihoods Without Colorbar" + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
    
    
    # Create relative uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    try:
        plt.imshow(background)
        plot = plt.imshow(uncertainty, cmap = color, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = color )
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.title(str(montage) + " Floating Scale Log Liklihoods", fontsize = fontsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    #plt.title(file)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods with Colorbar" + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
    
    
    # Create relative uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
    try:
        plt.imshow(background)
        plot = plt.imshow(uncertainty, cmap = color, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = color )
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods Without Colorbar" + str(bk_grd) + ".png")
    plt.close(fig)
    gc.collect()
            
    # Save semantic segmentation map 
    fig, ax = plt.subplots(figsize = (size, size)) 
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
    plt.title(str(montage) + " Class Segmentation", fontsize = fontsize*(0.7))
    plt.grid(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax, ticks=np.arange(np.min(segmentation),np.max(segmentation)+1))
    cbar.ax.tick_params(labelsize=25) 
    plt.tight_layout()
    plt.savefig(str(montage) + " Class Segmentation with Colorbar "  + str(bk_grd) + ".png", dpi=graphing_dpi)
    plt.close(fig)
    gc.collect()
    
    # Save semantic segmentation map 
    fig, ax = plt.subplots(figsize = (size, size)) 
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
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(str(montage) + " Class Segmentation Without Colorbar "  + str(bk_grd) + ".png", dpi=graphing_dpi)
    plt.close(fig)
    gc.collect()


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
      
    except: 
        pass 
          
    
    
    
    
    
    """
    print("Number of unique classes: " + str(len(list(np.unique(segmentation)))))
    print(" ")
    print("Plotting Complete")
    print(" ")
    print("Results located in ")
    print(" ")
    print(str(output_src) ) 
    """ 
        
        
        
        
        

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
              
        ratio = 2
        size = 10
        fontsize = 30
        fig = plt.figure(constrained_layout=False, figsize=(ratio*size, size), dpi=graphing_dpi)
        gs = fig.add_gridspec( ncols = 6, nrows = 3)
        
        ax1 = fig.add_subplot(gs[0:3, 0:3])
        ax1.set_title('Class Map', fontsize = 15)
        
        plt.sca(ax1)
        plt.tick_params(left = True)
        plt.tick_params(bottom = True)
        
        plot = plt.imshow( dilation, cmap = 'Blues', alpha = 1)     
        plot = plt.imshow( class_map, cmap = 'brg', alpha = 1)  
        
        plt.yticks([])
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(plot, cax=cax)   
        cb.ax.tick_params(labelsize=fontsize) 
        cb.remove()
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.figure.axes[-1].yaxis.label.set_size(10)    # colorbar label size 
        ax1.figure.axes[-1].xaxis.label.set_size(10)    # colorbar label size 
        plt.xlabel("Scale (um)", fontsize = 12)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        ax1.set_aspect(aspect = segmentation.shape[0]/segmentation.shape[1], adjustable = 'box')
        
        """
        try: 
            # add secondary axis labels for the stage locations             
            secondx_ax1 = ax1.secondary_xaxis('top', functions = (forward_x, reverse_x))
            secondx_ax1.set_xlabel('X Stage Location (mm)', fontsize = 12)
            secondx_ax1.tick_params(labelsize=10)
                        
            secondy_ax1 = ax1.secondary_yaxis('left',  functions = (forward_y, reverse_y))
            secondy_ax1.set_ylabel('Y Stage Location (mm)', fontsize = 12)
            secondy_ax1.tick_params(labelsize=10)
        except NameError: 
            pass 
         
        """
        
        try:
            plt.sca(ax1)
            plot = plt.imshow(background, alpha = 0.3, cmap = 'gray_r')
        except NameError:
            pass 
        
        
        ax2 = fig.add_subplot(gs[0, 3:5])
        ax2.set_title('Correlation Matrix', fontsize = 15)
        
        
        plt.sca(ax2)
        stdev = np.sqrt(np.abs(Covariance[ind2]))
        stdev[Covariance[ind2] < 0.0] = -1 * stdev[Covariance[ind2] < 0.0] 
        annot = np.diag(precisions_cholesky[ind2],0)
        annot = np.round(annot,2)
        annot = annot.astype('str')
        annot[annot=='0.0']=''
        try:     
            sns.heatmap(correlation_from_covariance(Covariance[ind2]), xticklabels = display_shells, yticklabels = display_shells, center = 0, vmin = 0, vmax = 1, linewidths=1, linecolor = 'white', cmap = 'bwr', mask = np.triu(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'vertical'})
        except NameError: 
            sns.heatmap(correlation_from_covariance(Covariance[ind2]), center = 0, vmin = 0, vmax = 1, linewidths=1, linecolor = 'white', cmap = 'bwr', mask = np.triu(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'vertical'})

        
        
        ax2.tick_params(axis='x', pad=5)
        ax2.set_yticklabels( ax2.get_yticklabels(), rotation=0)
        ax2.set_xticklabels( ax2.get_xticklabels(), rotation=90)
        
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(6)
            
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)               # colorbar tick size 
        ax2.figure.axes[-1].yaxis.label.set_size(10)    # colorbar label size 
        ax2.figure.axes[-1].xaxis.label.set_size(10)    # colorbar label size 
        plt.xlabel("Electron Shell", fontsize = 12)
        plt.ylabel("Electron Shell", fontsize = 12)

                
        plt.tight_layout()
       
        
        plt.savefig(str(montage) + " Class " + str(ind2) + " With Correlation.png")
        gc.collect()








def plot_model(Means, Covariance, Weights, uniques, analysis_file): 
    size = 40
    graphing_dpi = 400
    color = 'seismic_r'
    fontsize = 60
    ratio = 2 
   
    analysis = Path(analysis_file).stem
    components = len(Weights)

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
    """
    index_to_delete = []
    for row in range(len(Weights)):
        if (row in uniques) == False:
            index_to_delete.append(row)
            
    weights = np.delete(Weights, np.array(index_to_delete, dtype = np.int16), 0)
    """
    linkage_matrix = linkage_matrix.sort_values(by=['Class ID'])
    linkage_matrix['Area Percent'] = Weights * 100.0
    linkage_matrix.to_excel(str(analysis) + " Class Data.xlsx", index = False)
    
    # Save cluster weight graph
    fig, ax = plt.subplots(figsize=(size, size))
    plot_w = np.arange(components) + 1
    ax.bar(plot_w - 0.5, np.sort(Weights)[::-1], width=1., lw=0);
    ax.set_xlim(0.5, components);
    ax.set_xlabel('Dirichlet Process Components');
    ax.set_ylabel('Posterior expected mixture weight');
    fig.suptitle("Mixture Weight per Class" ,fontsize=20 )
    plt.tight_layout()
    plt.savefig( str(analysis) + "Mixture Weights" + ".png", dpi=graphing_dpi)
    plt.close(fig)
    gc.collect()
                           

def main():
    # Get user to select analysis file 
    analysis_path = filedialog.askopenfilename( title = "Select input analysis file")
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
                            
                            plot_model(Means, Covariance, Weights, uniques, analysis_file)
                            
                    
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
                                                        plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, bk_grd, montage)                          
                                            except:
                                                pass 
                                            
                            else:
                                background = file['Montages'][str(montage)]['Channels'][...]
                                background = np.sum(background, axis = 2 )
                                plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, "Spectral Intensity", montage)                      
    
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
            
            plot_model(Means, Covariance, Weights, uniques, analysis_file)
            
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
                                        plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, bk_grd, montage)                          
                            except:
                                pass 
                            
            else:
                background = file['Montages'][str(montage)]['Channels'][...]
                background = np.sum(background, axis = 2 )
                plot_GMM(Means, Covariance, Weights, segmentation, uncertainty, background, "Spectral Intensity", montage)         
                    
        
        
        
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    main()


                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
            



