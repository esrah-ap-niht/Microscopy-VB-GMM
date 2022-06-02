
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
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile as io
#import sklearn
from sklearn import mixture
#import scipy
import seaborn as sns
#from scipy.cluster.hierarchy import dendrogram
#import math

from pathlib import Path
import h5py
import glob
#import hyperspy.api as hs
#from matplotlib.ticker import (MultipleLocator)
#import matplotlib.ticker as tck
#import matplotlib as mpl
from tqdm import tqdm
import collections 

# BNPY is not available via pip or conda, therefore users must place it in the C drive for consistency 
try: 
    import bnpy
except ModuleNotFoundError:
    import sys
    sys.path.append(str(os.path.join('C:', os.sep, 'Program Files', 'bnpy' )) )
    import bnpy

##############################################################################################################
# plotting, garbage collection, and tkinter settings 
##############################################################################################################
gc.enable()
#plt.ioff()
root = Tk()
root.withdraw()
root.attributes('-topmost',1)
#sns.set(font_scale = 3)
#sns.set_style("white")

##############################################################################################################
# Define functions needed later 
##############################################################################################################

def load_data( file_list, filter_mode, filter_size ): 
    # The GMM can incorporate quantified elemental composition maps, xray intensity maps (e.g. Mn Ka), and xray data cubes (x pixel, y pixel, z spectra)
    # This function is a parser that loads a series of 2D inputs and layers them into a 3D array. 
    # Individual layers are assumed to already be stitched into a layerwise montage and spatially superimposed 
    
    for i, file in enumerate(file_list):
        
        # Create data array 
        if i == 0: 
            if file.endswith('.tif') or file.endswith('.tiff'):
                image = io.imread(file)
            elif file.endswith('.jpg') or file.endswith('.png'):
                image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            elif file.endswith('.csv'):
                image = pd.read_csv(file, sep=',',header=None)
                image = np.asarray(image)
            elif file.endswith('.xlsx'):
                image = pd.read_excel(file, index_col=None, header=None)
                image = np.asarray(image)
            elif file.endswith('.npy'): 
                image = np.load(file)
                image = np.asarray(image)
    
            if len(image.shape) > 2: 
                raise ValueError("ERROR: Input 2D layers not detected as 2D. Input '" + str(file) + "' has shape " + str(shape) )
            shape = ( image.shape[0], image.shape[1], len(file_list))
            array = np.zeros(shape, dtype = np.float32)
                    
        # Read file 
        if file.endswith('.tif') or file.endswith('.tiff'):
            image = io.imread(file)
            if len(image.shape) == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif file.endswith('.jpg') or file.endswith('.png'):
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        elif file.endswith('.csv'):
            image = pd.read_csv(file, sep=',',header=None)
            image = np.asarray(image)
        elif file.endswith('.xlsx'):
            image = pd.read_excel(file, index_col=None, header=None)
            image = np.asarray(image)
        elif file.endswith('.npy'): 
            image = np.load(file)
            image = np.asarray(image)
        
        # verify that the layer is 2D 
        if len(image.shape) > 2: 
            raise ValueError("ERROR: Input 2D layers not detected as 2D. Input '" + str(file) + "' has shape " + str(shape) )
            
        image = np.array(image) 
      
        # Apply chosen preprocessing filter         
        if filter_mode == 1: 
            image = cv2.medianBlur(image,filter_size)
        elif filter_mode == 2:
            image = cv2.GaussianBlur(image, (filter_size,filter_size),0 ) 
        elif filter_mode == 3:
            image = cv2.bilateralFilter(image,filter_size,75,75)
        
        # Add filtered image to dataset 
        array[:,:, i] = image
        
    # Reduce the precision of the array to the lowest acceptable level in order to reduce memory requirements
    array = np.array(array, dtype = np.float16)
    
    return array 


# Calculate the per-pixel uncertainty. Output used to select what data is appended to the training dataset 
def calc_uncertainty( data, weights, means, covar):
    
    # Use sklearn's module as it is reasonably optimized
    DPGMM = mixture.GaussianMixture(n_components = len(weights) )
    
    DPGMM.weights_ = np.asarray(weights)
    DPGMM.means_ = np.asarray(means)
    DPGMM.covariances_ = np.asarray(covar)
    precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covar))
    DPGMM.precisions_cholesky_ = precisions_cholesky 
    
    # If the dataset is too large, break it into mini-chunks to reduce memory requirements 
    print('Calculating Uncertainty')
    if data.shape[0] > 10_000:
        for point in tqdm(range(100)): 
            #print("Uncertainty Calculation Progress: " + str( round(point / 10_000 * 100, 2) ) )
            start = round( (data.shape[0] / 100) * point )
            end   = round( (data.shape[0] / 100) * (point +1 ) )
            calculation = DPGMM.score_samples( data[start:end, :] )  
            try: 
                uncertainty = np.concatenate( (uncertainty, calculation), axis = 0)
            except NameError: 
                uncertainty = calculation.copy() 
    else: 
        #print("Processing Uncertainty Calculation as Single Batch")
        uncertainty = DPGMM.score_samples( data )  
    return uncertainty

    
# Segment data per-pixel. 
def calc_segmentation( data, weights, means, covar):
        
    DPGMM = mixture.GaussianMixture(n_components = len(weights) )
    
    DPGMM.weights_ = np.asarray(weights)
    DPGMM.means_ = np.asarray(means)
    DPGMM.covariances_ = np.asarray(covar)
    precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covar))
    DPGMM.precisions_cholesky_ = precisions_cholesky 
    
    print("Segmenting Data")
    if data.shape[0] > 10_000:
        for point in tqdm(range(100)): 
            #print("Segmentation Calculation Progress: " + str( round(point / 10_000 * 100, 2) ) )
            start = round( (data.shape[0] / 100) * point )
            end   = round( (data.shape[0] / 100) * (point +1 ) )
            calculation = DPGMM.predict( data[start:end, :] )  
            try: 
                segmentation = np.concatenate( (segmentation, calculation), axis = 0)
            except NameError: 
                segmentation = calculation.copy() 
    else: 
        print("Processing Segmentation Calculation as Single Batch")
        segmentation = DPGMM.predict( data )  
    return segmentation

# Train the model given a training dataset. BNPY requires a directory location to store a training log
def fit_vbgmm( training_data, output_path): 
    dataset = bnpy.data.XData(training_data)
    
    merge_kwargs = dict(
        m_startLap=4,
        m_pair_ranking_procedure='total_size',
        m_pair_ranking_direction='descending',
        )

    delete_kwargs = dict(
        d_startLap=4,
        d_nRefineSteps=30,
        )

    birth_kwargs = dict(
        b_startLap=4,
        #b_stopLap=20,
        b_Kfresh=5)

    full_trained_model, full_info_dict = bnpy.run(
        dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
        output_path=(str(output_path) ),
        nLap=200, 
        nTask=1, 
        nBatch=10,
        gamma0=3.0, 
        sF=1.0, 
        ECovMat='eye',
        K=10, 
        initname='bregmankmeans',
        moves='birth,merge,delete,shuffle',
        **dict(list(delete_kwargs.items()) + list(merge_kwargs.items()) + list(birth_kwargs.items()) ))

    #loss = full_info_dict['lap_history']
    
    K = full_trained_model.allocModel.K # number of classes found by fitting training data 
    
    means = []
    covar = []
    weights = []

    for k in range(K):
        means.append(full_trained_model.obsModel.get_mean_for_comp(k) ) # means 
        covar.append(full_trained_model.obsModel.get_covar_mat_for_comp(k) ) # covar matrix
        weights.append(full_trained_model.allocModel.get_active_comp_probs()[k] ) # weights 

    #SS = full_trained_model.get_global_suff_stats(dataset, full_trained_model.calc_local_params(dataset), doPrecompEntropy=1)

    #Hresp = SS.getELBOTerm('Hresp') # ELBO calculation per cluster 


    return np.asarray(weights), np.asarray(means), np.asarray(covar), full_info_dict

    
def save_h5(h5_file, h5_path, array_or_attribute, data): 
    try: 
        file = h5py.File(h5_file, 'a')
    except: 
        file = h5py.File(h5_file, 'r+') 

    try:
        file.create_group(h5_path)
    except:
        pass 
    
    try: 
        if array_or_attribute[0] == 'array': 
            try:
                file[h5_path].create_dataset(array_or_attribute[1], data = data, chunks=True)
            except: 
                file[h5_path + '/' + str(array_or_attribute[1])] = data
            
        elif array_or_attribute[0] == 'attrs':
            try: 
                file[h5_path].attrs.create(array_or_attribute[1], data)
            except: 
                file[h5_path].attrs[array_or_attribute[1]] = data
                
    except: 
        pass
   
    file.close() 
    return 


def load_h5 (h5_file, h5_path):
    file = h5py.File(h5_file, 'r+')
    data = file[h5_path][...]
    file.close()
    return data
   
  # # h5_file = montage_file
  #  h5_path = 'Metadata/EDS Starting Bin Voltage (eV)'


#h5_file = analysis_file
#h5_path = 'Metadata/Data'


def main():
    ##############################################################################################################
    # Start of program 
    ##############################################################################################################
    global full_data
    global DPGMM
    global initial_weights
    autodidactic_loop = True 
    search_width = 1 # units are in eV  
    size = 40
    fontsize = 60
    ratio = 2
    color = 'seismic_r'
    sns.set(font_scale = 3)
    sns.set_style("white")
    
    # startup message to user 
    print("VBGMM may be run in one of two analysis modes")
    print("'Fast' analysis inputs a series of 2D images for each montage")
    print("Each montage must have the exact same image names and number of images")
    print("Use 'fast' when you already know what elements to look for")
    print("Suggested data types are Ka intensity, at% quant maps, and wt% quant maps")
    print("")
    print("'Unbiased' analysis inputs preprocessed H5 files from the microscopy parser software")
    print("Xray peaks will be automatically detected and phases segmented without human specification")
    print("The user will then be prompted to label the auto-identified peaks")
    print("")
    print("Enter 1 for 'fast' or 2 for 'unbiased' ")
    
    # get analysis type input 
    analysis_type =  int( input("") )
    
    print("")
    print("Enter the name of this analysis")
    # get analysis name input 
    analysis_name =  str( input("") )
    
    # verify that the input type is 1 or 2 
    if (analysis_type != 1) and (analysis_type != 2): 
        raise Exception("Error: analysis mode must be entered as '1' or '2' ")

    # Get user to select output directory 
    output_src = filedialog.askdirectory( title = "Select output directory")
    analysis_file = os.path.join(output_src, 'Analysis ' + analysis_name + '.h5')
    unique_montages = [] 
        
    ########
    # get AI parameters from user 
    print("You will be prompted to enter a series of AI analysis parameters.")
    print("Leave fields blank and press 'ENTER' for each prompt to use default values. Otherwise enter appropriace values and then press 'ENTER' for each parameter")
    print("")
    print("Select the preprocessing filter to use. 0 for no filter (use raw quantified data), 1 for median filter, 2 for Gaussian filter, 3 for bilateral filter. Recommend Gaussian")
    filter_mode = input("")
    if filter_mode == '': 
        filter_mode = 2
    else: 
        filter_mode = int(filter_mode)
        
    if filter_mode != 0:
        print("Enter the filter pixel size in odd positive integers greater than 1. Higher values result in smoothing over a larger area. Recommend 5 ")
        filter_size =  input("")
    if filter_size == '': 
        filter_size = 5
    else: 
        filter_size = int(filter_size)
        
    print("Enter integer number of initial training datapoints. Recommend 30_000")
    initial_sampling_size = input("")
    if initial_sampling_size == '': 
        initial_sampling_size = 30_000
    else: 
        initial_sampling_size = int(initial_sampling_size)
        
    print("Enter training dataset growth rate. Recommend 10_000")
    training_set_growth_rate = input("")
    if training_set_growth_rate == '': 
        training_set_growth_rate = 10_000
    else: 
        training_set_growth_rate = int(training_set_growth_rate)
        
    print("Enter Log Probability threshold for autodidactic loop. Set to -1000 to disable. Recommend -20")
    autodidectic_threshold = input("")
    if autodidectic_threshold == '': 
        autodidectic_threshold = -20
    else: 
        autodidectic_threshold = int(autodidectic_threshold)
        
    print("Enter the maximum integer number of training loops to execute. Recommend 3")
    max_autodidectic_iterations = input("")
    if max_autodidectic_iterations == '': 
        max_autodidectic_iterations = 3
    else: 
        max_autodidectic_iterations = int(max_autodidectic_iterations)
        
    print("Enter the cutoff number of un-identified pixels to end training for each montage. When fewer than this number of pixels remain un-identified, the model will end training for that montage")
    residual_pixel_threshold = input("")
    if residual_pixel_threshold == '': 
        residual_pixel_threshold = 50
    else: 
        residual_pixel_threshold = int(residual_pixel_threshold)
        
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'filter_mode'], np.array(filter_mode ).astype('S') )
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'filter_size'], np.array(filter_size ).astype('S') )
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'initial_sampling_size'], np.array(initial_sampling_size ).astype('S') )
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'training_set_growth_rate'], np.array(training_set_growth_rate ).astype('S') )
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'autodidectic_threshold'], np.array(autodidectic_threshold ).astype('S') )
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'max_autodidectic_iterations'], np.array(max_autodidectic_iterations ).astype('S') )
    save_h5(analysis_file, 'AI Parameters', ['attrs', 'residual_pixel_threshold'], np.array(residual_pixel_threshold ).astype('S') )
    
    ########
    if analysis_type == 1: 
        # Get user to select directories 
        input_src = []
        while True: 
            get_path = filedialog.askdirectory( title = "Select input directory(s). Files will be copied to the output directory. Press 'cancel' to stop adding folders")
            if get_path != '':
                input_src.append(get_path)
            else:
                break 
         
        stems_a = []
        
        for i, src in enumerate( tqdm(input_src, total = len(input_src), desc = 'Checking input directories' )): 
            stems_b = [] 
            for file in glob.glob(src + '/**/*', recursive=True):
                if Path(file).suffix in ['.png', '.jpg', '.tif', '.tiff']:
                    if i == 0:
                        stems_a.append(os.path.basename(file))
                    else: 
                        stems_b.append(os.path.basename(file))
            if (collections.Counter(stems_a) != collections.Counter(stems_b)) and (len(stems_b) > 0):
                raise Exception("Error: Files in all input folders must be identical. Found: " + str(stems_a) + " and " + str(stems_b) )
        print("Files in input directories are correctly labeled")
        print("")
        
        for i, src in enumerate( input_src ): 
            montage = os.path.split( os.path.dirname(file) )[1]
            print("Enter the um/pixel resolution for: " + str(montage) + "if available. Otherwise leave blank and press 'enter' ")
            print("")
            resolution = input("")
            if resolution == '': 
                pass
            else: 
                resolution = float(resolution)
                
            save_h5(analysis_file, 'Montages/' + str(montage), ['attrs', 'resolution'], np.array(resolution ) )
            
            
        for i, src in enumerate( tqdm(input_src, total = len(input_src), desc = "Creating analysis file: " + str(analysis_name) )): 
            montage = os.path.split( os.path.dirname(file) )[1]
            unique_montages.append(montage)
            file_list = []

            for file in glob.glob(src + '/**/*', recursive=True):
                if Path(file).suffix in ['.png', '.jpg', '.tif', '.tiff']:
                    file_list.append(file)        
                    
            data = load_data( file_list, filter_mode, filter_size )
            
            save_h5(analysis_file, 'Montages/' + str(montage), ['array', 'Channels'], data )
        
    ########
    elif analysis_type == 2: 
        # Get user to select input files  
        file_list = filedialog.askopenfilenames ( title = "Select input directory(s). Files will be copied to the output directory. Press 'cancel' to stop adding folders")

        # Get user to select output directory 
        #output_src = filedialog.askdirectory( title = "Select output directory")
        #analysis_file = os.path.join(output_src, 'Analysis ' + analysis_name + '.h5')       
        
    
        for file in file_list:
            if Path(file).suffix in ['.h5']:
                unique_montages.append(Path(file).stem.replace("Montage ","") )
            else: 
                raise Exception("Error: 'Unbiased' analysis only compatible with preprocessed H5 files")

        # Load autodetected peaks from all montages, convert from bins to eV
        # then compare peaks to remove duplicates and compile a list of all peaks 
        # found from the selected montages.
        # New peaks must be at least 'search_width' eV apart to be added. 
        analysis_kev = [] 
        for z, montage in enumerate(unique_montages):
           
            montage_file = os.path.abspath( file_list[z] )
            
            channel_offset = np.mean( np.asarray( load_h5(montage_file, 'Metadata/EDS Starting Bin Voltage (eV)'), dtype = float))
            channel_width =  np.mean( np.asarray( load_h5(montage_file, 'Metadata/EDS Voltage Bin Width (eV)'), dtype = float))
            montage_peaks = np.asarray( load_h5(montage_file, 'EDS/Autodetected Peak Bins')) 
            montage_kev = list((channel_offset + channel_width*np.asarray(montage_peaks))/1_000.0)

            search_width = 0.150 # KeV 
            i = 0 
            while True:          
                
                try: 
                    peak_candidate = montage_kev[i]
                
                    if np.all( abs(analysis_kev - peak_candidate) > search_width ): 
                        analysis_kev.append(peak_candidate)
                        montage_kev.remove(peak_candidate)
                        i = 0
                    else:
                        i += 1 
                            
                except: 
                    break 
            
        # Save detected KeV peaks from all montages 
        with h5py.File(analysis_file, 'r+') as file: 
            try:
                file.create_dataset( 'Channel KeV Peaks' , data = analysis_kev)
            except ValueError:
                pass
            
        # Now that we have the KeV peaks for all montages in the analysis, we have to go back to each montage and 
        # calculate the bin for each peak (because some KeV may be in one montage but not another) and then create the channel array 
        # for analysis 
        for z, montage in enumerate(unique_montages):
           
            print("Loading peaks for " + str(montage) )
            montage_file = os.path.abspath( file_list[z] )
            channel_offset = np.mean( np.asarray( load_h5(montage_file, 'Metadata/EDS Starting Bin Voltage (eV)'), dtype = float))
            channel_width =  np.mean( np.asarray( load_h5(montage_file, 'Metadata/EDS Voltage Bin Width (eV)'), dtype = float))
            montage_peaks = np.asarray( load_h5(montage_file, 'EDS/Autodetected Peak Bins')) 

            search_width = 0.150 # KeV 
            i = 0 
            while True:          
                
                try: 
                    montage_kev = list((channel_offset + channel_width*np.asarray(montage_peaks))/1_000.0)
                    peak_candidate = montage_kev[i]
                
                    if np.all( abs(analysis_kev - peak_candidate) > search_width ): 
                        #print(peak_candidate)
                        montage_peaks.append( (peak_candidate*1000 - channel_offset) / channel_width )
                        #montage_kev.remove(peak_candidate)
                        i = 0
                    else:
                        i += 1 
                            
                except: 
                    break 
                
            # Save the bins for each montage and construct the channel array 
            with h5py.File(analysis_file, 'r+') as file: 
                try:
                    file.create_dataset( 'Montages/' + str(montage) + '/Peak Bins' , data = montage_peaks)
                    
                except ValueError:
                    pass
                
            with h5py.File(montage_file, 'r+') as file: 
                try:
                    x_range = file['EDS']['Xray Spectrum'].shape[0]
                    y_range = file['EDS']['Xray Spectrum'].shape[1]
                except ValueError:
                    pass
                
            z_range = len(montage_peaks)
           
            #data = np.zeros( shape = (x_range, y_range, z_range), dtype = np.float16 )
            with h5py.File(analysis_file, 'r+') as file: 
                try:
                    file.create_dataset( 'Montages/' + str(montage) + '/Channels', shape = (x_range, y_range, z_range) )
                    
                except ValueError:
                    pass
            
            
            
            with h5py.File(montage_file, 'r+') as file: 
                with h5py.File(analysis_file, 'r+') as file2: 
                    #data = file['EDS']['Xray Spectrum'][:,:,2]
                    z_max = file['EDS']['Xray Spectrum'].shape[2]
                    
                    for q, peak in enumerate( tqdm(montage_peaks, total = len(montage_peaks) ) ):
                        
                        upper = int(round( min( z_max , peak + (search_width*1000 - channel_offset) / channel_width)))
                        lower = int(round( max( 0 , peak - (search_width*1000 - channel_offset) / channel_width )))
                        
                        layer = np.sum( file['EDS']['Xray Spectrum'][: ,: , lower:upper ], axis = 2)
                        
                        #layer = np.array(layer, dtype = np.uint16)
                        layer = np.array(layer, dtype = np.float32)
                        #cv2.im
                        
                        # Apply chosen preprocessing filter         
                        if filter_mode == 1: 
                            layer = cv2.medianBlur(layer,filter_size)
                        elif filter_mode == 2:
                            layer = cv2.GaussianBlur(layer, (filter_size,filter_size),0 ) 
                        elif filter_mode == 3:
                            layer = cv2.bilateralFilter(layer,filter_size,75,75)
                        
                        # Add filtered image to dataset 
                        file2['Montages/' + str(montage) + '/Channels'][:,:,q] = layer
                        gc.collect() 
            
    ########
    
    print("Final list of montages to analyze together: ")
    for montage in unique_montages:
        print(montage)    
    print("")
    
    ########           
       
    # create initial training dataset 
    print("Beginning Analysis")
    generation = 0
    training_data = [] 
    num_training_points_per_montage = int( round( initial_sampling_size / len(unique_montages), 0 ) )

    # It is desirable to trace where training data came from
    # The initial training data is randomly selected from all montages
    for i, montage in enumerate( unique_montages ):        
        save_file = h5py.File(analysis_file, 'r+') 
        x_range = save_file['Montages'][str(montage)]['Channels'].shape[0]
        y_range = save_file['Montages'][str(montage)]['Channels'].shape[1]
        save_file.close()
        
        training_mask = np.empty( shape = (x_range, y_range) )
        training_mask[:] = np.nan

        for index in range(num_training_points_per_montage): 
            x = random.randrange(0, x_range, 1)
            y = random.randrange(0, y_range, 1)
            training_mask[x,y] = 0
    
        save_h5(analysis_file, 'Montages/' + str(montage), ['array', 'training data'], training_mask )
        
        data = load_h5(analysis_file, 'Montages/' + str(montage) + '/Channels')
        data = data[training_mask == 0, :]
    
    
        for point in range(data.shape[0]): 
        
            training_data.append( data[point,:] ) 

        len(training_data)

    training_data = np.asarray( training_data )
    ########
    
    while autodidactic_loop == True: 
        
        # Train VB-GMM model on training subset
        gmm_name = 'Generation ' + str(generation) 
        weights, means, covar, loss = fit_vbgmm(training_data, "Model " + gmm_name) 
        components = len(weights)
        identified = [] 
        
        for montage in unique_montages: 
            data = load_h5(analysis_file, 'Montages/' + str(montage) + '/Channels')
            x = data.shape[0]
            y = data.shape[1]
            data = data.reshape(x*y, data.shape[2])
            
            uncertainty = calc_uncertainty(data, weights, means, covar)
            uncertainty = np.asarray(uncertainty).reshape(x, y)
            if (residual_pixel_threshold > len(uncertainty[uncertainty <= autodidectic_threshold].flatten() )):
                identified.append(True)
            else: 
                identified.append(False) 
                
            save_h5(analysis_file, 'Montages/' + str(montage), ['array', gmm_name + ' Uncertainty'], uncertainty )

        if (generation + 1) < max_autodidectic_iterations: 
            
            # Once the model tuning is complete, append new training data if necessary
            counter = 0 
            
            # After refitting the model and calculating the uncertainty of each montage
            # Append new training data as needed 
            for z, montage in enumerate(unique_montages): 
                # load uncertainty 
                    
                # if the number of 'unidentified' pixels is more than an acceptable number, add more training points 
                if identified[z]:
                    #(residual_pixel_threshold > len(array[uncertainty <= autodidectic_threshold].flatten() )):
                    counter += 1 
                    
                else: 
                    data = load_h5(analysis_file, 'Montages/' + str(montage) + '/Channels')
                    uncertainty = load_h5(analysis_file, 'Montages/' + str(montage) + '/' + gmm_name + ' Uncertainty')
                    training_mask = load_h5(analysis_file, 'Montages/' + str(montage) + '/training data')
                    
                    
                    
                    index = np.where(uncertainty <= autodidectic_threshold)
                    
                    #num_uncertainty = new_data.shape[0]
                    num_uncertainty = int( round( (len( identified) - sum( identified )) / len( identified) * training_set_growth_rate,0))
                    
                    new_data_index = random.choices( range( len(index[0] ) ), k = int( num_uncertainty) ) 
                    
                    training_mask[index[0][ new_data_index ] , index[1][ new_data_index ]] = generation 
                    
                    save_h5(analysis_file, 'Montages/' + str(montage), ['array', 'training data'], training_mask.astype('int16') )
                    
                    new_data = data[index[0][ new_data_index ] , index[1][ new_data_index ]]
                    
                    training_data = np.append(training_data, new_data, axis = 0) 

                    """
                    #new_data = np.asarray(new_data)
                    
                    print("Current number of training points")
                    print(str( training_data.shape[0] ))
                    print("Number of unidentified points")
                    print(str(num_uncertainty) )
                    print("Number of points appended")
                    print(str(int(training_set_growth_rate ) ))
                    print("Number of training points after autodidactic loop")
                    print(str( training_data.shape[0] ))
                    """
                    
                    generation += 1 
                   # np.save('Generation ' + str(generation) + ' Training Data.npy', training_data, allow_pickle=False)
                    gc.collect()
                    continue
                
        else: 
            autodidactic_loop = False
            print("Training Complete")
            break 
        
    for montage in unique_montages: 
        data = load_h5(analysis_file, 'Montages/' + str(montage) + '/Channels')
        x = data.shape[0]
        y = data.shape[1]
        data = data.reshape(x*y, data.shape[2])        
        segmentation = calc_segmentation(data, weights, means, covar)
        segmentation = np.asarray(segmentation).reshape(x, y)
        save_h5(analysis_file, 'Montages/' + str(montage), ['array', 'Segmentation'], segmentation )

    save_h5(analysis_file, 'GMM Parameters', ['array', 'Means'], means )
    save_h5(analysis_file, 'GMM Parameters', ['array', 'Weights'], weights )
    save_h5(analysis_file, 'GMM Parameters', ['array', 'Covariance'], covar )
    save_h5(analysis_file, 'GMM Parameters', ['array', 'Loss'], loss )












if __name__ == "__main__":
    main()





