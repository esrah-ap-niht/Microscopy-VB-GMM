
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
import microscopy_parsers
from pathlib import Path
import h5py
import glob
import hyperspy.api as hs
from matplotlib.ticker import (MultipleLocator)
import matplotlib.ticker as tck
import matplotlib as mpl
from tqdm import tqdm

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
plt.ioff()
root = Tk()
root.withdraw()
root.attributes('-topmost',1)
sns.set(font_scale = 3)
sns.set_style("white")

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
    
            shape = image.shape
            if len(shape) > 2: 
                raise ValueError("ERROR: Input 2D layers not detected as 2D. Input '" + str(file) + "' has shape " + str(shape) )
            shape = ( shape[0], shape[1], len(file_list))
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
        if len(shape) > 2: 
            raise ValueError("ERROR: Input 2D layers not detected as 2D. Input '" + str(file) + "' has shape " + str(shape) )
            
        image = np.array(image) 
        """
        file_list[i] = os.path.splitext(file)[0]
        
        # Apply chosen preprocessing filter         
        if filter_mode == 1: 
            image = cv2.medianBlur(image,filter_size)
        elif filter_mode == 2:
            image = cv2.GaussianBlur(image, (filter_size,filter_size),0 ) 
        elif filter_mode == 3:
            image = cv2.bilateralFilter(image,filter_size,75,75)
        """
        # Add filtered image to dataset 
        array[:,:, i] = image
        
    # Reduce the precision of the array to the lowest acceptable level in order to reduce memory requirements
    max_value = np.max(array)
    if max_value <= 255: 
        array = np.array(array, dtype = np.uint8)
    elif max_value <= 65_535:
        array = np.array(array, dtype = np.uint16)
    else: 
        array = np.array(array, dtype = np.uint32)

    return array 



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
        for point in tqdm(range(10_000)): 
            #print("Uncertainty Calculation Progress: " + str( round(point / 10_000 * 100, 2) ) )
            start = round( (data.shape[0] / 10_000) * point )
            end   = round( (data.shape[0] / 10_000) * (point +1 ) )
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
        for point in tqdm(range(10_000)): 
            #print("Segmentation Calculation Progress: " + str( round(point / 10_000 * 100, 2) ) )
            start = round( (data.shape[0] / 10_000) * point )
            end   = round( (data.shape[0] / 10_000) * (point +1 ) )
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
        nLap=100, 
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











##############################################################################################################
# Start of program 
##############################################################################################################
global full_data
global DPGMM
global initial_weights
size = 40
fontsize = 60
ratio = 2
color = 'seismic_r'
sns.set(font_scale = 3)
sns.set_style("white")

# Get user to input data format
print("Enter the format of input data. Data may be a series of montaged & superimposed 2D images or 3D data cubes")
print("The same input format must be used for all montages in the analysis")
print("0 - 2D images (png, jpg, npy, csv, xlsx, tif, tiff")
print("1 - Oxford 3D datacube .RPL & .RAW files")     
input_format = int(input(""))
print("")

graphing_dpi = int(input("Enter the resolution of final outputs in dots per inch (DPI)"))


# Get user to select input directory(s)
input_src = []
while True: 
    get_path = filedialog.askdirectory( title = "Select input directory(s). Files will be copied to the output directory. Press 'calcel' to stop adding folders")
    if get_path != '':
        input_src.append(get_path)
    else:
        break 
    
    
# Get user to select output directory 
output_src = filedialog.askdirectory( title = "Select output directory")


# Get list of all files in input directories. Create seperate file lists for oxford
file_list = []
file_list_stems = [] 
oxford_file_list = []


for src in tqdm(input_src, desc ="Reading input directories" ):
    for file in glob.glob(src + '/**/*', recursive=True):
        if Path(file).suffix in ['.tif', '.tiff', '.png', '.jpg', '.csv', '.xlsx', '.npy', '.h5oina', '.raw', '.rpl']:
            file_list.append(file)
            file_list_stems.append( os.path.split(file)[1] )
            
            if Path(file).suffix in ['.h5oina']: 
                oxford_file_list.append(file)


# If oxford files are present, parse metadata and find the unique montages available 
if len(oxford_file_list) > 0:
    print("Reading Oxford Metadata")
    print("")
    
    metadata_marker = False
    for file in file_list:
        if 'Metadata.csv' in file:
            metadata = pd.read_csv(file)
            metadata_marker = True
    if metadata_marker == False:
        metadata, ebsd_phases = microscopy_parsers.oxford_get_metadata(oxford_file_list)
    
    """
    print("Oxford files detected. Enter the nominal beam amperage in nano-amps")
    amperage = input("")
    print("")
    """
    
    channel_offset = np.mean(metadata['EDS Starting Bin Voltage (eV)'])
    channel_width =  np.mean(metadata['EDS Voltage Bin Width (eV/bin)'])
    
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

    try: 
        os.mkdir(os.path.join(output_src, 'Data'))
        os.chdir(os.path.join(output_src, 'Data'))
    except FileExistsError:
        os.chdir(os.path.join(output_src, 'Data'))
    metadata.to_csv('Metadata.csv', index=False)
    unique_montages = list(np.unique(np.asarray( metadata['Montage Label'].values, dtype = str ) ) )
    
# If oxford files are not available, parse montage names from directories 
elif input_format != 1:   
    unique_montages = []
    for src in input_src: 
        _, path = os.path.split(src)
        unique_montages.append(path)
else: 
    raise ValueError("ERROR: Oxford .h5oina files not found but Oxford RPL & RAW files specified for input format. Verify input file locations and formats and try again")
        
    
# Remove empty montage names or nan if present 
try: 
    unique_montages.remove('')
except ValueError: 
    pass
try: 
    unique_montages.remove('nan')
except ValueError: 
    pass


# Present detected montages to user and get input on which montages to include in analysis 
print(str( len(unique_montages)) + " unique montages found" )
for montage in unique_montages:
    print(montage)
print("")
print("Indicate with 'y' for yes or 'n' for no which montages are to be analyzed")
analyze = []
for montage in unique_montages:
    print(montage)    
    analyze.append( input() )
    
to_delete = list(np.where(np.asarray(analyze) == 'n')[0] )
to_delete.sort(reverse = True)

for val in to_delete: 
    del unique_montages[ val ]

print("")
print("Final list of montages to analyze together: ")
for montage in unique_montages:
    print(montage)    
print("")

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
    
    
# if Oxford 3D datacubes are used, auto detect peaks, and collect sum of spectra
if input_format == 1: 
    for montage in unique_montages:
        # check to see if a previous analysis has been performed and xray peaks already auto detected 
        checks = [True, True, True]
        
        for file in file_list: 
            if ("sum_spectrum" in file) and (montage in file):
                checks[0] = False
                sum_spectrum = np.load(file)
                continue 
            if ("highest_spectrum" in file) and (montage in file):
                checks[1] = False
                highest_spectrum = np.load(file)
                continue 
            if ("appended_peaks" in file) and (montage in file):
                checks[2] = False
                appended_peaks = np.load(file)
                continue 
                
        if np.all(checks): 
            print(' ')
            print("Auto-detecting XRay peaks for montage: " + str(montage))
            print("")
            # Find the eds xray peaks if no previous analysis 
            appended_peaks, montage_sum_spectrum, montage_highest_spectrum = microscopy_parsers.find_eds_xray_peaks(metadata, [montage], file_list, search_width)
        
            try: 
                os.mkdir(os.path.join(output_src, 'Data'))
                os.chdir(os.path.join(output_src, 'Data'))
            except FileExistsError:
                os.chdir(os.path.join(output_src, 'Data'))
                
            try: 
                os.mkdir(os.path.join(output_src, 'Data', 'Montages'))
                os.chdir(os.path.join(output_src, 'Data', 'Montages'))
            except FileExistsError:
                os.chdir(os.path.join(output_src, 'Data', 'Montages'))
                  
            try: 
                os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            except FileExistsError:
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
               
            np.save("sum_spectrum", np.asarray(montage_sum_spectrum).flatten(), allow_pickle = False)
            np.save("highest_spectrum", np.asarray(montage_highest_spectrum).flatten().flatten(), allow_pickle = False)
            np.save("appended_peaks", np.asarray(appended_peaks).flatten().flatten(), allow_pickle = False)
    
    
    for z, montage in enumerate(unique_montages):
        try: 
            os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
        except FileExistsError:
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
        
        print("Loading peaks for " + str(montage) )
        if z == 0:
            appended_peaks = np.load('appended_peaks.npy')
            continue 
        else: 
            candidates = np.load('appended_peaks.npy')
        
        i = 0 
        while True:          
            
            try: 
                peak_candidate = candidates[i]
            
                if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                    appended_peaks.append(peak_candidate)
                    candidates.remove(peak_candidate)
                    i = 0
                else:
                    i += 1 
                        
            except: 
                break 
        
    for z, montage in enumerate(unique_montages):
        try: 
            os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
        except FileExistsError:
            os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
        
        np.save("appended_peaks", appended_peaks, allow_pickle = False)
        
    #energy_locs = list((channel_offset + channel_width*np.asarray(appended_peaks))/1_000.0)
    #distance = 4
      
    for montage in unique_montages:
        
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
                   
            appended_peaks = np.load('appended_peaks.npy')
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
        
          
# Begin training loop
generation = 0


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
        
        autodidactic_loop = True 
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
        full_data = array.flatten().reshape(array.shape[0] * array.shape[1], len(appended_peaks))
    
        print("Beginning Analysis")
        
        if w == 0:
            # Sample randomly from the entire montage
            training_data = random.sample(list(full_data), min(int(initial_sampling_size), full_data.shape[0] ) )
            training_data = np.asarray(training_data)
            
            # Update working directory 
            try: 
                os.mkdir(os.path.join( output_src, 'Training'))
            except: 
                pass 
            
            try: 
                os.mkdir(os.path.join( output_src, 'Training', 'Training Data' ))
            except: 
                pass 
            
            
            os.chdir(os.path.join( output_src, 'Training', 'Training Data' ) )
            np.save('Generation ' + str(generation) + ' Training Data.npy', training_data, allow_pickle=False)
    
            # Define model parameters 
            #global DPGMM
            #global initial_weights
    
        loop = 0 
        while autodidactic_loop == True: 
            
            # Train VB-GMM model on training subset
            gmm_name = 'Generation ' + str(generation) 
            
            if (w == 0) or (loop != 0): 
                
                weights, means, covar, loss = fit_vbgmm(training_data, "Model " + gmm_name) 
                
                DPGMM = mixture.GaussianMixture(n_components = len(weights) )
                
                #if generation == 0 
                DPGMM.weights_ = np.asarray(weights)
                DPGMM.means_ = np.asarray(means)
                DPGMM.covariances_ = np.asarray(covar)
                precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covar))
                DPGMM.precisions_cholesky_ = precisions_cholesky 
                """
                # Reweight model classes based on entire dataset 
                initial_weights = DPGMM.weights_
                cons = {'type':'eq', 'fun': constraint}
                boundaries = []
                
                for i in range(len(initial_weights) ):
                    boundaries.append( (0,0.999_999) ) 
                    
                response = scipy.optimize.minimize(gmm_weights, initial_weights, tol = 1e-8, constraints = cons, bounds = boundaries, options = {'disp': True, 'maxiter' :1000} )
                DPGMM.weights_ = response.x
                """
                components = len(DPGMM.weights_)
            
                # Save model
                
                try: 
                    os.mkdir(gmm_name)
                    os.chdir(gmm_name)
                except FileExistsError:
                    os.chdir(gmm_name)
                np.save(gmm_name + '_weights', DPGMM.weights_, allow_pickle=False)
                np.save(gmm_name + '_means', DPGMM.means_, allow_pickle=False)
                np.save(gmm_name + '_covariances', DPGMM.covariances_, allow_pickle=False)
              
            
            uncertainty = calc_uncertainty(full_data, weights, means, covar)
            uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
            """
            # It is not necessary to re-analyze the entire dataset for every generation, only for the first and last iteration
            # Intermediate generations only need to have pixels with high uncertainty from the previous generation re-analyzed 
            # This results in significant time efficiencies  
            try: 
                uncertainty = uncertainty.flatten()
                subset = full_data[ uncertainty <= autodidectic_threshold, : ]
                uncertainty[ uncertainty <= autodidectic_threshold ] = calc_uncertainty(subset, weights, means, covar)
                uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
            except NameError: 
                uncertainty = calc_uncertainty(full_data, weights, means, covar)
                uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
            except ValueError:
                uncertainty = calc_uncertainty(full_data, weights, means, covar)
                uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
            except IndexError:
                uncertainty = calc_uncertainty(full_data, weights, means, covar)
                uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
            """
    
            
            np.save( str(montage) + '_Log_Probabilities.npy', uncertainty, allow_pickle=False)
            loop += 1
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
            
            
            # Return up one directory level 
            os.chdir( os.path.dirname(os.getcwd()) )
    
            # Once the model tuning is complete, append new training data if necessary
            if (residual_pixel_threshold > len(array[uncertainty <= autodidectic_threshold].flatten() )):
                autodidactic_loop = False
                print("Analysis Truncated Due to Residual Un-Identified Pixels")
                break 
            elif  (np.min(uncertainty) < autodidectic_threshold) and ( (generation + 1) < max_autodidectic_iterations):
                new_data = array[uncertainty <= autodidectic_threshold]
                num_uncertainty = new_data.shape[0]
                
                new_data = random.choices(list(new_data), k = int( training_set_growth_rate) ) 
                new_data = np.asarray(new_data)
                
                print("Current number of training points")
                print(str( training_data.shape[0] ))
                print("Number of unidentified points")
                print(str(num_uncertainty) )
                print("Number of points appended")
                print(str(int(training_set_growth_rate ) ))
                training_data = np.append(training_data, new_data, axis = 0) 
                print("Number of training points after autodidactic loop")
                print(str( training_data.shape[0] ))
        
                generation += 1 
                np.save('Generation ' + str(generation) + ' Training Data.npy', training_data, allow_pickle=False)
                gc.collect()
                continue
            else: 
                autodidactic_loop = False
                print("Analysis Complete")
                break 
    try:     
        os.mkdir("Analysis")
        os.chdir("Analysis")
    except FileExistsError:    
        os.chdir("Analysis")
    
    # Save updated model
    np.save('Final_Model_Weights', DPGMM.weights_, allow_pickle=False)
    np.save('Final_Model_Means', DPGMM.means_, allow_pickle=False)
    np.save('Final_Model_Covariances', DPGMM.covariances_, allow_pickle=False)
    
                
        
#####################################################################
# Create final analysis folder 
print("Beginning Plotting")

for w, montage in enumerate(unique_montages): 
    # Check if the segmentation and uncertainty maps have already been calculated. If not, calculate and save results 
    try:     
        os.mkdir(os.path.join(output_src ,"Analysis"))
        os.chdir(os.path.join(output_src ,"Analysis"))
    except FileExistsError:    
        os.chdir(os.path.join(output_src ,"Analysis"))
        
    try:     
        os.mkdir(os.path.join(output_src ,"Analysis", str(montage)))
        os.chdir(os.path.join(output_src ,"Analysis", str(montage)))
    except FileExistsError:    
        os.chdir(os.path.join(output_src ,"Analysis", str(montage)))
        
        
    
    if os.path.exists( os.path.join(output_src, 'Analysis', str(montage), 'Segmentation.npy' )) and os.path.exists( os.path.join(output_src, 'Analysis', str(montage), 'Log_Probabilities.npy' )):
        pass 
    else: 
        # We only need to load data once for both calculations 
        array = np.load(os.path.join(output_src, 'Data', 'Montages', str(montage), "['EDS', 'XRay'].npy") )   
        full_data = array.flatten().reshape(array.shape[0] * array.shape[1], len(appended_peaks))
        if os.path.exists( os.path.join(output_src, 'Analysis', str(montage), 'Log_Probabilities.npy' )):
            pass
        else:
            segmentation = calc_segmentation(full_data, DPGMM.weights_, DPGMM.means_, DPGMM.covariances_)
            segmentation = np.asarray(segmentation).reshape(array.shape[0], array.shape[1])
            np.save('Segmentation', segmentation, allow_pickle=False)
            
        if os.path.exists( os.path.join(output_src, 'Analysis', str(montage), 'Log_Probabilities.npy' )):
            pass
        else:
            uncertainty = calc_uncertainty(full_data, DPGMM.weights_, DPGMM.means_, DPGMM.covariances_)
            uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
            np.save('Log_Probabilities.npy', uncertainty, allow_pickle=False)
        
        
    
#################################################################################################
for w, montage in enumerate(unique_montages): 
    
    print("")
    print("Finding available backgrounds ")

    oxford_file_list2 = [] 
    
    for montage in unique_montages: 
        
        for index, file in metadata.iterrows(): 
            
            if montage == file['Montage Label']:
                oxford_file_list2.append( oxford_file_list[index] ) 
    
    oxford_file_list = oxford_file_list2
    
    specifications = [
        ['Electron Image', "SE"], 
        ['Electron Image', "BSE"], 
        ['Electron Image', "FSE Lower Left"], 
        ['Electron Image', "FSE Lower Centre"], 
        ['Electron Image', "FSE Lower Right"], 
        ['Electron Image', "FSE Upper Left"], 
        ['Electron Image', "FSE Upper Right"],
        ['EDS', 'Live Time'],
        ['EDS', 'Real Time'],
        ['EBSD', 'Band Contrast'],
        ['EBSD', "Band Slope"], 
        ['EBSD', "Bands"],
        ['EBSD', "Beam Position X"],
        ['EBSD', "Beam Position Y"],
        ['EBSD', "Detector Distance"],
        ['EBSD', "Error"]]
    
    '''
    , 
        #['EBSD', "Euler"], 
        
        ['EBSD', "Mean Angular Deviation"], 
        ['EBSD', "Pattern Center X"], 
        ['EBSD', "Pattern Center Y"], 
        ['EBSD', "Pattern Qualtiy"], 
        ['EBSD', "Phase"], 
        ['EBSD', "X"], 
        ['EBSD', "Y"] ]
    '''
    
    available_specifications = []
    
    for specification in specifications:
        
        for montage in unique_montages:
            
            specification_found = False
            
            try: 
                os.mkdir(os.path.join(output_src, 'Data'))
            except: 
                pass 
            
            try: 
                os.mkdir(os.path.join(output_src, 'Data', 'Montages'))
            except: 
                pass 
            
            try: 
                os.mkdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
            except FileExistsError:
                os.chdir(os.path.join(output_src, 'Data', 'Montages', str(montage) ))
                
            try:  
                for file in file_list: 
                    if (str(specification) in file) and (montage in file):
                        available_specifications.append(specification) 
                        specification_found = True
                        break 
                if specification_found: 
                    pass 
            except: 
                pass 
                
            if specification_found == False:
                    print("Stitching: " + str(montage) + " - " + str(specification) )
                    stitched = microscopy_parsers.oxford_montage_stitcher(montage, metadata, specification, file_list, None, None)
                    available_specifications.append(specification) 
                    np.save( str(specification) + '.npy', stitched, allow_pickle=False)
            
            

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

"""
order = np.argsort(np.asarray(display_shells ))
display_shells.sort()
temp_appended_peaks = [] 


for value in order: 
    



test = appended_peaks.copy()

test = list(test.flatten()) 
test.sort(key = list(order.flatten()) )

(key = order)

sorted(test , key = order.flatten() )


"""
    
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


components = len(DPGMM.weights_)
energy_locs = list((channel_offset + channel_width*np.asarray(appended_peaks))/1_000.0)

temp = []
for i in range(len(display_shells)):
    if str(display_shells[i]) != '':
        temp.append(str(display_shells[i]) + ": " + str( round(energy_locs[i], 3) ) + "KeV")
    else: 
        temp.append( str( round(energy_locs[i], 3) ) + "KeV")
        
display_shells = temp.copy()



for z, montage in enumerate(unique_montages): 
    try:     
        os.mkdir(os.path.join(output_src ,"Analysis"))
        os.chdir(os.path.join(output_src ,"Analysis"))
    except FileExistsError:    
        os.chdir(os.path.join(output_src ,"Analysis"))
        
    try:     
        os.mkdir(os.path.join(output_src ,"Analysis", str(montage)))
        os.chdir(os.path.join(output_src ,"Analysis", str(montage)))
    except FileExistsError:    
        os.chdir(os.path.join(output_src ,"Analysis", str(montage)))
        
    # Create agglomerative hierarchial model and save results 
    fig, ax = plt.subplots(figsize=(size, size))
    heir = sklearn.cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage = 'ward')
    clustering = heir.fit(DPGMM.means_)  
    linkage_matrix = plot_dendrogram(clustering, truncate_mode='level', p=100)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(str(montage) + " Consolidation Dendrogram" + ".png")
    plt.close(fig)
    gc.collect()
     
    # Prepare HAM linkage matrix for exporting 
    # Remove all non-first-level linkages 
    
    linkage_matrix = np.delete(linkage_matrix,3,1)
    
    # Reformat to three columns 
    linkage_matrix = np.concatenate( (np.delete(linkage_matrix, 1, 1), np.delete(linkage_matrix, 0, 1) ), axis = 0)
    
    index_to_delete = []
    uniques = np.unique(segmentation)
    for i, row in enumerate(linkage_matrix[:,0]):
        if (row in uniques) == False:
            index_to_delete.append(i)
            
    linkage_matrix = np.delete(linkage_matrix, np.array(index_to_delete, dtype = np.int16), 0)
    
    linkage_matrix = pd.DataFrame( data = linkage_matrix, columns = ["Class ID", "Dissimilarity Scale"])
    
    index_to_delete = []
    for row in range(len(DPGMM.weights_)):
        if (row in uniques) == False:
            index_to_delete.append(row)
            
    weights = np.delete(DPGMM.weights_, np.array(index_to_delete, dtype = np.int16), 0)

    linkage_matrix = linkage_matrix.sort_values(by=['Class ID'])
    linkage_matrix['Area Percent'] = weights * 100.0
    """
    for j, element in enumerate(display_shells): 
        element_percents = []
        for i, class_id in enumerate(linkage_matrix['Class ID']): 
            element_percents.append(DPGMM.means_[int(class_id)][j] )
        linkage_matrix[str(element)] = element_percents
    """   
    linkage_matrix.to_excel(str(montage) + " Class Data.xlsx", index = False)
    

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
    plt.savefig(str(montage) + " Fixed Scale Log Liklihoods" + ".png")
    plt.close(fig)
    gc.collect()
    
    

    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
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
    fig, ax = plt.subplots(figsize=(size, size), dpi=graphing_dpi)
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
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods with Colorbar" + ".png")
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
    plt.savefig(str(montage) + " Floating Scale Log Liklihoods Without Colorbar" + ".png")
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
    plt.savefig("Mixture Weights" + ".png", dpi=graphing_dpi)
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax, ticks=np.arange(np.min(segmentation),np.max(segmentation)+1))
    cbar.ax.tick_params(labelsize=25) 
    plt.tight_layout()
    plt.title(str(montage) + " Class Segmentation", fontsize = fontsize*(0.7))
    plt.savefig(str(montage) + " Class Segmentation with Colorbar" + ".png", dpi=graphing_dpi)
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
    plt.savefig(str(montage) + " Class Segmentation Without Colorbar" + ".png", dpi=graphing_dpi)
    plt.close(fig)
    gc.collect()
    
    # Get the um/px scale 
    x_scale = list( np.unique( metadata['EDS X Step Size (um)'][metadata['Montage Label'] == montage] ) )
    y_scale = list( np.unique( metadata['EDS Y Step Size (um)'][metadata['Montage Label'] == montage] ) )
    
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
        
    x_list =  list( np.unique( metadata['EDS Stage X Position (mm)'][metadata['Montage Label'] == montage] ) )
    y_list =  list( np.unique( metadata['EDS Stage Y Position (mm)'][metadata['Montage Label'] == montage] ) ) 
   
    global x_min
    global y_min
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
    

    
    # Create binary class maps and save 
    w = (-DPGMM.weights_).argsort()
    uniques = np.unique(segmentation)
    w = uniques
    
    
    for ind2 in w:
        
        # Ignore extraneous classes that are not in the segmentation 
        if ind2 in uniques: 
            gc.collect()
            
            
            #test = segmentation[0:1000, 0:1000]
            
            class_map = np.ones(shape = segmentation.shape, dtype = np.uint8 )
            class_map[segmentation != ind2] = 0
            dilation = cv2.dilate(class_map, np.ones((3,3), 'uint8'), iterations = 3)
            
            dilation = dilation * 255
            class_map = class_map * 255
            
            
            # we want to 
       
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
            
            # add secondary axis labels for the stage locations             
            secondx_ax1 = ax1.secondary_xaxis('top', functions = (forward_x, reverse_x))
            secondx_ax1.set_xlabel('X Stage Location (mm)', fontsize = 12)
            secondx_ax1.tick_params(labelsize=10)
                        
            secondy_ax1 = ax1.secondary_yaxis('left',  functions = (forward_y, reverse_y))
            secondy_ax1.set_ylabel('Y Stage Location (mm)', fontsize = 12)
            secondy_ax1.tick_params(labelsize=10)
            
            try:
                plt.sca(ax1)
                plot = plt.imshow(background, alpha = 0.3, cmap = 'gray_r')
            except NameError:
                pass 
            
            
            ax2 = fig.add_subplot(gs[0, 3:5])
            ax2.set_title('Correlation Matrix', fontsize = 15)
            
            
            plt.sca(ax2)
            stdev = np.sqrt(np.abs(DPGMM.covariances_[ind2]))
            stdev[DPGMM.covariances_[ind2] < 0.0] = -1 * stdev[DPGMM.covariances_[ind2] < 0.0] 
            annot = np.diag(DPGMM.precisions_cholesky_[ind2],0)
            annot = np.round(annot,2)
            annot = annot.astype('str')
            annot[annot=='0.0']=''
            sns.heatmap(correlation_from_covariance(DPGMM.covariances_[ind2]), xticklabels = display_shells, yticklabels = display_shells, center = 0, vmin = 0, vmax = 1, linewidths=1, linecolor = 'white', cmap = 'bwr', mask = np.triu(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'vertical'})
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
            
            
            """
            ax3 = fig.add_subplot(gs[1, 3:5])
            ax3.set_title('Class ' + str(ind2) + ' Sum of Spectrum', fontsize = 15)
            plt.sca(ax3)
            #plt.plot( (channel_offset + channel_width*np.linspace(0,1024, num=1024))/1_000.0, class_spectra[0,ind2,:] )
            plt.tick_params(left = True)
            plt.tick_params(bottom = True)
            plt.xticks( fontsize = 10)
            plt.yticks( fontsize = 10)
            plt.xlabel("KeV", fontsize = 12)
            plt.ylabel("Total Counts", fontsize = 12)
            ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            
            ax3.spines["top"].set_visible(False)
            
          
            
            
            ax5 = fig.add_subplot(gs[:, 5])
            ax5.set_title('Metadata', fontsize = 15)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)
            plt.xticks([])
            plt.yticks([])
            rows = ['Project:',
                    'Specimen',
                    'KeV',
                    'Amps',
                    'Date',
                    'Acq Time',
                    'Avg Frames',
                    'Process Time',
                    'Resolution',
                    'Magnification']
            
            data = [[ 66386, 174296,  75131, 577908,  32015],
                    [ 58230, 381139,  78045,  99308, 160454],
                    [ 89135,  80552, 152558, 497981, 603535],
                    [ 78415,  81858, 150656, 193263,  69638],
                    [139361, 331509, 343164, 781380,  52269],
                    [ 66386, 174296,  75131, 577908,  32015],
                            [ 58230, 381139,  78045,  99308, 160454],
                            [ 89135,  80552, 152558, 497981, 603535],
                            [ 78415,  81858, 150656, 193263,  69638],
                            [139361, 331509, 343164, 781380,  52269]
                            ]
                    
            #plt.table( cellText = data, loc = 'center')
            """        
                    
            plt.tight_layout()
           
            
            plt.savefig(str(montage) + " Class " + str(ind2) + " With Correlation.png")
            gc.collect()
            

          
            
    print("Number of unique classes: " + str(len(list(np.unique(segmentation)))))
    print(" ")
    print("Plotting Complete")
    print(" ")
    print("Results located in ")
    print(" ")
    print(str(output_src) ) 

"""
full_data = np.load("Filtered Array.npy")
#training_data = np.load("Initial Training Data.npy")
generation = 2

covariances = np.load("Generation " + str(generation) + "_covariances.npy")
means = np.load("Generation " + str(generation) + "_means.npy")
weights = np.load("Generation " + str(generation) + "_weights.npy")
precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covariances))
segmentation = np.load("Segmentation.npy")
uncertainty = np.load("Log_Probabilities.npy")

covariances = np.load("Final_Model_Covariances.npy")
means = np.load("Final_Model_Means.npy")
weights = np.load("Final_Model_Weights.npy")
precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covariances))
segmentation = np.load("FeNiCr LAM try2_Segmentation.npy")
uncertainty = np.load("FeNiCr LAM try2_Log_Probabilities.npy")

DPGMM = mixture.GaussianMixture(n_components = len(weights) )   
DPGMM.weights_ = np.asarray(weights)
DPGMM.means_ = np.asarray(means)
DPGMM.covariances_ = np.asarray(covariances)
precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covariances))
DPGMM.precisions_cholesky_ = precisions_cholesky 

    
DPGMM.covariances_ = covariances 
DPGMM.means_ = means 
DPGMM.weights_ = weights 
DPGMM.precisions_cholesky_ = precisions_cholesky 

sum_spectrum = np.load('sum_spectrum.npy')
highest_spectrum = np.load('highest_spectrum.npy')
display_shells = np.load('display_shells.npy')
appended_peaks = np.load('appended_peaks.npy')

array = np.load("['EDS', 'XRay'].npy")


np.save('Class Spectra', array, allow_pickle = False)
class_spectra = np.load('Class Spectra.npy')

background = np.load("['EBSD', 'Bands'].npy")





"""



















