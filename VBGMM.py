# import necessary packages 
import pandas as pd
from  pandas.errors import EmptyDataError
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

# Default settings 
gc.enable()
plt.ioff()
root = Tk()
root.withdraw() 

# Define functions needed later 

def score_samples(data): 
    return DPGMM.score_samples( data )

# Conversion of covariance matrix to correlation matrix (i.e. normalization)
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# Create linkage matrix and then plot the dendrogram
def plot_dendrogram(model, **kwargs):
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

# Function to calculate the accuracy of a set of mixture weights 
def GMM_weights(weights):
    DPGMM.weights_ = weights
    index = np.random.randint(full_data.shape[1], size=20_000)
    score = abs( min( DPGMM.score_samples(np.asarray(full_data[index,:])) ) )
    DPGMM.weights_ = initial_weights
    print(weights[0])
    return score

# GMM mixture weight constraint. Sum of mixture must equal 1 
def constraint(t):
    return sum(t) - 1

#############################################################################################################
# Start of program 

generation = 0
global full_data
ratio = 2
color = 'jet_r'
sns.set(font_scale = 3)
sns.set_style("white")
size = 30
fontsize = 30
autodidactic_loop = True 
run = True

# The program may be run in one of three modes 
# mode 1) One-time analysis. Use for a single dataset 
# mode 2) Setup for a queuing system. Opens a .txt file and appends the necessary information to conduct an analysis 
# mode 3) Continuous analysis. Reads the necessary parameters and input file locations from a .txt file and modifies the .txt file once an analysis is completed. 

operating_mode = int(input("""Select the operating mode. 
                           Enter 1 to conduct a one-time analysis
                           2 to enter information for a queuing file
                           or 3 to enter continuous analysis mode reading from a queuing file """))

# If conducting a one-time analysis or setting up a queue item
if (operating_mode == 1) or (operating_mode == 2): 
    
    filter_mode = int(input("Select the preprocessing filter to use. 0 for no filter (use raw quantified data), 1 for median filter, 2 for Gaussian filter, 3 for bilateral filter. Recommend Gaussian. "))
    if filter_mode != 0:
        filter_size =  int(input("Enter the filter pixel size in odd positive integers greater than 1. Higher values result in smoothing over a larger area. Recommend 5 "))
    number_of_classes = int(input("Enter initial integer number of classes. Recommend 30 to 40. "))
    EM_tolerance = float(input("Enter Expectation Maximization tolerance. Use smaller values for more precise composition estimates at the expense of longer analysis times. Recommend '1e-5'. Do not use quotes "))
    number_of_initializations = int(input("Enter integer number of training initializations. Recommend 5 to 10 "))
    initial_sampling_size = int(input("Enter integer number of initial training datapoints. Recommend 30_000 "))
    training_set_growth_rate = float(input("Enter training dataset growth rate. Recommend 30_000 "))
    autodidectic_threshold = float(input("Enter Log Probability threshold for autodidactic loop. Set to -1000 to disable. Recommend -40 "))
    max_autodidectic_iterations = int(input("Enter the maximum integer number of training loops to execute. Recommend 3 "))
    eds_quality_threshold = float(input("Enter EDS quality threshold (e.g. 99 for 99%). This filter is used to remove data artifacts where the sum of elements (at% or wt%) is less than 100%. Set to 0 to disable "))
    residual_pixel_threshold = float(input("Enter the cutoff number of un-identified pixels to end training. When fewer than this number of pixels remain un-identified, the model will end training "))
    print(' ')
    
    # Get user to select input directory 
    input_src = filedialog.askdirectory( title = "Select input directory. Multispectral files should be located in a single folder and TIFF, PNG, or JPG formats")
        
    # Get user to select optinal background image (e.g. SEM or BSE)
    background_filepath = filedialog.askopenfilename( title = "Select optional background image such as Secondary, Backscatter or EBSD Quality Index")
    
    # Get user to select output directory 
    output_src = filedialog.askdirectory( title = "Select output directory")
    
    # If setting up a queue item, write parameters to file 
    if (operating_mode == 2):
        
        # Get user to select queuing file 
        queue_filepath = filedialog.askopenfilename( title = "Select .CSV file with queueing information. If no file exists, manually create an empty .CSV file", filetypes=[("CSV files", ".csv")])
        
        # If the file does not have appropriate headers, add them to the file 
        try: 
            queue = pd.read_csv(queue_filepath, sep=',')
        except EmptyDataError: # if file is empty 
            queue = pd.DataFrame(columns= ['filter_mode', 
                                           'filter_size',
                                           'number_of_classes',
                                           'EM_tolerance',
                                           'number_of_initializations',
                                           'initial_sampling_size',
                                           'training_set_growth_rate',
                                           'autodidectic_threshold',
                                           'max_autodidectic_iterations',
                                           'eds_quality_threshold',
                                           'residual_pixel_threshold',
                                           'input_src',
                                           'background_filepath',
                                           'output_src'
                                           ])
            
            queue.to_csv(queue_filepath, index=None)
            
        else: 
            print("Queue file recognized ")
             
        entry = pd.DataFrame(data = [[filter_mode, 
                                     filter_size,
                                     number_of_classes,
                                     EM_tolerance,
                                     number_of_initializations,
                                     initial_sampling_size,
                                     training_set_growth_rate,
                                     autodidectic_threshold,
                                     max_autodidectic_iterations,
                                     eds_quality_threshold,
                                     residual_pixel_threshold,
                                     input_src,
                                     background_filepath,
                                     output_src]], 
                             columns = queue.columns, 
                             index = None)
       
        queue = pd.concat([queue, entry])
        
        # save the appended queue 
        queue.to_csv(queue_filepath, index=None)
        print("New entry added to queue file")
        raise SystemExit

elif (operating_mode == 3):
    # Get user to select queuing file 
    queue_filepath = filedialog.askopenfilename( title = "Select .CSV file with queueing information. If no file exists, manually create an empty .CSV file", filetypes=[("CSV files", ".csv")])
        
        
while run: 
    
    # read top entry in queue file if running in continuous mode 
    if (operating_mode == 3):
        check_for_entries = True
        
        while check_for_entries: 
            queue = pd.read_csv(queue_filepath, sep=',')
            
            if len(queue.index) >= 1: 
                for col in enumerate(list(queue.columns)):
                    locals()[col[1] ] = queue.iloc[0][col[1] ]
                check_for_entries = False
                continue
            else: 
                time.sleep(10)
            
    image_list = []
    
    os.chdir(input_src)
    for file in os.listdir(): 
        if file.endswith('.tif') or file.endswith('.tiff') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.csv') or file.endswith('.xlsx'):
            image_list.append(file)
    
    # Create data array 
    file = image_list[0]
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
    
    shape = image.shape
    shape = ( shape[0], shape[1], len(image_list))
    array = np.empty(shape)
    
    # Read images and filter to reduce noise 
    for i, file in enumerate(image_list):
        # Read image file and remove filetype tag
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
        
        if (image.shape[0] != shape[0]) or (image.shape[1] != shape[1]): 
            print("Error: Layer " + str(os.path.splitext(file)[0]) + " has different dimensions than other multispectral images.")
            print("Correct images to have identical dimensions and try again")
            raise SystemExit
        
        image_list[i] = os.path.splitext(file)[0]
        print("Reading " + str(image_list[i]))
        # Apply chosen preprocessing filter         
        if filter_mode == 1: 
            image = cv2.medianBlur(image,filter_size)
        elif filter_mode == 2:
            image = cv2.GaussianBlur(image, (filter_size,filter_size),0 ) 
        elif filter_mode == 3:
            image = cv2.bilateralFilter(image,filter_size,75,75)
        
        # Add filtered image to dataset 
        array[:,:, i] = image
    
    # Update working directory 
    os.chdir(output_src)
    
    # If a background image was selected, load and check for dimension match 
    if background_filepath != '': 
        background_image = cv2.imread(background_filepath, cv2.IMREAD_GRAYSCALE) 
        if background_image.shape != shape:
            background_image = cv2.resize(background_image, (shape[1], shape[0]), interpolation = cv2.INTER_AREA)
            print("Background image has different dimensions than multispectral data. Adjusting background image to match")
        else: 
            print("Background image has same dimenstions as multispectral data. Proceeding with analysis")
    else: 
        background_image = np.mean(array, axis = 2)
        if background_image.shape != shape:
            background_image = cv2.resize(background_image, (shape[1], shape[0]), interpolation = cv2.INTER_AREA)
        else: 
            pass 
        
    if eds_quality_threshold >= np.max(array):
        print("Invalid pixels detected based on EDS Quality Threshold")
        print("Modifying invalid pixels")
        for i in range(array.shape[2]):
            #array[:,:,i] = np.ma.masked_where(np.sum(array, axis = 2) <= eds_quality_threshold, array[:, :, i])
            array[:, :, i][np.sum(array, axis = 2) <= eds_quality_threshold] = 0
    else: 
        print("No invalid pixels detected based on EDS Quality Threshold")
        
    # Save preprocessed data 
    print("Saving Data")
    np.save('Filtered Array.npy', array.data, allow_pickle=False)
    print("Preprocessing complete")
    print(" ")
    print("Data read, filtered, and saved to NPY array")
    print(" ")
    
    # Save a summation of elemental intensities. Useful for highlighting pores and other areas that did not emit XRays
    fig, ax = plt.subplots(figsize=(size, size), dpi=300)
    ax = plt.subplot()
    try:
        plt.imshow(background_image)
        plot = plt.imshow(np.sum(array, axis = 2), cmap = 'coolwarm', alpha = 0.7 )
    except NameError:
        plot = plt.imshow(np.sum(array, axis = 2), cmap = 'coolwarm' )
        pass 
    plt.title('Sum of Intensities')
    plt.xticks([])
    plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plot, cax=cax)
    plt.tight_layout()
    plt.savefig("Sum of Intensities" + ".png", dpi=300)
    plt.close(fig)
    gc.collect()
    
    # Plot histograms of the elemental intensities. Provides clues to users as to whether their data is reasonable or suspect
    # Reorganize data for histogram 
    histogram_size = 5_000
    hist_data = pd.DataFrame(data = None, columns = ["Data", "Element"])
    
    for i, layer in enumerate(image_list):      
        res = array.shape[0]*array.shape[1]
        data = np.array( [  (random.sample(list(array[:, :, i].flatten() ), min(histogram_size, res) ) )  ,  [layer] *min(histogram_size, res) ]  ).T
       
        hist_data = hist_data.append(pd.DataFrame(data = data, columns = ["Data", "Element"])  )
        print("Histogram data for layer " + str(layer) + " collected")
    del data 
    
    # Make sure elemental values are speficied as float
    hist_data = hist_data.astype({"Data": float, "Element": str})
    hist_data = hist_data.reset_index()
     
    fig, ax = plt.subplots(figsize=(size, size), dpi=300)
    ax = plt.subplot()
    sns.kdeplot( data = hist_data, 
                x = 'Data', 
                hue = 'Element', 
                common_norm=False, 
                cut = 0, 
                lw = 6, 
                log_scale = (False, False),
                palette = "tab20")
    plt.xlabel("Element Intensity")
    plt.ylabel("Relative Frequency")
    plt.title('Histograms of Intensities of Filtered Pixels')
    plt.tight_layout()
    plt.savefig("Histograms of Intensities of Filtered Pixels" + ".png", dpi=300)
    plt.close(fig)
    del hist_data
    gc.collect()
    
    
    full_data = array.flatten().reshape(array.shape[0] * array.shape[1], len(image_list))
    
    print("Beginning Analysis")
    # Sample randomly from the entire montage
    training_data =  list(array.flatten().reshape(array.shape[0] * array.shape[1], len(image_list))[:,:] )     
    training_data = random.sample(training_data, min(int(initial_sampling_size), full_data.shape[0] ) )
    training_data = np.asarray(training_data)
    np.save('Initial Training Data.npy', training_data, allow_pickle=False)
    
    # Define model parameters 
    global DPGMM
    global initial_weights
    DPGMM = mixture.BayesianGaussianMixture(n_components = number_of_classes, 
                                                        max_iter = 2_000,
                                                        n_init = number_of_initializations,
                                                        tol = EM_tolerance, 
                                                        init_params='random', 
                                                        weight_concentration_prior_type='dirichlet_process',
                                                        weight_concentration_prior = 1/number_of_classes, 
                                                        verbose = True, 
                                                        warm_start = True)
    
    while autodidactic_loop == True: 
        # Train VB-GMM model on training subset
        DPGMM.fit(training_data)
        
        # Reweight model classes based on entire dataset 
        initial_weights = DPGMM.weights_
        cons = {'type':'eq', 'fun': constraint}
        boundaries = []
        
        for i in range(len(initial_weights) ):
            boundaries.append( (0,0.999_999) ) 
            
        response = scipy.optimize.minimize(GMM_weights, initial_weights, tol = 1e-8, constraints = cons, bounds = boundaries, options = {'disp': True, 'maxiter' :1000} )
        DPGMM.weights_ = response.x
        components = len(DPGMM.weights_)
    
        # Save model
        gmm_name = 'Generation ' + str(generation) 
        os.mkdir(gmm_name)
        os.chdir(gmm_name)
        np.save(gmm_name + '_weights', DPGMM.weights_, allow_pickle=False)
        np.save(gmm_name + '_means', DPGMM.precisions_cholesky_, allow_pickle=False)
        np.save(gmm_name + '_covariances', DPGMM.covariances_, allow_pickle=False)
        
        try: 
            uncertainty = DPGMM.score_samples( full_data ) 
            gc.collect()
        except MemoryError:
            mid_point = round(full_data.shape[0] / 3.0)
            
            uncertainty = DPGMM.score_samples( full_data[0:mid_point, :] )    
            uncertainty1 = DPGMM.score_samples( full_data[mid_point: mid_point*2, :] )    
            uncertainty = np.concatenate( (uncertainty, uncertainty1), axis = 0)
            del uncertainty1
            gc.collect()
            
            uncertainty1 = DPGMM.score_samples( full_data[mid_point*2: full_data.shape[0], :] )    
            uncertainty = np.concatenate( (uncertainty, uncertainty1), axis = 0)
            del uncertainty1
            gc.collect()
                
        
        uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
        np.save('Log_Probabilities.npy', uncertainty, allow_pickle=False)
    
        # Create absolute uncertainty intensity map
        fig, ax = plt.subplots(figsize=(size, size), dpi=300)
        try:
            plt.imshow(background_image)
            plot = plt.imshow(uncertainty, cmap = "jet_r", vmin = max(-50, autodidectic_threshold), vmax = 0, alpha = 0.7 )
        except NameError:
            plot = plt.imshow(uncertainty, cmap = "jet_r", vmin = max(-50, autodidectic_threshold), vmax = 0 )
            pass 
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.ax.tick_params(labelsize=fontsize*(0.7))
        plt.title(file)
        plt.xticks([])
        plt.yticks([])
        plt.title("Absolute Scaled Log Probabilities", fontsize = fontsize*(0.7))
        plt.tight_layout()
        plt.savefig("Absolute Scaled Log Probabilities" + ".png")
        plt.close(fig)
        gc.collect()
        
        # Create relative uncertainty intensity map
        fig, ax = plt.subplots(figsize=(size, size), dpi=300)
        try:
            plt.imshow(background_image)
            plot = plt.imshow(uncertainty, cmap = "jet_r", alpha = 0.7 )
        except NameError:
            plot = plt.imshow(uncertainty, cmap = "jet_r" )
            pass 
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.ax.tick_params(labelsize=fontsize*(0.7))
        plt.title(file)
        plt.xticks([])
        plt.yticks([])
        plt.title("Relative Scaled Log Probabilities", fontsize = fontsize*(0.7))
        plt.tight_layout()
        plt.savefig("Relative Scaled Log Probabilities" + ".png")
        plt.close(fig)
        gc.collect()
        
        fig, ax = plt.subplots(figsize=(size, size), dpi=300)
        ax = plt.subplot()
        sns.kdeplot( data = uncertainty.flatten() , 
                    lw = 6, 
                    log_scale = (False, False) )
        plt.xlabel("Element Intensity")
        plt.ylabel("Relative Frequency")
        plt.title('Histograms of Log Probabilities')
        plt.tight_layout()
        plt.savefig("Histograms of Log Probabilities" + ".png", dpi=300)
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
        if DPGMM.converged_ == False: 
            autodidactic_loop = False
            print("Analysis Halted due to no convergence")
            break 
        elif (residual_pixel_threshold > len(array[uncertainty <= autodidectic_threshold].flatten() )):
            autodidactic_loop = False
            print("Analysis Truncated Due to Residual Un-Identified Pixels")
            break 
        elif  (np.min(uncertainty) < autodidectic_threshold) and ( (generation + 1) < max_autodidectic_iterations):
            new_data = array[uncertainty <= autodidectic_threshold]
            new_data = random.sample(list(new_data), int( min(training_set_growth_rate, len(list(new_data))) ) ) 
            new_data = np.asarray(new_data)
            
            print("Current number of training points")
            print(str( training_data.shape[0] ))
            print("Number of unidentified points")
            print(str(new_data.shape[0]))
            print("Number of points appended")
            print(str(int(training_set_growth_rate ) ))
            training_data = np.append(training_data, new_data, axis = 0) 
            print("Number of training points after autodidactic loop")
            print(str( training_data.shape[0] ))
    
            generation += 1 
            number_of_classes += 5
            np.save('Generation ' + str(generation) + ' Training Data.npy', training_data, allow_pickle=False)
            gc.collect()
            continue
        else: 
            autodidactic_loop = False
            print("Analysis Complete")
            break 
        
    # Create final analysis folder 
    print("Beginning Plotting")
    os.mkdir("Analysis")
    os.chdir("Analysis")
    
    # Save updated model
    np.save('Final_Model_Weights', DPGMM.weights_, allow_pickle=False)
    np.save('Final_Model_Means', DPGMM.precisions_cholesky_, allow_pickle=False)
    np.save('Final_Model_Covariances', DPGMM.covariances_, allow_pickle=False)
    np.save('Log_Probabilities.npy', uncertainty, allow_pickle=False)
    components = len(DPGMM.weights_)
    
    size = 60
    fontsize = 60
    
    # Segment montage data       
    try: 
        segmentation = DPGMM.predict(full_data)
        gc.collect()
        print("Segmentation: 100%")

    except MemoryError:
        mid_point = round(full_data.shape[0] / 5.0)
        
        segmentation = DPGMM.predict( full_data[0:mid_point, :] )    
        print("Segmentation: 20%")
        
        segmentation1 = DPGMM.predict( full_data[mid_point: mid_point*2, :] )    
        segmentation = np.concatenate( (segmentation, segmentation1), axis = 0)
        del segmentation1
        gc.collect()
        print("Segmentation: 40%")

        segmentation1 = DPGMM.predict( full_data[mid_point*2: mid_point*3, :] )    
        segmentation = np.concatenate( (segmentation, segmentation1), axis = 0)
        del segmentation1
        gc.collect()
        print("Segmentation: 60%")

        segmentation1 = DPGMM.predict( full_data[mid_point*3: mid_point*4, :] )    
        segmentation = np.concatenate( (segmentation, segmentation1), axis = 0)
        del segmentation1
        gc.collect()
        print("Segmentation: 80%")

        segmentation1 = DPGMM.predict( full_data[mid_point*4: full_data.shape[0], :] )    
        segmentation = np.concatenate( (segmentation, segmentation1), axis = 0)
        del segmentation1
        gc.collect()
        print("Segmentation: 100%")

    segmentation = np.asarray(segmentation).reshape(array.shape[0], array.shape[1])
    np.save('Segmentation', segmentation, allow_pickle=False)
    
    # apply mask to hide pixels that sum to less than the eds_quality_threshold. 
    # this is used to remove pixels that are shaddowed, artifacts, etc. 
    mask = np.ones(image.shape, dtype = np.uint8)*255
    mask = np.ma.masked_where(np.sum(array, axis = 2) <= eds_quality_threshold, mask)
    
    fig, ax = plt.subplots(figsize=(size, size))
    try:
        plt.sca(ax)
        plot = plt.pcolormesh(background_image, alpha = 1, cmap = 'gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(plot, cax=cax)   
        cb.remove()
    except NameError:
        pass 
            
    plt.imshow(mask, cmap = 'brg', alpha = 0.7)
    plt.tight_layout()
    plt.savefig("EDS Quality Mask", bbox_inches = 'tight', dpi = 300)
    plt.close(fig)
    gc.collect()
   
    # Create agglomerative hierarchial model and save results 
    fig, ax = plt.subplots(figsize=(size, size))
    heir = sklearn.cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage = 'complete')
    clustering = heir.fit(DPGMM.means_)  
    linkage_matrix = plot_dendrogram(clustering, truncate_mode='level', p=100)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig("Consolidation Dendrogram" + ".png")
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
    
    for j, element in enumerate(image_list): 
        element_percents = []
        for i, class_id in enumerate(linkage_matrix['Class ID']): 
            element_percents.append(DPGMM.means_[int(class_id)][j] )
        linkage_matrix[str(element)] = element_percents
        
    linkage_matrix.to_excel("Class Data.xlsx")
    
    
    # Create absolute uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=300)
    try:
        plt.imshow(background_image)
        plot = plt.imshow(uncertainty, cmap = "jet_r", vmin = max(-50, autodidectic_threshold), vmax = 0, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = "jet_r", vmin = max(-50, autodidectic_threshold), vmax = 0 )
        pass 
    plt.xticks([])
    plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    plt.title(file)
    plt.xticks([])
    plt.yticks([])
    plt.title("Absolute Scaled Log Probabilities", fontsize = fontsize*(0.7))
    plt.tight_layout()
    plt.savefig("Absolute Scaled Log Probabilities with Colorbar" + ".png")
    plt.close(fig)
    gc.collect()
    
    
        # Create absolute uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=300)
    try:
        plt.imshow(background_image)
        plot = plt.imshow(uncertainty, cmap = "jet_r", vmin = max(-50, autodidectic_threshold), vmax = 0, alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = "jet_r", vmin = max(-50, autodidectic_threshold), vmax = 0 )
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.title(file)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("Absolute Scaled Log Probabilities without Colorbar" + ".png")
    plt.close(fig)
    gc.collect()
    
    
    # Create relative uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=300)
    try:
        plt.imshow(background_image)
        plot = plt.imshow(uncertainty, cmap = "jet_r", alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = "jet_r" )
        pass 
    plt.xticks([])
    plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    plt.title(file)
    plt.xticks([])
    plt.yticks([])
    plt.title("Relative Scaled Log Probabilities", fontsize = fontsize*(0.7))
    plt.tight_layout()
    plt.savefig("Relative Scaled Log Probabilities with Colorbar" + ".png")
    plt.close(fig)
    gc.collect()
    
    
    # Create relative uncertainty intensity map
    fig, ax = plt.subplots(figsize=(size, size), dpi=300)
    try:
        plt.imshow(background_image)
        plot = plt.imshow(uncertainty, cmap = "jet_r", alpha = 0.7 )
    except NameError:
        plot = plt.imshow(uncertainty, cmap = "jet_r" )
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("Relative Scaled Log Probabilities without Colorbar" + ".png")
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
    
    # Save semantic segmentation map 
    fig, ax = plt.subplots(figsize = (size, size)) 
    ax = plt.subplot()
    cmap = plt.get_cmap('nipy_spectral', np.max(segmentation)-np.min(segmentation)+1)
    plt.gca().invert_yaxis()
    try:
        plt.imshow(background_image)
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
    plt.title("Class Segmentation", fontsize = fontsize*(0.7))
    plt.savefig("Class Segmentation with Colorbar" + ".png", dpi=300)
    plt.close(fig)
    gc.collect()
    
    # Save semantic segmentation map 
    fig, ax = plt.subplots(figsize = (size, size)) 
    ax = plt.subplot()
    cmap = plt.get_cmap('nipy_spectral', np.max(segmentation)-np.min(segmentation)+1)
    plt.gca().invert_yaxis()
    try:
        plt.imshow(background_image)
        plot = plt.imshow(segmentation, cmap=cmap, alpha = 0.7, vmin = np.min(segmentation)-.5, vmax = np.max(segmentation)+.5)
    except NameError:
        plot = plt.imshow(segmentation, cmap=cmap, alpha = 1, vmin = np.min(segmentation)-.5, vmax = np.max(segmentation)+.5)
        pass 
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("Class Segmentation without Colorbar" + ".png", dpi=300)
    plt.close(fig)
    gc.collect()
    
    
    # Create binary class maps and save 
    w = (-DPGMM.weights_).argsort()

    for ind2 in w:
        
        # Ignore extraneous classes that are not in the segmentation 
        if ind2 in uniques: 
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ratio*size, size), dpi=300)
            stdev = np.sqrt(np.abs(DPGMM.covariances_[ind2]))
            stdev[DPGMM.covariances_[ind2] < 0.0] = -1 * stdev[DPGMM.covariances_[ind2] < 0.0] 
            annot = np.diag(DPGMM.precisions_cholesky_[ind2],0)
            annot = np.round(annot,2)
            annot = annot.astype('str')
            annot[annot=='0.0']=''
            sns.heatmap(correlation_from_covariance(DPGMM.covariances_[ind2]), xticklabels = image_list, yticklabels = [ str(element) + ": " + str( round(DPGMM.means_[ind2][i], 2) ) for i, element in enumerate(image_list) ], center = 0, vmin = 0, vmax = 1, linewidths=1, linecolor = 'white', cmap = 'bwr', mask = np.triu(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'horizontal'})
            ax2.tick_params(axis='x', pad=5)
            ax2.set_yticklabels( ax2.get_yticklabels(), rotation=0)
            
            for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                label.set_fontsize(60)
                        
            plt.sca(ax1)
            plot = plt.pcolormesh( np.ma.masked_where(segmentation != ind2, np.ones(segmentation.shape, dtype = np.uint8)), cmap = 'brg', alpha = 1)        
            plt.xticks([])
            plt.yticks([])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(plot, cax=cax)   
            cb.ax.tick_params(labelsize=60) 
            cb.remove()
            ax1.invert_yaxis()
            
            try:
                plt.sca(ax1)
                plot = plt.pcolormesh(background_image, alpha = 0.2, cmap = 'gray')
            except NameError:
                pass 
            
            plt.tight_layout()
            plt.savefig("Class " + str(ind2) + " With Covariance Table.png")
            plt.close(fig)
            gc.collect()
        
        
            fig, (ax1) = plt.subplots(1, 1, figsize=(size, size), dpi=300)
    
            plt.sca(ax1)
            plot = plt.pcolormesh( np.ma.masked_where(segmentation != ind2, np.ones(segmentation.shape, dtype = np.uint8)), cmap = 'brg', alpha = 1)
            plt.xticks([])
            plt.yticks([])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(plot, cax=cax)   
            cb.remove()
            ax1.invert_yaxis()
            
            try:
                plt.sca(ax1)
                plot = plt.pcolormesh(background_image, alpha = 0.2, cmap = 'gray')
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(plot, cax=cax)   
                cb.remove()
            except NameError:
                pass 
            plt.tight_layout()
            plt.savefig("Class " + str(ind2) + " Without Covariance Table.png") 
            plt.close(fig)
            gc.collect()
            
            
            
            
            
    
    print("Number of unique classes: " + str(len(list(np.unique(segmentation)))))
    print(" ")
    print("Plotting Complete")
    print(" ")
    print("Results located in ")
    print(" ")
    print(str(output_src) ) 

    if (operating_mode == 3):
        # Now that the first row in the queue file has been read, delete the entry
        queue = queue.tail(queue.shape[0] -1)
        queue.to_csv(queue_filepath, index=None)
        
    else: 
        run = False 
        raise SystemExit 


"""
full_data = np.load("Filtered Array.npy")
#training_data = np.load("Initial Training Data.npy")
covariances = np.load("Final_Model_Covariances.npy")
means = np.load("Final_Model_Means.npy")
weights = np.load("Final_Model_Weights.npy")
precisions_cholesky = np.linalg.cholesky(np.linalg.inv(covariances))
segmentation = np.load("Segmentation.npy")
uncertainty = np.load("Log_Probabilities.npy")

DPGMM.covariances_ = covariances 
DPGMM.means_ = means 
DPGMM.weights_ = weights 
DPGMM.precisions_cholesky_ = precisions_cholesky 






















"""

