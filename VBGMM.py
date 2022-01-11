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

gc.enable()

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




    
# Start of program 
generation = 0
global full_data
global DPGMM
global initial_weights
ratio = 2
color = 'gist_rainbow'
plt.ioff()
root = Tk()
root.withdraw()
sns.set(font_scale = 3)
sns.set_style("white")
size = 30
fontsize = 30

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
print(' ')
old_score = -1_000_000_000
autodidactic_loop = True 

# Get user to select input directory 
input_src = filedialog.askdirectory( title = "Select input directory. Multispectral files should be located in a single folder and TIFF, PNG, or JPG formats")
    
# Get user to select optinal background image (e.g. SEM or BSE)
background_filepath = filedialog.askopenfilename( title = "Select optional background image such as Secondary, Backscatter or EBSD Quality Index")

# Get user to select output directory 
output_src = filedialog.askdirectory( title = "Select output directory")

# Get list of files and screen for valid file types  
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
    
for i in range(array.shape[2]):
    #array[:,:,i] = np.ma.masked_where(np.sum(array, axis = 2) <= eds_quality_threshold, array[:, :, i])
    array[:, :, i][np.sum(array, axis = 2) <= eds_quality_threshold] = 0

# Save preprocessed data 
np.save('Filtered Array.npy', array.data, allow_pickle=False)

print("Preprocessing complete")
print(" ")
print("Data read, filtered, and saved to NPY array")
print(" ")

full_data = array.flatten().reshape(array.shape[0] * array.shape[1], len(image_list))

print("Beginning Analysis")
# Sample randomly from the entire montage
training_data =  list(array.flatten().reshape(array.shape[0] * array.shape[1], len(image_list))[:,:] )     
training_data = random.sample(training_data, min(int(initial_sampling_size), full_data.shape[0] ) )
training_data = np.asarray(training_data)
np.save('Initial Training Data.npy', training_data, allow_pickle=False)

while autodidactic_loop == True: 
    # Train VB-GMM model on training subset
    DPGMM = mixture.BayesianGaussianMixture(n_components = number_of_classes, 
                                                    max_iter = 100000000,
                                                    n_init = number_of_initializations,
                                                    tol = EM_tolerance, 
                                                    init_params='random', 
                                                    weight_concentration_prior_type='dirichlet_process',
                                                    weight_concentration_prior = 1/number_of_classes, 
                                                    verbose = True)
    DPGMM.fit(training_data)
    
    # Reweight model classes based on entire dataset 
    initial_weights = DPGMM.weights_
    cons = {'type':'eq', 'fun': constraint}
    boundaries = []
    
    for i in range(len(initial_weights) ):
        boundaries.append( (0,0.999_999) ) 
        
    response = scipy.optimize.minimize(GMM_weights, initial_weights, tol = 1e-8, constraints = cons, bounds = boundaries, options = {'disp': True, 'maxiter' :1000} )
    DPGMM.weights_ = response.x

    # Save model
    gmm_name = 'Generation ' + str(generation) 
    os.mkdir(gmm_name)
    os.chdir(gmm_name)
    np.save(gmm_name + '_weights', DPGMM.weights_, allow_pickle=False)
    np.save(gmm_name + '_means', DPGMM.means_, allow_pickle=False)
    np.save(gmm_name + '_covariances', DPGMM.covariances_, allow_pickle=False)

    # Create uncertainty intensity map     
    uncertainty = DPGMM.score_samples( full_data )
    uncertainty = np.asarray(uncertainty).reshape(array.shape[0], array.shape[1])
    np.save('Log_Probabilities.npy', uncertainty, allow_pickle=False)

    # Return up one directory level 
    os.chdir( os.path.dirname(os.getcwd()) )
    
    fig, ax = plt.subplots(figsize=(size, size))
    plot = plt.imshow(uncertainty, cmap = color, vmin = max(-50, autodidectic_threshold), vmax = 0 )
    plt.xticks([])
    plt.yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize*(0.7))
    plt.title("Log Probability", size = fontsize*(1) )
    plt.xticks([])
    plt.yticks([])
    string = "Generation" + str(generation) + " Log Probabilities" + ".png"
    plt.savefig(string, bbox_inches = 'tight')
    plt.close(fig)
    
    # Due to scipy limtations, we must create a second 'plain' GMM to calculate the AIC or BIC scores 
    GMM = mixture.GaussianMixture(n_components = number_of_classes)
    GMM.weights_ = DPGMM.weights_
    GMM.means_ = DPGMM.means_
    GMM.covariances_ = DPGMM.covariances_
    GMM.precisions_cholesky_ = DPGMM.precisions_cholesky_

    # Once the model tuning is complete, append new training data if necessary
    if  (np.min(uncertainty) < autodidectic_threshold) and ( (generation + 1) < max_autodidectic_iterations):
        new_data = array[uncertainty <= autodidectic_threshold]
        new_data = random.sample(list(new_data), min(new_data.shape[0], int(training_set_growth_rate ) ) )
        new_data = np.asarray(new_data)
        
        print("Current number of training points")
        print(str( training_data.shape[0] ))
        print("Number of points appended")
        print(str(new_data.shape[0]))
        training_data = np.append(training_data, new_data, axis = 0) 
        print("Number of training points after autodidactic loop")
        print(str( training_data.shape[0] ))
 
        generation += 1 
        number_of_classes += 5
        old_score = -1_000_000_000
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
np.save('Final_Model_Means', DPGMM.means_, allow_pickle=False)
np.save('Final_Model_Covariances', DPGMM.covariances_, allow_pickle=False)
np.save('Log_Probabilities.npy', uncertainty, allow_pickle=False)
components = len(DPGMM.weights_)

size = 60
fontsize = 60

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
plt.savefig("Consolidation Dendrogram" + ".png")
plt.close(fig)
gc.collect()

# Create uncertainty intensity map
fig, ax = plt.subplots(figsize=(size, size), dpi=300)
try:
    plt.imshow(background_image)
    plot = plt.imshow(uncertainty, cmap = color, vmin = max(-50, autodidectic_threshold), vmax = 0, alpha = 0.7 )
except NameError:
    plot = plt.imshow(uncertainty, cmap = color, vmin = max(-50, autodidectic_threshold), vmax = 0 )
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
plt.title("Log Probabilities", fontsize = fontsize*(0.7))
plt.savefig("Log Probabilities" + ".png")
plt.close(fig)
gc.collect()

# Save cluster weight graph
fig, ax = plt.subplots(figsize=(size, size))
plot_w = np.arange(components) + 1
ax.bar(plot_w - 0.5, np.sort(DPGMM.weights_)[::-1], width=1., lw=0);
ax.set_xlim(0.5, components);
ax.set_xlabel('Number of Classes');
ax.set_ylabel('Posterior expected mixture weight');
fig.suptitle("Mixture Weight per Class" ,fontsize=20 )
plt.savefig("Mixture Weights" + ".png", dpi=300)
plt.close(fig)
gc.collect()

# Segment montage data 
segmentation = DPGMM.predict(full_data)
segmentation = np.asarray(segmentation).reshape(array.shape[0], array.shape[1])
np.save('Segmentation', segmentation, allow_pickle=False)


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
plt.title("Class Segmentation")
plt.savefig("Class Segmentation" + ".png", dpi=300)
plt.close(fig)
gc.collect()

# Save a summation of elemental intensities. Useful for highlighting pores and other areas that did not emit XRays
fig, ax = plt.subplots(figsize=(size, size), dpi=300)
ax = plt.subplot()
try:
    plt.imshow(background_image)
    plot = plt.imshow(np.sum(array, axis = 2), cmap = color, alpha = 0.7 ) 
except NameError:
    plot = plt.imshow(np.sum(array, axis = 2), cmap = color ) 
    pass 
plt.title('Sum of Elemental Intensities')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(plot, cax=cax)
plt.savefig("Sum of Elemental Intensities" + ".png", dpi=300)
plt.close(fig)
gc.collect()

# Create binary class maps and save 
w = (-DPGMM.weights_).argsort()

for ind2 in w:
    
    # Ignore extraneous classes that are not in the segmentation 
    if ind2 in segmentation: 
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ratio*size, size), dpi=300)
        stdev = np.sqrt(np.abs(DPGMM.covariances_[ind2]))
        stdev[DPGMM.covariances_[ind2] < 0.0] = -1 * stdev[DPGMM.covariances_[ind2] < 0.0] 
        annot = np.diag(DPGMM.means_[ind2],0)
        annot = np.round(annot,2)
        annot = annot.astype('str')
        annot[annot=='0.0']=''
        sns.heatmap(stdev, xticklabels = image_list, yticklabels = image_list, center = 0, linewidths=.5, robust = True, annot = annot, fmt='', linecolor = 'black' ,cmap = 'PRGn', cbar_kws={'label': 'Sqrt(Covariance)', 'orientation': 'horizontal'})
        sns.heatmap(correlation_from_covariance(DPGMM.covariances_[ind2]), xticklabels = image_list, yticklabels = image_list,center = 0, vmin = -1, vmax = 1, linewidths=.5, linecolor = 'black', cmap = 'seismic', mask = np.tril(stdev), cbar_kws={'label': 'Correlation', 'orientation': 'horizontal'})
        ax2.tick_params(axis='x', pad=5)
        ax2.set_yticklabels( ax2.get_yticklabels(), rotation=0)
               
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
        
        fig.suptitle("Class: " + str(ind2), size = fontsize)
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
        fig.suptitle("Class: " + str(ind2), size = fontsize)
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







