


##############################################################################################################
# import necessary packages 
##############################################################################################################


import matplotlib
matplotlib.use('TkAgg')

import time

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
#plt.ion()
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
    montage_paths = filedialog.askopenfilenames( title = "Select montage files(s). Press 'cancel' to stop adding folders")
    if montage_paths != '':
        for path in montage_paths:
            montage_list.append(path)
    else:
        break
          
widths = []
offsets = []
sums = []
highest = [] 
bins = [] 
# extract calibration parameters, sum of spectrum and highest intensity spectrum 
for montage_path in montage_list: 
    with h5py.File(montage_path, 'r+') as file:     
        channel_width = file['Metadata']['EDS Voltage Bin Width (eV)'][...]
        temp = [ float(x.decode('utf-8') ) for x in channel_width]
        channel_width = statistics.mean( temp.copy() )
        
        channel_offset = file['Metadata']['EDS Starting Bin Voltage (eV)'][...]
        temp = [ float(x.decode('utf-8') ) for x in channel_offset]
        channel_offset = statistics.mean( temp.copy() )
        
        sum_spectrum = file['EDS']['Sum of Spectrum'][...]
        highest_spectrum = file['EDS']['Highest Intensity Spectrum'][...]
        peak_bins = file['EDS']['Autodetected Peak Bins'][...]

    # verify that the calibrations are correct
    # Oxford in particular does not appear to export the correct calibration for EDS datasets
    # but rather the most recent calibration - which may or may not be appropriate 
    print(os.path.basename(montage_path))
    print("")
    
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
    
    # append values to lists
    # each list entry is for a unique montage 
    widths.append(channel_offset)
    offsets.append(channel_width)
    sums.append(sum_spectrum)
    highest.append(highest_spectrum)
    bins.append(peak_bins)
    
########
search_width = 3
element_search_half_distance = 0.2 # +- distance to search for x-ray band matches from peaks. Units are KeV 

auto_suggest_filters = ['Ka']
element_filters = [] 
distance = 3
    

temp = list( bins[0].copy() ) 
highest_spectrum = highest[0].copy() 
sum_spectrum = sums[0].copy()

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
            
                        
            
            
            #tellme('You will define a triangle, click to begin')
            
            fontsize = 12
            
            fig, ( ax2, ax3, ax4) = plt.subplots(3, 1, figsize = (14, 10 ) )
            
            #plt.clf()
            #plt.setp(plt.gca(), autoscale_on=False)

            
            fig.suptitle('Potential Peak at ' + str( round( (channel_offset + channel_width*peak)/1_000.0, 2 )) + "KeV", fontsize=fontsize)
            fig.tight_layout()
            #plt.subplots_adjust(top=0.85)
            
            
            
            
            plt.sca(ax2)
            plt.plot((channel_offset + channel_width*np.linspace(0,len(sum_spectrum), num=len(sum_spectrum)))/1_000.0, highest_spectrum, color = "black")

            plt.scatter(x = (channel_offset + channel_width*np.asarray(temp))/1_000.0, y = highest_spectrum[temp], s = 75, alpha = 1)

            plt.scatter(x = (channel_offset + channel_width*max_x)/1_000.0, y = highest_spectrum[ max_x ], color = 'red', s = 150, alpha = 1)
            plt.title("Sum of Spectrum of Entire Dataset - Suggested Peaks", fontsize=fontsize)
            plt.tight_layout()
            plt.xlabel("KeV", fontsize=fontsize)
            plt.ylabel("X-Ray Counts", fontsize=fontsize)
            plt.xticks(fontsize= fontsize)
            plt.yticks(fontsize= fontsize)
            
            
            #ax2.tick_params(which = 'major', length = 25, bottom = True, left = True)
            #ax2.tick_params(which = 'minor', axis = 'x', length = 15, bottom = True, left = True)
            
            ax2.minorticks_on()
            
            #ax2.xaxis.set_major_locator(MultipleLocator(5))
            #ax2.xaxis.set_minor_locator(tck.AutoMinorLocator())
                 
            
            
            plt.sca(ax3)
            plt.scatter(x = (channel_offset + channel_width*np.asarray(temp))/1_000.0, y = sum_spectrum[temp])

            plt.scatter(x = (channel_offset + channel_width*max_x)/1_000.0, y = sum_spectrum[ max_x ], color = 'red')
            plt.plot((channel_offset + channel_width*np.linspace(0,len(sum_spectrum), num=len(sum_spectrum)))/1_000.0, sum_spectrum, color = "black")
 
         
            plt.title("Sum of Spectrum of Entire Dataset - Suggested Peaks", fontsize=fontsize)
            plt.tight_layout()
            plt.xlabel("KeV", fontsize=fontsize)
            plt.ylabel("X-Ray Counts", fontsize=fontsize)
            plt.xticks(fontsize= fontsize)
            plt.yticks(fontsize= fontsize)
            #ax3.tick_params(which = 'major', length = 25, bottom = True, left = True)
            #ax3.tick_params(which = 'minor', axis = 'x', length = 15, bottom = True, left = True)
            
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
            plt.scatter(x = (channel_offset + channel_width*max_x)/1_000.0, y = sum_spectrum[ max_x ], color = 'red')
            plt.plot((channel_offset + channel_width*np.linspace(0,len(sum_spectrum), num=len(sum_spectrum)))/1_000.0, sum_spectrum, color = "black")
            plt.title("Sum of Spectrum of Entire Dataset - User Specified Peaks", fontsize=fontsize)
            
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
            plt.xlabel("KeV", fontsize=fontsize)
            plt.ylabel("X-Ray Counts", fontsize=fontsize)
            plt.xticks(fontsize= fontsize)
            plt.yticks(fontsize= fontsize)
            #ax4.tick_params(which = 'major', length = 25, bottom = True, left = True)
            #ax4.tick_params(which = 'minor', axis = 'x', length = 15, bottom = True, left = True)
            
            ax4.minorticks_on()
            
            ax4.xaxis.set_major_locator(MultipleLocator(5))
            ax4.xaxis.set_minor_locator(tck.AutoMinorLocator())
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.draw()
            
            while True:
                if plt.waitforbuttonpress():
                    break
    
            #plt.waitforbuttonpress()
            
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
    
    
    
    












