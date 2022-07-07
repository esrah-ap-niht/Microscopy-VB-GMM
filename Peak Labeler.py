


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

root = Tk()
root.withdraw()
root.attributes('-topmost',1)

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
          
widths = []
offsets = []
sums = []
highest = [] 
search_width = 3
element_search_half_distance = 0.2 # +- distance to search for x-ray band matches from peaks. Units are KeV 
auto_suggest_filters = ['Ka']
element_filters = [] 
distance = 3

########
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
            file['Metadata']['EDS Starting Bin Voltage (eV)'][...] = channel_offset
        print("")
            
        print("Autodetected EDS Channel Width is: " + str(channel_width) + " eV")
        print("Enter optional replacement value in eV and press ENTER, otherwise leave field blank and press ENTER to keep autodetected value")
        new = input()
        if new != "": 
            channel_width = new
            file['Metadata']['EDS Voltage Bin Width (eV)'][...] = channel_width
        print("")
        
        # append values to lists
        # each list entry is for a unique montage 
        widths.append(channel_width)
        offsets.append(channel_offset)
        sums.append(sum_spectrum)
        highest.append(highest_spectrum)

########
with h5py.File(analysis_file, 'r+') as file:     
    peak_energies = file['Channel KeV Peaks'][...]  # KeV units 
    
########
for z, montage_path in enumerate(montage_list): 
    
    channel_width = widths[z]               # eV units 
    channel_offset = offsets[z]             # eV units
    temp = list( peak_energies.copy() ) 
    highest_spectrum = highest[z].copy()    # KeV units 
    sum_spectrum = sums[z].copy()           # KeV units 
    new_command = ''
    bins = [round(x) for x in (peak_energies*1_000.0 - channel_offset)/channel_width ] 
    
    while True:      
            
        if len(temp) == 0:
            break 
        if new_command == 'exit':
            break
        
        for i, peak in enumerate(temp):  
            if new_command == 'exit':
                break
            
            while True: 
                
                if new_command == 'exit':
                    break
                
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
                
                fig, ( ax2, ax3, ax4) = plt.subplots(3, 1, figsize = (14, 8 ) )
                
                #plt.clf()
                #plt.setp(plt.gca(), autoscale_on=False)
    
                
                fig.suptitle('Potential Peak at ' + str( round( peak, 2 )) + "KeV", fontsize=fontsize)
                fig.tight_layout()
                #plt.subplots_adjust(top=0.85)
                
                
                
                
                plt.sca(ax2)
                plt.plot((channel_offset + channel_width*np.linspace(0,len(sum_spectrum), num=len(sum_spectrum)))/1_000.0, highest_spectrum, color = "black")
    
                plt.scatter(x = temp, y = highest_spectrum[bins], s = 75, alpha = 1)
    
                plt.scatter(x = peak, y = highest_spectrum[ bins[i] ], color = 'red', s = 150, alpha = 1)
                plt.title("Maximum Bin Intensity of Entire Dataset - Suggested Peaks", fontsize=fontsize)
                plt.tight_layout()
                plt.xlabel("KeV", fontsize=fontsize)
                plt.ylabel("X-Ray Counts", fontsize=fontsize)
                plt.xticks(fontsize= fontsize)
                plt.yticks(fontsize= fontsize)
                
                
                ax2.tick_params(which = 'major', length = 10, width = 3, axis = 'x', bottom = True, left = True)
                ax2.tick_params(which = 'minor', length = 5, width = 3, axis = 'x', bottom = True, left = True)
                
                ax2.minorticks_on()
                
                #ax2.xaxis.set_major_locator(MultipleLocator(5))
                #ax2.xaxis.set_minor_locator(tck.AutoMinorLocator())
                     
                
                
                plt.sca(ax3)
                plt.scatter(x = temp, y = sum_spectrum[bins],  s = 75, alpha = 1)
    
                plt.scatter(x = peak, y = sum_spectrum[ bins[i] ], color = 'red',  s = 150)
                plt.plot((channel_offset + channel_width*np.linspace(0,len(sum_spectrum), num=len(sum_spectrum)))/1_000.0, sum_spectrum, color = "black")
     
             
                plt.title("Sum of Spectrum of Entire Dataset - Suggested Peaks", fontsize=fontsize)
                plt.tight_layout()
                plt.xlabel("KeV", fontsize=fontsize)
                plt.ylabel("X-Ray Counts", fontsize=fontsize)
                plt.xticks(fontsize= fontsize)
                plt.yticks(fontsize= fontsize)
                
                ax3.tick_params(which = 'major', length = 10, width = 3, axis = 'x', bottom = True, left = True)
                ax3.tick_params(which = 'minor', length = 5, width = 3, axis = 'x', bottom = True, left = True)
                
                ax3.minorticks_on()
                
                ax3.xaxis.set_major_locator(MultipleLocator(5))
                ax3.xaxis.set_minor_locator(tck.AutoMinorLocator())
                
                auto_suggest_elements = hs.eds.get_xray_lines_near_energy(peak, only_lines = auto_suggest_filters, width = 0.5)
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
                plt.scatter(x = peak, y = sum_spectrum[ bins[i] ], color = 'red',  s = 150)
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
                ax4.tick_params(which = 'major', length = 10, width = 3, axis = 'x', bottom = True, left = True)
                ax4.tick_params(which = 'minor', length = 5, width = 3, axis = 'x', bottom = True, left = True)
                
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
                print("Enter 'exit' to discontinue labeling peaks")
                new_command = input("")
                plt.close(fig)
                
                if (new_command == 'end') or (new_command == 'exit'): 
                    print("Continuing to next peak")
                    break 
                elif new_command == 'z': 
                    temp.remove(peak)
                    bins.remove( bins[i] )
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
                
                
###########     
#element_filters.sort()
energy_locs = list(peak_energies)

element_energies = []
element_shells = []
display_shells = ["" for n in range(len(peak_energies) )]

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

with h5py.File(analysis_file, 'r+') as file: 
    try:
        file.create_dataset( 'Autodetected Peak Labels' , data = display_shells)
    except:
        
        try: 
            file['Autodetected Peak Labels'][...] = display_shells
        except: 
            pass 
          
            
            
            
            
            
            
            
        




