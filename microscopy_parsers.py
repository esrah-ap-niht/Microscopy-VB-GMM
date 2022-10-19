
# import necessary packages 
import pandas as pd
import numpy as np
import os 
import gc
from tkinter import filedialog
from tkinter import *
import h5py
import hyperspy.api as hs
from pathlib import Path
import math
from tqdm import tqdm
import glob
import sys
import scipy
import statistics 

##############################################################################################################
# garbage collection and tkinter settings 
##############################################################################################################
gc.enable()
root = Tk()
root.withdraw()
root.attributes('-topmost',1)


def stitch(array, block, x_size, y_size, x_location, y_location):
    array[y_location:y_location + y_size, x_location:x_location + x_size] = block
    return array


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
                file[h5_path].create_dataset(array_or_attribute[1], data = data, chunks=True, dtype = 'float64')
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


def oxford_parse_h5(file_path):
    # Input a single file address to parse for metadata. 
    # Returns a dataframe and dictionary - metadata and EBSD phases 
    ############################################################################################################################
    
    # check that the file is inded HDF5 or another compatable format 
    try: 
        file = h5py.File(os.path.join(file_path), "r")
    except OSError: 
        print("Error. Selected file type is incompatable with HF5 format")
        sys.exit()
    else: 
        pass 
        
    ##########
    # collect dataset-wide metadata 
    try: 
        hardware_vendor =                       str(file['Manufacturer'][...][0] ).split("'")[1::2][0]
        software_version =                      str(file['Software Version'][...][0] ).split("'")[1::2][0]
        format_version =                        str(file['Format Version'][...][0] ).split("'")[1::2][0]
        
        # Oxford does not store the montage labels inside their H5 files, but rather in the file name
        # to extract the montage label, one must remove the project, site, speimen, and field information that is also in the filename 
        if "Electron Image" in file['1'].keys(): 
            try: 
                
                project_name =                      str( file['1']['Electron Image']['Header']['Project Label'][...][0] ).split("'")[1::2][0]
                site_name =                         str( file['1']['Electron Image']['Header']['Site Label'][...][0] ).split("'")[1::2][0]
                specimen_name =                     str( file['1']['Electron Image']['Header']['Specimen Label'][...][0] ).split("'")[1::2][0]
                field_name =                        str( file['1']['Electron Image']['Header']['Analysis Label'][...][0] ).split("'")[1::2][0]        
                
                montage_name = Path(file_path).stem
                
                #montage_name = montage_name.removeprefix(project_name)
                montage_name = montage_name.replace(project_name, '', 1)
                montage_name = montage_name.strip()
                
                #montage_name = montage_name.removeprefix(specimen_name)
                montage_name = montage_name.replace(specimen_name, '', 1)
                montage_name = montage_name.strip()


                #montage_name = montage_name.removesuffix(field_name)
                montage_name = montage_name.replace(field_name, '', 1)
                montage_name = montage_name.strip()
                
                #montage_name = montage_name.removesuffix(site_name)
                montage_name = montage_name.replace(site_name, '', 1)
                montage_name = montage_name.strip()

            except KeyError: 
                print("ERROR: Unable to access the electron image folder to parse montage label")
                
        elif "EDS" in file['1'].keys(): 
            try: 
                
                project_name =                      str( file['1']['EDS']['Header']['Project Label'][...][0] ).split("'")[1::2][0]
                site_name =                         str( file['1']['EDS']['Header']['Site Label'][...][0] ).split("'")[1::2][0]
                specimen_name =                     str( file['1']['EDS']['Header']['Specimen Label'][...][0] ).split("'")[1::2][0]
                field_name =                        str( file['1']['EDS']['Header']['Analysis Label'][...][0] ).split("'")[1::2][0]        
                
                montage_name = Path(file_path).stem
                
                #montage_name = montage_name.removeprefix(project_name)
                montage_name = montage_name.replace(project_name, '', 1)
                montage_name = montage_name.strip()
                
                #montage_name = montage_name.removeprefix(specimen_name)
                montage_name = montage_name.replace(specimen_name, '', 1)
                montage_name = montage_name.strip()


                #montage_name = montage_name.removesuffix(field_name)
                montage_name = montage_name.replace(field_name, '', 1)
                montage_name = montage_name.strip()
                
                #montage_name = montage_name.removesuffix(site_name)
                montage_name = montage_name.replace(site_name, '', 1)
                montage_name = montage_name.strip()
                
            except KeyError: 
                print("ERROR: Unable to access the EDS folder to parse montage label")
                
        elif "EBSD" in file['1'].keys(): 
            try: 
                
                project_name =                      str( file['1']['EBSD']['Header']['Project Label'][...][0] ).split("'")[1::2][0]
                site_name =                         str( file['1']['EBSD']['Header']['Site Label'][...][0] ).split("'")[1::2][0]
                specimen_name =                     str( file['1']['EBSD']['Header']['Specimen Label'][...][0] ).split("'")[1::2][0]
                field_name =                        str( file['1']['EBSD']['Header']['Analysis Label'][...][0] ).split("'")[1::2][0]        
                
                montage_name = Path(file_path).stem
                
                #montage_name = montage_name.removeprefix(project_name)
                montage_name = montage_name.replace(project_name, '', 1)
                montage_name = montage_name.strip()
                
                #montage_name = montage_name.removeprefix(specimen_name)
                montage_name = montage_name.replace(specimen_name, '', 1)
                montage_name = montage_name.strip()


                #montage_name = montage_name.removesuffix(field_name)
                montage_name = montage_name.replace(field_name, '', 1)
                montage_name = montage_name.strip()
                
                #montage_name = montage_name.removesuffix(site_name)
                montage_name = montage_name.replace(site_name, '', 1)
                montage_name = montage_name.strip()
                
            except KeyError: 
                print("ERROR: Unable to access the EBSD folder to parse montage label")
        else: 
            montage_name = ''
        
    except: 
        print("ERROR: Unable to parse data-set level metadata")
        
    ##########
    # if available, collect SE/BSE metadata
    try:    
        electron_image_project =                str( file['1']['Electron Image']['Header']['Project Label'][...][0] ).split("'")[1::2][0]
        electron_image_specimen =               str( file['1']['Electron Image']['Header']['Specimen Label'][...][0] ).split("'")[1::2][0]
        electron_image_site =                   str( file['1']['Electron Image']['Header']['Site Label'][...][0] ).split("'")[1::2][0]
        electron_image_field =                  str( file['1']['Electron Image']['Header']['Analysis Label'][...][0] ).split("'")[1::2][0]        
        electron_image_voltage =                float( file['1']['Electron Image']['Header']['Beam Voltage'][...] )
        electron_image_magnification =          float( file['1']['Electron Image']['Header']['Magnification'][...] )
        electron_image_unique_vendor_id =       str(file['1']['Electron Image']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0]
        electron_image_drift_correction =       str(file['1']['Electron Image']['Header']['Drift Correction'][...] )
        electron_image_dwell_time =             float( file['1']['Electron Image']['Header']['Dwell Time'][...] )
        electron_image_average_frames =         int( file['1']['Electron Image']['Header']['Number Frames Averaged'][...] )
        electron_image_x_cells =                int(file['1']['Electron Image']['Header']['X Cells'][...] )
        electron_image_y_cells =                int(file['1']['Electron Image']['Header']['Y Cells'][...] )
        electron_image_x_step =                 float(file['1']['Electron Image']['Header']['X Step'][...] )
        electron_image_y_step =                 float(file['1']['Electron Image']['Header']['Y Step'][...] )
        electron_image_date =                   str( file['1']['Electron Image']['Header']['Acquisition Date'][...] ).split("'")[1::2][0]
        electron_image_x_pos =                  float(file['1']['Electron Image']['Header']['Stage Position']['X'][...] )
        electron_image_y_pos =                  float(file['1']['Electron Image']['Header']['Stage Position']['Y'][...] )
        electron_image_z_pos =                  float(file['1']['Electron Image']['Header']['Stage Position']['Z'][...] )
        electron_image_tilt_radians  =          float(file['1']['Electron Image']['Header']['Stage Position']['Tilt'][...] )
        electron_image_rotation_radians =       float(file['1']['Electron Image']['Header']['Stage Position']['Rotation'][...] )
        electron_image_working_distance =       float(file['1']['Electron Image']['Header']['Working Distance'][...] )

    except: 
        electron_image_project =                np.nan 
        electron_image_specimen =               np.nan  
        electron_image_site =                   np.nan 
        electron_image_field =                  np.nan 
        electron_image_voltage =                np.nan 
        electron_image_magnification =          np.nan 
        electron_image_unique_vendor_id =       np.nan 
        electron_image_drift_correction =       np.nan 
        electron_image_dwell_time =             np.nan 
        electron_image_average_frames =         np.nan 
        electron_image_x_cells =                np.nan 
        electron_image_y_cells =                np.nan 
        electron_image_x_step =                 np.nan 
        electron_image_y_step =                 np.nan 
        electron_image_date =                   np.nan 
        electron_image_x_pos =                  np.nan 
        electron_image_y_pos =                  np.nan 
        electron_image_z_pos =                  np.nan 
        electron_image_tilt_radians  =          np.nan 
        electron_image_rotation_radians =       np.nan 
        electron_image_working_distance =       np.nan 

    ##########
    # if available, collect EDS metadata
    try:   
        eds_project =                           str( file['1']['EDS']['Header']['Project Label'][...][0] ).split("'")[1::2][0]
        eds_specimen =                          str( file['1']['EDS']['Header']['Specimen Label'][...][0] ).split("'")[1::2][0]
        eds_site =                              str( file['1']['EDS']['Header']['Site Label'][...][0] ).split("'")[1::2][0]
        eds_voltage =                           float( file['1']['EDS']['Header']['Beam Voltage'][...] )
        eds_magnification =                     float( file['1']['EDS']['Header']['Magnification'][...] )
        eds_field =                             str( file['1']['EDS']['Header']['Analysis Label'][...][0] ).split("'")[1::2][0]  
        eds_binning =                           int( file['1']['EDS']['Header']['Binning'][...] )
        eds_bin_width =                         float( file['1']['EDS']['Header']['Channel Width'][...] )
        eds_start_channel =                     float( file['1']['EDS']['Header']['Start Channel'][...] )
        eds_averaged_frames =                   int( file['1']['EDS']['Header']['Number Frames'][...] )
        eds_process_time =                      int( file['1']['EDS']['Header']['Process Time'][...] ) 
        eds_x_cells =                           int(file['1']['EDS']['Header']['X Cells'][...] )
        eds_y_cells =                           int(file['1']['EDS']['Header']['Y Cells'][...] )
        eds_x_step =                            float(file['1']['EDS']['Header']['X Step'][...] )
        eds_y_step =                            float(file['1']['EDS']['Header']['Y Step'][...] )
        eds_real_time_sum =                     np.sum( file['1']['EDS']['Data']['Real Time'][...] )        # note: real time is the actual amount of time used to collect the data in a given pixel
        eds_live_time =                         np.average(file['1']['EDS']['Data']['Live Time'][...] )      # note: live time is a singular value that is constant for a dataset. It is unclear why the vendor stores an array of values rather than just one value. 
        eds_unique_vendor_id =                  str(file['1']['EDS']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0]
        eds_detector_azimuth_radians =          float(file['1']['EDS']['Header']['Detector Azimuth'][...] )
        eds_detector_elevation_radians =        float(file['1']['EDS']['Header']['Detector Elevation'][...] )
        eds_hardware_detector_serial_number =   str(file['1']['EDS']['Header']['Detector Serial Number'][...][0] ).split("'")[1::2][0]
        eds_hardware_detector_model =           str(file['1']['EDS']['Header']['Detector Type Id'].attrs['Name'] )
        eds_drift_correction =                  str(file['1']['EDS']['Header']['Drift Correction'][...] )
        eds_num_bins =                          int(file['1']['EDS']['Header']['Number Channels'][...] )
        eds_processor =                         str(file['1']['EDS']['Header']['Processor Type'][...][0] ).split("'")[1::2][0]
        eds_date =                              str( file['1']['EDS']['Header']['Acquisition Date'][...][0] ).split("'")[1::2][0]
        eds_x_pos =                             float(file['1']['EDS']['Header']['Stage Position']['X'][...] )
        eds_y_pos =                             float(file['1']['EDS']['Header']['Stage Position']['Y'][...] )
        eds_z_pos =                             float(file['1']['EDS']['Header']['Stage Position']['Z'][...] )
        eds_tilt_radians  =                     float(file['1']['EDS']['Header']['Stage Position']['Tilt'][...] )
        eds_rotation_radians =                  float(file['1']['EDS']['Header']['Stage Position']['Rotation'][...] )
        eds_strobe_area =                       float(file['1']['EDS']['Header']['Strobe Area'][...] )
        eds_strobe_FWHM =                       float(file['1']['EDS']['Header']['Strobe FWHM'][...] )
        eds_window_type =                       str(file['1']['EDS']['Header']['Window Type'][...][0] ).split("'")[1::2][0]
        eds_working_distance =                  float(file['1']['EDS']['Header']['Working Distance'][...] )

    except: 
        eds_project =                           np.nan 
        eds_specimen =                          np.nan 
        eds_site =                              np.nan 
        eds_voltage =                           np.nan 
        eds_magnification =                     np.nan 
        eds_field =                             np.nan 
        eds_binning =                           np.nan 
        eds_bin_width =                         np.nan 
        eds_start_channel =                     np.nan 
        eds_averaged_frames =                   np.nan 
        eds_process_time =                      np.nan 
        eds_x_cells =                           np.nan 
        eds_y_cells =                           np.nan 
        eds_x_step =                            np.nan 
        eds_y_step =                            np.nan 
        eds_real_time_sum =                     np.nan
        eds_live_time =                         np.nan
        eds_unique_vendor_id =                  np.nan
        eds_detector_azimuth_radians =          np.nan
        eds_detector_elevation_radians =        np.nan
        eds_hardware_detector_serial_number =   np.nan
        eds_hardware_detector_model =           np.nan
        eds_drift_correction =                  np.nan
        eds_num_bins =                          np.nan
        eds_processor =                         np.nan
        eds_date =                              np.nan
        eds_magnification =                     np.nan
        eds_x_pos =                             np.nan
        eds_y_pos =                             np.nan
        eds_z_pos =                             np.nan
        eds_tilt_radians  =                     np.nan
        eds_rotation_radians =                  np.nan
        eds_strobe_area =                       np.nan
        eds_strobe_FWHM =                       np.nan
        eds_window_type =                       np.nan
        eds_working_distance =                  np.nan
        
    ##########
    # if available, collect EBSD metadata
    try: 
        ebsd_project =                          str( file['1']['EDS']['Header']['Project Label'][...][0] ).split("'")[1::2][0]
        ebsd_specimen =                         str( file['1']['EDS']['Header']['Specimen Label'][...][0] ).split("'")[1::2][0]
        ebsd_site =                             str( file['1']['EDS']['Header']['Site Label'][...][0] ).split("'")[1::2][0]
        ebsd_voltage =                          float( file['1']['EDS']['Header']['Beam Voltage'][...] )
        ebsd_magnification =                    float( file['1']['EDS']['Header']['Magnification'][...] )
        ebsd_field =                            str( file['1']['EDS']['Header']['Analysis Label'][...][0] ).split("'")[1::2][0]  
        ebsd_acq_date =                         str(file['1']['EBSD']['Header']['Acquisition Date'][...] )
        ebsd_acq_speed =                        float(file['1']['EBSD']['Header']['Acquisition Speed'][...] )
        ebsd_acq_time =                         float(file['1']['EBSD']['Header']['Acquisition Time'][...] )
        ebsd_unique_ID =                        str(file['1']['EBSD']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0]  
        ebsd_background_correction =            str(file['1']['EBSD']['Header']['Auto Background Correction'][...] )
        ebsd_band_detection_mode =              str(file['1']['EBSD']['Header']['Band Detection Mode'][...] )
        ebsd_bounding_box =                     list(file['1']['EBSD']['Header']['Bounding Box Size'][...] )
        ebsd_bounding_box_x = ebsd_bounding_box[0]
        ebsd_bounding_box_y = ebsd_bounding_box[1]
        ebsd_camera_binning_mode =              str(file['1']['EBSD']['Header']['Camera Binning Mode'][...] )
        ebsd_camera_exposure_time =             float(file['1']['EBSD']['Header']['Camera Exposure Time'][...] )
        ebsd_camera_gain =                      float(file['1']['EBSD']['Header']['Camera Gain'][...] )
        ebsd_detector_insertion_distance =      float(file['1']['EBSD']['Header']['Detector Insertion Distance'][...] )
        ebsd_detector_orientation_euler =       list(file['1']['EBSD']['Header']['Detector Orientation Euler'][...] )[0]
        ebsd_detector_orientation_euler_a =     ebsd_detector_orientation_euler[0]
        ebsd_detector_orientation_euler_b =     ebsd_detector_orientation_euler[1]
        ebsd_detector_orientation_euler_c =     ebsd_detector_orientation_euler[2]
        ebsd_drift_correction =                 str(file['1']['EBSD']['Header']['Drift Correction'][...] )
        ebsd_hit_rate =                         float(file['1']['EBSD']['Header']['Hit Rate'][...] )
        ebsd_hough_resolution =                 int(file['1']['EBSD']['Header']['Hough Resolution'][...] )
        ebsd_indexing_mode =                    str(file['1']['EBSD']['Header']['Indexing Mode'][...] )
        ebsd_lens_distortion =                  float(file['1']['EBSD']['Header']['Lens Distortion'][...] )
        ebsd_lens_field_view =                  float(file['1']['EBSD']['Header']['Lens Field View'][...] )
        ebsd_number_bands_detected =            int(file['1']['EBSD']['Header']['Number Frames Averaged'][...] )
        ebsd_number_frames_averaged =           int(file['1']['EBSD']['Header']['Pattern Height'][...] )
        ebsd_pattern_height =                   int(file['1']['EBSD']['Header']['Pattern Width'][...] )
        ebsd_pattern_width =                    int(file['1']['EBSD']['Header']['Pattern Height'][...] )
        ebsd_project_file =                     str(file['1']['EBSD']['Header']['Project File'][...] )
        ebsd_project_notes =                    str(file['1']['EBSD']['Header']['Project Notes'][...] )
        ebsd_relative_offset =                  list(file['1']['EBSD']['Header']['Relative Offset'][...] )
        ebsd_relative_offset_x = ebsd_relative_offset[0]
        ebsd_relative_offset_y = ebsd_relative_offset[1]
        ebsd_relative_size =                    list(file['1']['EBSD']['Header']['Relative Size'][...] )
        ebsd_relative_size_x = ebsd_relative_size[0]
        ebsd_relative_size_y = ebsd_relative_size[1]
        ebsd_scanning_rotation_angle =          float(file['1']['EBSD']['Header']['Scanning Rotation Angle'][...] )
        ebsd_site_notes =                       str(file['1']['EBSD']['Header']['Site Notes'][...] )
        ebsd_specimen_notes =                   str(file['1']['EBSD']['Header']['Specimen Notes'][...] )
        ebsd_specimen_orientation =             list(file['1']['EBSD']['Header']['Specimen Orientation Euler'][...] )[0]
        ebsd_specimen_orientation_a =           ebsd_specimen_orientation[0]
        ebsd_specimen_orientation_b =           ebsd_specimen_orientation[1]
        ebsd_specimen_orientation_c =           ebsd_specimen_orientation[2]
        ebsd_stage_x =                          float(file['1']['EBSD']['Header']['Stage Position']['X'][...] )
        ebsd_stage_y =                          float(file['1']['EBSD']['Header']['Stage Position']['Y'][...] )
        ebsd_stage_z =                          float(file['1']['EBSD']['Header']['Stage Position']['Z'][...] )
        ebsd_stage_rotation =                   float(file['1']['EBSD']['Header']['Stage Position']['Rotation'][...] )
        ebsd_stage_tilt =                       float(file['1']['EBSD']['Header']['Stage Position']['Tilt'][...] )
        ebsd_static_background_correction =     int(file['1']['EBSD']['Header']['Static Background Correction'][...] )
        ebsd_tilt_angle =                       float(file['1']['EBSD']['Header']['Tilt Angle'][...] )
        ebsd_tilt_axis =                        float(file['1']['EBSD']['Header']['Tilt Axis'][...] )
        ebsd_working_distance =                 float(file['1']['EDS']['Header']['Working Distance'][...] )
        ebsd_xcells =                           int(file['1']['EDS']['Header']['X Cells'][...] )
        ebsd_ycells =                           int(file['1']['EDS']['Header']['Y Cells'][...] )
        ebsd_xstep =                            float(file['1']['EDS']['Header']['X Step'][...] )
        ebsd_ystep =                            float(file['1']['EDS']['Header']['Y Step'][...] )
        
    except: 
        ebsd_project =                          np.nan
        ebsd_specimen =                         np.nan
        ebsd_site =                             np.nan
        ebsd_voltage =                          np.nan
        ebsd_magnification =                    np.nan
        ebsd_field =                            np.nan
        ebsd_acq_date =                         np.nan
        ebsd_acq_speed =                        np.nan 
        ebsd_acq_time =                         np.nan
        ebsd_label =                            np.nan
        ebsd_unique_ID =                        np.nan 
        ebsd_background_correction =            np.nan
        ebsd_band_detection_mode =              np.nan
        ebsd_beam_voltage =                     np.nan
        ebsd_bounding_box =                     np.nan
        ebsd_bounding_box_x =                   np.nan
        ebsd_bounding_box_y =                   np.nan
        ebsd_camera_binning_mode =              np.nan
        ebsd_camera_exposure_time =             np.nan
        ebsd_camera_gain =                      np.nan
        ebsd_detector_insertion_distance =      np.nan
        ebsd_detector_orientation_euler =       np.nan
        ebsd_detector_orientation_euler_a =     np.nan
        ebsd_detector_orientation_euler_b =     np.nan
        ebsd_detector_orientation_euler_c =     np.nan 
        ebsd_drift_correction =                 np.nan
        ebsd_hit_rate =                         np.nan
        ebsd_hough_resolution =                 np.nan
        ebsd_indexing_mode =                    np.nan
        ebsd_lens_distortion =                  np.nan
        ebsd_lens_field_view =                  np.nan
        ebsd_magnification =                    np.nan
        ebsd_number_bands_detected =            np.nan
        ebsd_number_frames_averaged =           np.nan
        ebsd_pattern_height =                   np.nan
        ebsd_pattern_width =                    np.nan
        ebsd_project_file =                     np.nan
        ebsd_project_label =                    np.nan
        ebsd_project_notes =                    np.nan
        ebsd_relative_offset =                  np.nan
        ebsd_relative_offset_x =                np.nan
        ebsd_relative_offset_y =                np.nan
        ebsd_relative_size =                    np.nan
        ebsd_relative_size_x =                  np.nan
        ebsd_relative_size_y =                  np.nan
        ebsd_scanning_rotation_angle =          np.nan
        ebsd_site_label =                       np.nan
        ebsd_site_notes =                       np.nan
        ebsd_specimen_label =                   np.nan
        ebsd_specimen_notes =                   np.nan
        ebsd_specimen_orientation =             np.nan
        ebsd_specimen_orientation_a =           np.nan
        ebsd_specimen_orientation_b =           np.nan
        ebsd_specimen_orientation_c =           np.nan
        ebsd_stage_x =                          np.nan
        ebsd_stage_y =                          np.nan
        ebsd_stage_z =                          np.nan
        ebsd_stage_rotation =                   np.nan
        ebsd_stage_tilt =                       np.nan
        ebsd_static_background_correction =     np.nan
        ebsd_tilt_angle =                       np.nan
        ebsd_tilt_axis =                        np.nan
        ebsd_working_distance =                 np.nan
        ebsd_xcells =                           np.nan
        ebsd_ycells =                           np.nan
        ebsd_xstep =                            np.nan
        ebsd_ystep =                            np.nan
    
    ##########
    metadata = [] 
    
    metadata.append( (
        montage_name,
        hardware_vendor,
        software_version, 
        format_version, 

        electron_image_project,
        electron_image_specimen,
        electron_image_site,
        electron_image_field,
        electron_image_voltage,
        electron_image_magnification,
        electron_image_unique_vendor_id,
        electron_image_drift_correction,
        electron_image_dwell_time,
        electron_image_average_frames,
        electron_image_x_cells,
        electron_image_y_cells,
        electron_image_x_step,
        electron_image_y_step,
        electron_image_date,
        electron_image_x_pos,
        electron_image_y_pos,
        electron_image_z_pos,
        electron_image_tilt_radians,
        electron_image_rotation_radians,
        electron_image_working_distance,
        
        eds_project,
        eds_specimen,
        eds_site,
        eds_field,
        eds_voltage,
        eds_magnification,
        eds_binning,
        eds_bin_width,
        eds_start_channel,
        eds_averaged_frames,
        eds_process_time,
        eds_x_cells,
        eds_y_cells,
        eds_x_step,
        eds_y_step,
        eds_real_time_sum,
        eds_live_time,
        eds_unique_vendor_id,
        eds_detector_azimuth_radians,
        eds_detector_elevation_radians,
        eds_hardware_detector_serial_number,
        eds_hardware_detector_model,
        eds_drift_correction,
        eds_num_bins,
        eds_processor,
        eds_date,
        eds_x_pos,
        eds_y_pos,
        eds_z_pos,
        eds_tilt_radians,
        eds_rotation_radians,
        eds_strobe_area,
        eds_strobe_FWHM,
        eds_window_type,
        eds_working_distance,
        
        ebsd_project,
        ebsd_specimen,
        ebsd_site,
        ebsd_field,
        ebsd_voltage,
        ebsd_magnification,
        ebsd_acq_date,
        ebsd_acq_speed,
        ebsd_acq_time,
        ebsd_unique_ID,
        ebsd_background_correction,
        ebsd_band_detection_mode,
        ebsd_bounding_box_x,
        ebsd_bounding_box_y,
        ebsd_camera_binning_mode,
        ebsd_camera_exposure_time,
        ebsd_camera_gain,
        ebsd_detector_insertion_distance,
        ebsd_detector_orientation_euler_a,
        ebsd_detector_orientation_euler_b,
        ebsd_detector_orientation_euler_c,
        ebsd_drift_correction,
        ebsd_hit_rate,
        ebsd_hough_resolution,
        ebsd_indexing_mode,
        ebsd_lens_distortion,
        ebsd_lens_field_view,
        ebsd_magnification,
        ebsd_number_bands_detected,
        ebsd_number_frames_averaged,
        ebsd_pattern_height,
        ebsd_pattern_width,
        ebsd_project_file,
        ebsd_project_notes,
        ebsd_relative_offset_x,
        ebsd_relative_offset_y,      
        ebsd_relative_size_x,        
        ebsd_relative_size_y,
        ebsd_scanning_rotation_angle,
        ebsd_site_notes,
        ebsd_specimen_notes,
        ebsd_specimen_orientation_a,
        ebsd_specimen_orientation_b,
        ebsd_specimen_orientation_c,
        ebsd_stage_x,
        ebsd_stage_y ,
        ebsd_stage_z,
        ebsd_stage_rotation,
        ebsd_stage_tilt,
        ebsd_static_background_correction,
        ebsd_tilt_angle,
        ebsd_tilt_axis,
        ebsd_working_distance,
        ebsd_xcells,
        ebsd_ycells,
        ebsd_xstep,
        ebsd_ystep
        ) ) 

    metadata = np.asarray(metadata, dtype=object)

    ##########    
    cols = ["Montage Label",
            "Hardware Vendor",
            "Software Version", 
            "Format Version", 

            "SEM Project",
            "SEM Specimen",
            "SEM Site",
            "SEM Field",
            "SEM Voltage (KeV)",
            "SEM Magnification",            
            "SEM Vendor Unique ID",
            "SEM Drift Correction",
            "SEM Dwell Time (us)",
            "SEM Average Frames",
            "SEM Number of X Cells",
            "SEM Number of Y Cells",
            "SEM X Step Size (um)",
            "SEM Y Step Size (um)",
            "SEM Date",
            "SEM Stage X Position (mm)",
            "SEM Stage Y Position (mm)",
            "SEM Stage Z Position (mm)",
            "SEM Stage Tilt (rad)",
            "SEM Stage Rotation (rad)",
            "SEM Working Distance (mm)",
            
            "EDS Project",
            "EDS Specimen",
            "EDS Site",
            "EDS Field",
            "EDS Voltage (KeV)",
            "EDS Magnification",
            "EDS Binning Factor",
            "EDS Voltage Bin Width (eV)",
            "EDS Starting Bin Voltage (eV)",
            "EDS Number of Averaged Frames",
            "EDS Process Time",
            "EDS Number of X Cells",
            "EDS Number of Y Cells",
            "EDS X Step Size (um)",
            "EDS Y Step Size (um)",
            "EDS Real Time Sum (s)",
            "EDS Live Time (s)",
            "EDS Vendor Unique ID",
            "EDS Azimuth Angle (rad)",
            "EDS Detector Angle (rad)",
            "EDS Detector Serial Number",
            "EDS Detector Model Number",
            "EDS Drift Correction",
            "EDS Number of Channels",
            "EDS Processor Type",
            "EDS Date",
            "EDS Stage X Position (mm)",
            "EDS Stage Y Position (mm)",
            "EDS Stage Z Position (mm)",
            "EDS Stage Tilt (rad)",
            "EDS Stage Rotation (rad)",
            "EDS Strobe Area",
            "EDS Strobe FWHM (ev)",
            "EDS Window Type",
            "EDS Working Distance (mm)",
            
            "EBSD Project",
            "EBSD Specimen",
            "EBSD Site",
            "EBSD Field",
            "EBSD Voltage (KeV)",
            "EBSD Magnification",
            "EBSD Acquisition Date",
            "EBSD Acquisition Speed (Hz)",
            "EBSD Acquisition Time (s)",
            "EBSD Vendor Unique ID",
            "EBSD Auto Background Correction",
            "EBSD Band Detection Mode",
            "EBSD Bounding Box X (um)",
            "EBSD Bounding Box Y (um)",
            "EBSD Camera Binning Mode",
            "EBSD Camera Exposure Time (ms)",
            "EBSD Camera Gain",
            "EBSD Detector Insertion Distance (mm)",
            "EBSD Detector Orientation Euler A (rad)",
            "EBSD Detector Orientation Euler B (rad)",
            "EBSD Detector Orientation Euler C (rad)",
            "EBSD Drift Correction",
            "EBSD Hit Rate",
            "EBSD Hough Resolution",
            "EBSD Indexing Mode",
            "EBSD Lens Distortion",
            "EBSD Lens Field View (mm)",
            "EBSD Magnification",
            "EBSD Number Bands Detected",
            "EBSD Number Frames Averaged",
            "EBSD Pattern Height (px)",
            "EBSD Pattern Width (px)",
            "EBSD Project File",
            "EBSD Project Notes",
            "EBSD Relative Offset X",
            "EBSD Relative Offset Y",
            "EBSD Relative Size X",
            "EBSD Relative Size Y",
            "EBSD Scanning Rotation Angle (rad)",
            "EBSD Site Notes",
            "EBSD Specimen Notes",
            "EBSD Specimen Orientation A",
            "EBSD Specimen Orientation B",
            "EBSD Specimen Orientation C",
            "EBSD Stage X Position (mm)",
            "EBSD Stage Y Position (mm)" ,
            "EBSD Stage Z Position (mm)",
            "EBSD Stage Rotation (rad)",
            "EBSD Stage Tilt (rad)",
            "EBSD Static Background Correction",
            "EBSD Tilt Angle (rad)",
            "EBSD Tilt Axis",
            "EBSD Working Distance (mm)",
            "EBSD Number X Pixels",
            "EBSD Number Y Pixels",
            "EBSD X Step Size (um)",
            "EBSD Y Step Size (um)"]

    metadata = pd.DataFrame( data = metadata, columns = cols)
    
    ##########
    # Parse phases if EBSD data is available 
    try: 
        ebsd_phases = {}
        for phase in list( file['1']['EBSD']['Header']['Phases'].keys() ): 
            
            temp = {}
            temp['Color'] =                     file['1']['EBSD']['Header']['Phases'][phase]['Color'][...]
            temp['Database Id'] =               file['1']['EBSD']['Header']['Phases'][phase]['Database Id'][...]
            temp['Lattice Angles'] =            file['1']['EBSD']['Header']['Phases'][phase]['Lattice Angles'][...]
            temp['Lattice Dimensions'] =        file['1']['EBSD']['Header']['Phases'][phase]['Lattice Dimensions'][...]
            temp['Laue Group'] =                file['1']['EBSD']['Header']['Phases'][phase]['Laue Group'][...]
            temp['Number Reflectors'] =         file['1']['EBSD']['Header']['Phases'][phase]['Number Reflectors'][...]
            temp['Phase Id'] =                  file['1']['EBSD']['Header']['Phases'][phase]['Phase Id'][...]
            temp['Phase Name'] =                file['1']['EBSD']['Header']['Phases'][phase]['Phase Name'][...]
            temp['Reference'] =                 file['1']['EBSD']['Header']['Phases'][phase]['Reference'][...]
            temp['Space Group'] =               file['1']['EBSD']['Header']['Phases'][phase]['Space Group'][...]
            
            ebsd_phases[phase] = temp
    except: 
        pass 
    ##########

    return metadata, ebsd_phases


def oxford_parse_rpl(file_path): 
    # Reads RPL EDS files and returns the hyperspectral datacube 
    #####################################################################################
    
    cube = hs.load(file_path)
    data = cube.data
    
    new_data = np.zeros(shape = (data.shape[1], data.shape[2], data.shape[0] ), dtype = np.float32 )   
    for i in range(data.shape[0]):
        new_data[:,:, i ] = np.flip( data[i,:,:], 1)

    return new_data 


def oxford_get_metadata(file_list): 
    # Input list of file addresses to parse for metadata. 
    # Returns a dataframe and list of dictionaries - metadata and EBSD phases 
    #####################################################################################
    
    initializing_flag = True
    for i, file_path in enumerate( tqdm(file_list, total = len(file_list) ) ): 

        # skip file_path file that are not h5oina 
        if Path(file_path).suffix == '.h5oina': 

            if "Montaged Map Data" in file_path:
                    continue 
                
            # verify that the file is H5 compatible 
            try: 
                file = h5py.File(os.path.join(file_path), "r")
            except OSError: 
                print("Error. Selected file: " + str(file_path) + " is incompatable with HF5 format")
                sys.exit()
            else: 
                pass 
         
            # On the first iteration, create dataframe.
            # On subsequent iterations, append to dataframe 
            if initializing_flag: 
                metadata, ebsd_phases_placeholder = oxford_parse_h5(file_path)
                ebsd_phases = []
                ebsd_phases.append(ebsd_phases_placeholder)
                initializing_flag = False
            else: 
                metadata_placeholder, ebsd_phases_placeholder = oxford_parse_h5(file_path)
                
                metadata = pd.concat( [metadata, metadata_placeholder], ignore_index = True )
                ebsd_phases.append(ebsd_phases_placeholder)
            
        else:
            # skip to next file in list if not H5OINA
            continue   
        
        
    return metadata, ebsd_phases 



def oxford_montage_stitcher(montage, metadata, specification, file_list, eds_layers, eds_binning_width):
    # This function stitches together the 2D images and 3D EDS datacubes for Oxford montages 
    # Metadata and H5 file paths are required, RPL files are optional and only used if stitching EDS datacubes 
    #####################################################################################
    
    montage = montage.replace("Montage", "", 1)
    montage = montage.strip()
    
    oxford_file_list = []
    for file in file_list:
        if Path(file).suffix in ['.h5oina']: 
            oxford_file_list.append(file)
            
    array_flag = True
    
    if specification[0] == "Electron Image": 
        array_ID_list = metadata['SEM Vendor Unique ID'][metadata['Montage Label'] == montage]

        if specification[1] in ["SE", "BSE", "FSE Lower Left", "FSE Lower Centre", "FSE Lower Right", "FSE Upper Left", "FSE Upper Right"]:
        
            # Get the um/px scale 
            x_scale = list( np.unique( metadata['SEM X Step Size (um)'][metadata['Montage Label'] == montage] ) )
            y_scale = list( np.unique( metadata['SEM Y Step Size (um)'][[a and b for a, b in zip(metadata['Montage Label'] == montage, metadata['SEM Field'] != 'Montaged Map Data')]] ) )
            
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
                
            # get the XY coordinates for each field 
            x_list =  list( np.unique( metadata['SEM Stage X Position (mm)'][metadata['Montage Label'] == montage] ) )
            y_list =  list( np.unique( metadata['SEM Stage Y Position (mm)'][metadata['Montage Label'] == montage] ) ) 
   
            # find minimum coodinates so that we can create an appropriately sized array 
            x_min = int( math.ceil(min(x_list)*1000/x_scale))
            y_min = int( math.ceil(min(y_list)*1000/y_scale))
            
            for i, path in enumerate( tqdm(file_list, total = len(oxford_file_list) ) ):
                
                if "Montaged Map Data" in path:
                    continue 
                
                # skip over file that are not h5oina 
                if Path(path).suffix == '.h5oina': 
    
                    # verify that the file is H5 compatible 
                    try: 
                        file = h5py.File(os.path.join(path), "r")
                    except OSError: 
                        print("Error. Selected file type is incompatable with HF5 format")
                        sys.exit()
                    else: 
                        pass 
                
                else:
                    # skip to next file in list 
                    continue 
                    
                try: 
                    if str(file['1']['Electron Image']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0] in array_ID_list.values:
                        
                        # For the first field to add, find the pixel shape of a field and create an empty array to hold data 
                        if array_flag:
                            # get pixel shape of a field 
                            x_size = int(file['1'][specification[0]]['Header']['X Cells'][...])
                            y_size = int(file['1'][specification[0]]['Header']['Y Cells'][...])
                            
                            # determine how large of an array is neeed 
                            x_range = int (math.ceil( ( 1000*(max(x_list) - min(x_list))/x_scale ) ) + x_size )
                            y_range = int( math.ceil( ( 1000*(max(y_list) - min(y_list))/y_scale ) ) + y_size )
                          
                            # Specify the appropriate depth for the stitched array
                            if ( specification[1] in ["SE", "BSE"] ) or ( "FSE" in specification[1] ): 
                                array = np.zeros( shape = (y_range, x_range) )
                                array_flag = False
                            else: 
                                raise Exception("Specification must be one of 'SE', 'BSE' or 'FSE' for SEM datasets. Instead found: " + str(specification[1]))
                           
                        # load data, reshape and get the XY coodinates for the field 
                        if specification[1] == "SE":
                            #field_ID = str(file['1']['Electron Image']['Header']['Analysis Label'][0] ).split("'")[1::2][0].split(" ")[2]
                            #block = file['1']['Electron Image']['Data']['SE']['Electron Image ' + str(field_ID)][...].reshape( y_size, x_size )
                            
                            field_ID = list( file['1']['Electron Image']['Data']['SE'].keys() )[0]
                            block = file['1']['Electron Image']['Data']['SE'][str(field_ID)][...].reshape( y_size, x_size )
                            block = np.flip( block, 1)
                        
                        elif specification[1] == "BSE":
                            #field_ID = str(file['1']['Electron Image']['Header']['Analysis Label'][0] ).split("'")[1::2][0].split(" ")[2]
                            #block = file['1']['Electron Image']['Data']['BSE']['Electron Image ' + str(field_ID)][...].reshape( y_size, x_size )
                            
                            field_ID = list( file['1']['Electron Image']['Data']['BSE'].keys() )[0]
                            block = file['1']['Electron Image']['Data']['BSE'][str(field_ID)][...].reshape( y_size, x_size )
                            block = np.flip( block, 1)
                            
                        elif specification[1] in ["FSE Lower Left", "FSE Lower Centre", "FSE Lower Right", "FSE Upper Left", "FSE Upper Right"]:
                            field_ID = list(file['1']['Electron Image']['Data']['FSE'].keys())[0].replace("Lower Centre ", "")
                            block = file['1']['Electron Image']['Data']['FSE'][specification[1].replace('FSE ', '') + ' ' + str(field_ID)][...].reshape( y_size, x_size )
                            block = np.flip( block, 1)
                            
                        if isinstance(block, (np.ndarray)): 
                            
                            x_location = int( np.round( 1000*file['1']['Electron Image']['Header']['Stage Position']['X'][...]/x_scale, 0 ) ) - x_min
                            y_location = int( np.round( 1000*file['1']['Electron Image']['Header']['Stage Position']['Y'][...]/y_scale, 0 ) ) - y_min
                                            
                            if x_location < 0:
                                x_location = 0
                            
                            if y_location < 0:
                                y_location = 0
                            
                            array = stitch(array, block, x_size, y_size, x_location, y_location)
                            gc.collect()
                except KeyError:
                    pass 
                        
        
    elif specification[0] == "EDS":     
        array_ID_list = metadata['EDS Vendor Unique ID'][metadata['Montage Label'] == montage]

               
        if specification[1] in ["Real Time", "Live Time", "XRay"]:
        
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
                
            # get the XY coordinates for each field 
            x_list =  list( np.unique( metadata['EDS Stage X Position (mm)'][metadata['Montage Label'] == montage] ) )
            y_list =  list( np.unique( metadata['EDS Stage Y Position (mm)'][metadata['Montage Label'] == montage] ) ) 
   
            # find minimum coodinates so that we can create an appropriately sized array 
            x_min = int( math.ceil(min(x_list)*1000/x_scale))
            y_min = int( math.ceil(min(y_list)*1000/y_scale))
            
            for i, path in enumerate( tqdm(file_list, total = len(oxford_file_list) ) ):
                
                if "Montaged Map Data" in path:
                    continue 
                
                # skip over file that are not h5oina 
                if Path(path).suffix == '.h5oina': 
    
                    # verify that the file is H5 compatible 
                    try: 
                        file = h5py.File(os.path.join(path), "r")
                    except OSError: 
                        print("Error. Selected file type is incompatable with HF5 format")
                        sys.exit()
                    else: 
                        pass 
                
                else:
                    # skip to next file in list 
                    continue 
                    
                if str(file['1']['EDS']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0] in array_ID_list.values:
                    
                    # For the first field to add, find the pixel shape of a field and create an empty array to hold data 
                    if array_flag:
                        # get pixel shape of a field 
                        x_size = int(file['1'][specification[0]]['Header']['X Cells'][...])
                        y_size = int(file['1'][specification[0]]['Header']['Y Cells'][...])
                        
                        # determine how large of an array is neeed 
                        x_range = int (math.ceil( ( 1000*(max(x_list) - min(x_list))/x_scale ) ) + x_size )
                        y_range = int( math.ceil( ( 1000*(max(y_list) - min(y_list))/y_scale ) ) + y_size )
                      
                        # Specify the appropriate depth for the stitched array
                        if specification[1] in ["Real Time", "Live Time"]: 
                            array = np.zeros( shape = (y_range, x_range), dtype = np.uint8 )
                            array_flag = False
                        elif specification[1] == "XRay": 
                            #z = int(file['1']['EDS']['Header']['Number Channels'][...])
                            array = np.zeros( shape = (y_range, x_range, len(eds_layers)), dtype = np.uint8 )
                            array_flag = False 
                        else: 
                            raise Exception("Specification must be one of 'Live Time', 'Real Time', or 'XRay' for EDS datasets. Instead found: " + str(specification[1]))
                       
                    # load data, reshape and get the XY coodinates for the field 
                    if specification[1] == "Real Time":
                        block = file['1']['EDS']['Data']['Real Time'][...].reshape( y_size, x_size )
                        block = np.flip( block, 1)
                        
                    if specification[1] == "Live Time":
                        block = file['1']['EDS']['Data']['Live Time'][...].reshape( y_size, x_size )
                        block = np.flip( block, 1)
                        
                    elif specification[1] == "XRay": 
                        field_ID = str(file['1']['EDS']['Header']['Analysis Label'][0] ).split("'")[1::2][0]
                        
                        for path in file_list:
                            
                            if ( Path(path).stem.startswith("EDS " + field_ID) ) and ( Path(path).suffix == '.npy'):
                                temp = np.load(path)
                                block = np.zeros( shape = (temp.shape[0], temp.shape[1], len(eds_layers) ))
                                for zz, row in enumerate(eds_layers):
                                    block[:,:,zz] = np.sum( temp[:,:, row - eds_binning_width:row + eds_binning_width], axis = 2 )
                                break 
                                
                            elif ( Path(path).stem.startswith("EDS " + field_ID) ) and ( Path(path).suffix == '.rpl'):
                                temp = oxford_parse_rpl(path)
                                block = np.zeros( shape = (temp.shape[0], temp.shape[1], len(eds_layers) ))
                                for zz, row in enumerate(eds_layers):
                                    block[:,:,zz] = np.sum( temp[:,:, row - eds_binning_width:row + eds_binning_width], axis = 2 )
                                
                                break 
                        
                    x_location = int( np.round( 1000*file['1']['EDS']['Header']['Stage Position']['X'][...]/x_scale, 0 ) ) - x_min
                    y_location = int( np.round( 1000*file['1']['EDS']['Header']['Stage Position']['Y'][...]/y_scale, 0 ) ) - y_min
                            
                    if x_location < 0:
                        x_location = 0
                    
                    if y_location < 0:
                        y_location = 0
                        
                    array = stitch(array, block, x_size, y_size, x_location, y_location)
                    
                    gc.collect()
                    
                    
    elif specification[0] == "EBSD": 
        array_ID_list = metadata['EDS Vendor Unique ID'][metadata['Montage Label'] == montage]
        
        if specification[1] in ["Band Contrast", "Band Slope", "Bands", "Beam Position X", "Beam Position Y", "Detector Distance", "Error", "Euler", "Mean Angular Deviation", "Pattern Center X", "Pattern Center Y", "Pattern Qualtiy", "Phase", "X", "Y"]:
        
            # Get the um/px scale 
            x_scale = list( np.unique( metadata['EBSD X Step Size (um)'][metadata['Montage Label'] == montage] ) )
            y_scale = list( np.unique( metadata['EBSD Y Step Size (um)'][metadata['Montage Label'] == montage] ) )
            
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
                
            # get the XY coordinates for each field 
            x_list =  list( np.unique( metadata['EBSD Stage X Position (mm)'][metadata['Montage Label'] == montage] ) )
            y_list =  list( np.unique( metadata['EBSD Stage Y Position (mm)'][metadata['Montage Label'] == montage] ) ) 
   
            # find minimum coodinates so that we can create an appropriately sized array 
            x_min = int( math.ceil(min(x_list)*1000/x_scale))
            y_min = int( math.ceil(min(y_list)*1000/y_scale))

            for i, path in enumerate( tqdm(file_list, total = len(oxford_file_list) ) ):
                
                if "Montaged Map Data" in path:
                    continue 
                
                # skip over file that are not h5oina 
                if Path(path).suffix == '.h5oina': 
    
                    # verify that the file is H5 compatible 
                    try: 
                        file = h5py.File(os.path.join(path), "r")
                    except OSError: 
                        print("Error. Selected file type is incompatable with HF5 format")
                        sys.exit()
                    else: 
                        pass 
                
                else:
                    # skip to next file in list 
                    continue 
                    
              
                if str(file['1']['EBSD']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0] in array_ID_list.values:

                    # For the first field to add, find the pixel shape of a field and create an empty array to hold data 
                    if array_flag:
                        # get pixel shape of a field 
                        x_size = int(file['1']['EBSD']['Header']['X Cells'][...])
                        y_size = int(file['1']['EBSD']['Header']['Y Cells'][...])
                        
                        # determine how large of an array is neeed 
                        x_range = int (math.ceil( ( 1000*(max(x_list) - min(x_list))/x_scale ) ) + x_size )
                        y_range = int( math.ceil( ( 1000*(max(y_list) - min(y_list))/y_scale ) ) + y_size )
                      
                        # Specify the appropriate depth for the stitched array
                        if specification[1] in ["Band Contrast", "Band Slope", "Bands", "Beam Position X", "Beam Position Y", "Detector Distance", "Error", "Euler", "Mean Angular Deviation", "Pattern Center X", "Pattern Center Y", "Pattern Qualtiy", "Phase", "X", "Y"]: 
                            array = np.zeros( shape = (y_range, x_range) )
                            array_flag = False
                        else: 
                            raise Exception('Specification must be one of "Band Contrast", "Band Slope", "Bands", "Beam Position X", "Beam Position Y", "Detector Distance", "Error", "Euler", "Mean Angular Deviation", "Pattern Center X", "Pattern Center Y", "Pattern Qualtiy", "Phase", "X", "Y" for EBSD datasets. Instead found: ' + str(specification[1]))
                       
                    # load data, reshape and get the XY coodinates for the field 
                    field_ID = str(file['1']['EBSD']['Header']['Analysis Label'][0] ).split("'")[1::2][0].split(" ")[2]
                    block = file['1']['EBSD']['Data'][specification[1]][...].reshape( y_size, x_size )
                    block = np.flip( block, 1)
                        
                    x_location = int( np.round( 1000*file['1']['EBSD']['Header']['Stage Position']['X'][...]/x_scale, 0 ) ) - x_min
                    y_location = int( np.round( 1000*file['1']['EBSD']['Header']['Stage Position']['Y'][...]/y_scale, 0 ) ) - y_min
                                    
                    if x_location < 0:
                        x_location = 0
                    
                    if y_location < 0:
                        y_location = 0
                    
                    array = stitch(array, block, x_size, y_size, x_location, y_location)
                    gc.collect()
    
    return array
   

def convert_oxford_RPL_to_H5(montage, metadata, file_list, output_src): 
    
    montage = montage.replace("Montage", "", 1)
    montage = montage.strip()
    
    oxford_file_list = []
    for file in file_list:
        if Path(file).suffix in ['.h5oina']: 
            oxford_file_list.append(file)
            
    montage_file = os.path.join(output_src, 'Montage ' + str(montage) + '.h5')
    
    array_ID_list = metadata['EDS Vendor Unique ID'][metadata['Montage Label'] == montage]

    # Get the um/px scale 
    x_scale = list( np.unique( metadata['EDS X Step Size (um)'][metadata['Montage Label'] == montage] ) )
    y_scale = list( np.unique( metadata['EDS Y Step Size (um)'][metadata['Montage Label'] == montage] ) )
    
    # remove nan if present 
    x_scale = [x for x in x_scale if np.isnan(float(x)) == False]
    y_scale = [x for x in y_scale if np.isnan(float(x)) == False]
        
    x_scale = np.unique( np.asarray(x_scale, dtype = float))
    y_scale = np.unique( np.asarray(y_scale, dtype = float))
    
    # verify that only one value for the resolution exists for a given montage 
    if (len(x_scale) > 1) or (len(y_scale) > 1): 
        raise Exception("Multiple unique values found for um/px resolution: " + str(x_scale) + ", " + str(y_scale) ) 
    else: 
        x_scale = x_scale
        y_scale = y_scale
        
    # get the XY coordinates for each field 
    x_list =  list( np.asarray(np.unique( metadata['EDS Stage X Position (mm)'][metadata['Montage Label'] == montage] ), dtype = float ) )
    y_list =  list( np.asarray(np.unique( metadata['EDS Stage Y Position (mm)'][metadata['Montage Label'] == montage] ), dtype = float ) )

    # find minimum coodinates so that we can create an appropriately sized array 
    x_min = int( math.ceil(min(x_list)*1000/x_scale))
    y_min = int( math.ceil(min(y_list)*1000/y_scale))
    
    # get pixel shape of a field 
    x_size = np.asarray(np.unique( metadata['EDS Number of X Cells'][metadata['Montage Label'] == montage] ), dtype = int ) 
    y_size = np.asarray(np.unique( metadata['EDS Number of Y Cells'][metadata['Montage Label'] == montage] ), dtype = int )
    
    # determine how large of an array is neeed 
    x_range = int (math.ceil( ( 1000*(max(x_list) - min(x_list))/x_scale ) ) + x_size )
    y_range = int( math.ceil( ( 1000*(max(y_list) - min(y_list))/y_scale ) ) + y_size )
    
    for i, path in enumerate( tqdm(oxford_file_list, total = len(oxford_file_list), desc = 'Finding sum of spectrum and auto-detecting peaks' ) ):
        
        if "Montaged Map Data" in path:
            continue 
        
        # skip over file that are not h5oina 
        if Path(path).suffix == '.h5oina': 
            
            # verify that the file is H5 compatible 
            try: 
                file = h5py.File(os.path.join(path), "r")
            
            except OSError: 
                print("Error. Selected file type is incompatable with HF5 format")
                sys.exit()
            else: 
                pass 
        
        else:
            # skip to next file in list 
            continue 
            
        if str(file['1']['EDS']['Header']['Analysis Unique Identifier'][...][0] ).split("'")[1::2][0] in array_ID_list.values:
            
            # load data, reshape and get the XY coodinates for the field 
            #try: 
            field_ID = str(file['1']['EDS']['Header']['Analysis Label'][0] ).split("'")[1::2][0]
            
            for path2 in file_list:
                
                if ( Path(path2).stem.startswith("EDS " + field_ID) ) and ( Path(path2).suffix == '.npy'):
                    block = np.load(path2)
                    break 
                    
                elif ( Path(path2).stem.startswith("EDS " + field_ID) ) and ( Path(path2).suffix == '.rpl'):
                    block = oxford_parse_rpl(path2)
                    break 
            
            block = np.array(block, dtype = np.int16)
            
            x_location = int( np.round( 1000*file['1']['EDS']['Header']['Stage Position']['X'][...]/x_scale, 0 ) ) - x_min
            y_location = int( np.round( 1000*file['1']['EDS']['Header']['Stage Position']['Y'][...]/y_scale, 0 ) ) - y_min
            file.close()
            
            if x_location < 0:
                x_location = 0
            
            if y_location < 0:
                y_location = 0
            
            save_file = h5py.File(montage_file, 'a') 

            try:
                save_file.create_group('EDS')
            except:
                pass 
            
            try: 
                save_file['EDS'].create_dataset('Xray Spectrum', shape = (y_range, x_range, block.shape[2] ), chunks=(True, True, 50), dtype = 'int64')
            except: 
                pass 
            
            try: 
                save_file['EDS'].create_dataset('Xray Intensity', shape = (y_range, x_range ), chunks=True, dtype = 'int64')
            except: 
                pass 
            
            save_file['EDS']['Xray Intensity'][y_location:y_location + int(y_size), x_location:x_location + int(x_size)] = np.sum( block, axis = 2)
            save_file['EDS']['Xray Spectrum'][y_location:y_location + int(y_size), x_location:x_location + int(x_size), :] = block 

            save_file.close() 
        
            new_highest_spectrum =  np.max( block[:, :, :], axis = (0,1)).flatten() 
            new_sum_spectrum =  np.sum( block[:, :, :], axis = (0,1)).flatten() 
            
            # Sum new block with existing data
            try: 
                sum_spectrum += new_sum_spectrum
            except NameError:
                sum_spectrum = np.zeros( shape = new_sum_spectrum.shape, dtype = np.float64 )
                sum_spectrum += new_sum_spectrum
            
            # Keep only highest peaks from every block of data 
            try: 
                highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
            except NameError:
                highest_spectrum = np.zeros( shape = new_highest_spectrum.shape , dtype = np.float64 )
                highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                
            gc.collect()
            
    # Find peaks from both the average spectrum intensity as well as the maximum spectrum intensity 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    appended_peaks = []
    search_width = 3
    peaks_mean, _ = scipy.signal.find_peaks(sum_spectrum, 
                                       prominence = (0.10*statistics.median( sum_spectrum ), None),
                                       distance = 5
                                       )
    
    peaks_max, _ = scipy.signal.find_peaks(highest_spectrum, 
                                       height = (np.percentile(highest_spectrum, 95), None ),
                                       prominence = 2,
                                       distance = 5
                                       )
    
    peaks_mean = list( peaks_mean )
    peaks_max = list( peaks_max)  
    
    # Sort candidate peaks by the x ray intensity. We want to add the strongest peaks first
    a = np.argsort(sum_spectrum[peaks_mean]) 
    peaks_mean[:] = [peaks_mean[i] for i in a][:len(peaks_mean)]             
    
    a = np.argsort(highest_spectrum[peaks_max]) 
    peaks_max[:] = [peaks_max[i] for i in a][:len(peaks_max)]      
    
    # Iterate through all candidate peaks that were found for the montage and include the peak if it is more than X bins from previously detected peaks 
    i = 0 
    while True:          
        
        try: 
            peak_candidate = peaks_mean[i]
        
            if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                appended_peaks.append(peak_candidate)
                peaks_mean.remove(peak_candidate)
                i = 0
            else:
                i += 1 
                    
        except: 
            break 

    i = 0 
    while True:          
        
        try: 
            peak_candidate = peaks_max[i]
        
            if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                appended_peaks.append(peak_candidate)
                peaks_max.remove(peak_candidate)
                i = 0
            else:
                i += 1 
                    
        except: 
            break 
     
    return appended_peaks, sum_spectrum, highest_spectrum 

def oxford_parse():
    # Get user to select input directory(s)
    input_src = []
    while True: 
        get_path = filedialog.askdirectory( title = "Select input directory(s) for H5OINA, RPL, and RAW files. Press 'cancel' to stop adding folders")
        if get_path != '':
            input_src.append(get_path)
        else:
            break 
        
    # Get user to select output directory 
    output_src = filedialog.askdirectory( title = "Select output directory")
    
    # Parse directories for Oxford files and 2D images to use as inputs 
    print("Preprocessing...")
    print("")
    # Get list of all files in input directory. Create seperate file lists for oxford
    file_list = []
    file_list_stems = [] 
    oxford_file_list = []
    for src in input_src:
        for file in glob.glob(src + '/**/*', recursive=True):
            if Path(file).suffix in ['.tif', '.tiff', '.png', '.jpg', '.csv', '.xlsx', '.npy', '.h5oina', '.raw', '.rpl']:
                file_list.append(file)
                file_list_stems.append( os.path.split(file)[1] )
                if Path(file).suffix in ['.h5oina']: 
                    oxford_file_list.append(file)
                    
    # If oxford files are present, parse metadata and then stitch datacube and other available layers 
    ########
    if len(oxford_file_list) > 0:
       
        print("Parsing metadata...")
        print("")
        metadata, ebsd_phases = oxford_get_metadata(oxford_file_list)
    
        # find the montages available for each input directory
        unique_montages = list(np.unique(np.asarray( metadata['Montage Label'].values, dtype = str ) ) )
        
        # Remove empty montage names or nan if present 
        try: 
            unique_montages.remove('')
        except ValueError: 
            pass
        try: 
            unique_montages.remove('nan')
        except ValueError: 
            pass
        
        # parse each montage seperately and save as H5 
        for montage in unique_montages:
            print("Preprocessing: " + str(montage) )
            print("")
          
            # Stitch the datacube for the montage and return the autodetected peaks, sum of spectrum, and highest intensity spectrum 
            appended_peaks, sum_spectrum, highest_spectrum  = convert_oxford_RPL_to_H5(montage, metadata, file_list, output_src)
            
            # Make montage filepath 
            montage_file = os.path.join(output_src, 'Montage ' + str(montage) + '.h5')
            
            # Save autodetected peaks, sum of spectrum, and highest intensity spectrum for each montage 
            save_h5(montage_file, 'EDS', ['array', 'Autodetected Peak Bins'], np.array(appended_peaks ).astype(np.uint16) )
            save_h5(montage_file, 'EDS', ['array', 'Sum of Spectrum'], np.array(sum_spectrum ).astype(np.uint64) )
            save_h5(montage_file, 'EDS', ['array', 'Highest Intensity Spectrum'], np.array(highest_spectrum ).astype(np.uint64) )
            print("Saved datacube, autodetected peaks, sum of spectrum, highest intensity spectrum")
            print("")
            
            # Save the metadata for each montage seperately 
            
            for header in metadata.columns: 
                with h5py.File(montage_file, 'r+') as file: 
                    try:
                        file.create_dataset( 'Metadata/'+str(header) , data = metadata[str(header)][metadata['Montage Label'] == montage].astype('S'))
                    except ValueError:
                        pass
                    
                #save_h5(montage_file, 'Metadata', ['array', str(header)], metadata[str(header)][metadata['Montage Label'] == montage].astype('str') ) 
    
            print("Finding available backgrounds ")
            print("")
          
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
                ['EBSD', "Error"],
                ['EBSD', "Euler"], 
                ['EBSD', "Mean Angular Deviation"], 
                ['EBSD', "Pattern Center X"], 
                ['EBSD', "Pattern Center Y"], 
                ['EBSD', "Pattern Qualtiy"], 
                ['EBSD', "Phase"], 
                ['EBSD', "X"], 
                ['EBSD', "Y"] ]
            
            for specification in specifications:
                try: 
                    print("Stitching: " + str(specification) )
                    stitched = oxford_montage_stitcher(montage, metadata, specification, file_list, None, None)
                    save_h5(montage_file, specification[0], ['array', specification[1] ], np.array(stitched ).astype(np.uint16) )
                    
                except: 
                    pass 
            print("")
    
    ########
    else: 
        raise ValueError("ERROR: Oxford .h5oina files not found but Oxford RPL & RAW files specified for input format. Verify input file locations and formats and try again")
        




def bruker_parse_SEM():
    # choose bcf file(s) to open
    file_list = filedialog.askopenfilenames ( title = "Select .bcf file", filetypes=[("Bruker files", ".bcf")])
    output_src = filedialog.askdirectory( title = "Select output directory")
      
    print("Enter the name of this montage/dataset")
    montage = str( input("") ) 
    
    montage_file = os.path.join(output_src, 'Montage ' + str(montage) + '.h5')
    
    print("")
    print("Parsing metadata")
    file_list = list(file_list)
    for i, file_name in enumerate(file_list): 
        if any(x in file_name for x in ['[', ']']): 
            new_file_name = file_name
            try: 
                new_file_name = new_file_name.replace('[', '_')
            except: 
                pass 
            
            try: 
                new_file_name = new_file_name.replace(']', '_')
            except: 
                pass 
            
            os.rename(file_name, new_file_name)
            file_list[i] = new_file_name
    file_list = tuple(file_list)
    
    metadata = bruker_parse_metadata(file_list, montage)
    
    


    try: 
        file = h5py.File(montage_file, 'a')
    except: 
        file = h5py.File(montage_file, 'r+') 

    try:
        file.create_group('Metadata')
    except:
        pass 
    
    try:
        for header in metadata.columns: 
            file.create_dataset( 'Metadata/'+str(header) , data = metadata[str(header)][metadata['Montage Label'] == montage].astype('S'))
    except ValueError:
        pass
    
    try:
        file.close()
    except:
        pass 
                      
    print("")
    print("Stitching datacube")
    appended_peaks, sum_spectrum, highest_spectrum  = bruker_montage(montage_file, montage, metadata, file_list)
    save_h5(montage_file, 'EDS', ['array', 'Autodetected Peak Bins'], np.array(appended_peaks ).astype(np.uint16) )
    save_h5(montage_file, 'EDS', ['array', 'Sum of Spectrum'], np.array(sum_spectrum ).astype(np.uint16) )
    save_h5(montage_file, 'EDS', ['array', 'Highest Intensity Spectrum'], np.array(highest_spectrum ).astype(np.uint16) )
    gc.collect()
    
    return 

def bruker_parse_STEM(): 
    # choose bcf file(s) to open
    file_list = filedialog.askopenfilenames ( title = "Select .bcf file", filetypes=[("Bruker files", ".bcf")])
    output_src = filedialog.askdirectory( title = "Select output directory")
        
    #file_path = file_list[0]
    if isinstance(file_list, str):
        file_list = list(file_list)
        
    for file_path in file_list: 
        # load all data with hyperspy 
        try: 
            all_data = hs.load(file_path)
        except TypeError: 
            print("ERROR:" + str(file_path) + " is not a compatable format for Hyperspy")
            continue 
        
        # Find and load the EDS data 
        for i in range(len(all_data)):
            data = all_data[i]
            
            if (data.metadata.Signal.signal_type == "EDS_TEM") or (data.metadata.Signal.signal_type == "EDS_SEM"): 
                montage = Path(file_path).stem
                montage_file = os.path.join(output_src, 'Montage ' + str(montage) + '.h5')
                
                print("")
                print("Processing: " + str(montage) )
                print("Parsing metadata")
                metadata = bruker_parse_bcf_metadata(montage, "Bruker", "Unknown", "Unknown", file_path)
                
                try: 
                    file = h5py.File(montage_file, 'a')
                except: 
                    file = h5py.File(montage_file, 'r+') 

                try:
                    file.create_group('Metadata')
                except:
                    pass 
                
                try:
                    for header in metadata.columns: 
                        file.create_dataset( 'Metadata/'+str(header) , data = metadata[str(header)][metadata['Montage Label'] == montage].astype('S'))
                except ValueError:
                    pass
                
                try:
                    file.close()
                except:
                    pass 
                
                print("Parsing datacube")               
                appended_peaks, sum_spectrum, highest_spectrum = bruker_montage(montage_file, montage, metadata, str(file_path))
                save_h5(montage_file, 'EDS', ['array', 'Autodetected Peak Bins'], np.array(appended_peaks ).astype(np.uint16) )
                save_h5(montage_file, 'EDS', ['array', 'Sum of Spectrum'], np.array(sum_spectrum ).astype(np.uint16) )
                save_h5(montage_file, 'EDS', ['array', 'Highest Intensity Spectrum'], np.array(highest_spectrum ).astype(np.uint16) )
                gc.collect()
                
    return 




def bruker_montage(montage_file, montage, metadata, file_list): 
    
    if isinstance(file_list, tuple): 
        x_scale = np.unique( metadata['EDS X Step Size (um)'] )
        y_scale = np.unique( metadata['EDS Y Step Size (um)'] )
        
        # get the XY coordinates for each field 
        x_list =  list( np.unique( metadata['EDS Stage X Position (mm)'][metadata['Montage Label'] == montage] ) )
        y_list =  list( np.unique( metadata['EDS Stage Y Position (mm)'][metadata['Montage Label'] == montage] ) ) 
       
        x_size = np.unique(metadata['EDS Number of X Cells'])
        y_size = np.unique(metadata['EDS Number of Y Cells'])
        
        # find minimum coodinates so that we can create an appropriately sized array 
        x_min = int( math.ceil(min(x_list)*1000.0/x_scale))
        y_min = int( math.ceil(min(y_list)*1000.0/y_scale))
        
        # determine how large of an array is neeed 
        x_range = int (math.ceil( ( 1000*(max(x_list) - min(x_list))/x_scale ) ) + x_size )
        y_range = int( math.ceil( ( 1000*(max(y_list) - min(y_list))/y_scale ) ) + y_size )
      
        eds_layers = np.unique( metadata["EDS Number of Channels"] )
        
        array = np.zeros( shape = (y_range, x_range, len(eds_layers)), dtype = np.uint16 )
    
        for file_path in tqdm(file_list, total = len(file_list) ): 
            # load all data with hyperspy 
            try: 
                all_data = hs.load(file_path)
            except: 
                continue 
            
            # Find and load the EDS data 
            for i in range(len(all_data)):
                data = all_data[i]
                
                if data.metadata.General.title == 'EDX':
                
                    try:       
                        eds_x_pos = data.original_metadata.Stage.X                          # stage position in um 
                        eds_y_pos = data.original_metadata.Stage.Y                          # stage position in um 
                
                        x_location = (eds_x_pos)/x_scale - x_min
                        y_location = (eds_y_pos)/y_scale - y_min
    
                        if x_location < 0:
                            x_location = 0
                        else:
                            x_location = int(x_location)
                
                        if y_location < 0:
                            y_location = 0
                        else:
                            y_location = int(y_location)
                        
                        block = data.data
                        temp_block = np.empty( shape = block.shape, dtype = np.uint16 )
                        for i in range(block.shape[2]):
                            temp_block[:,:, i ] = np.flip( block[:,:,i], 1)
                        block = temp_block.copy()
                        del temp_block
                        
                    except: 
                        pass 
    
                    save_file = h5py.File(montage_file, 'a') 
    
                    try:
                        save_file.create_group('EDS')
                    except:
                        pass 
                
                    try: 
                        save_file['EDS'].create_dataset('Xray Spectrum', shape = (y_range, x_range, block.shape[2] ), chunks= True, dtype = 'int16')
                    except: 
                        pass 
              
                    try: 
                        save_file['EDS'].create_dataset('Xray Intensity', shape = (y_range, x_range ), chunks=True, dtype = 'int64')
                    except: 
                        pass 
                    
                    save_file['EDS']['Xray Spectrum'][y_location:y_location + int(y_size), x_location:x_location + int(x_size), :] = block 
                    save_file['EDS']['Xray Intensity'][y_location:y_location + int(y_size), x_location:x_location + int(x_size)] = np.sum( block, axis = 2)
    
                    save_file.close() 
                    
                    new_highest_spectrum =  np.max( block[:, :, :], axis = (0,1)).flatten() 
                    new_sum_spectrum =  np.sum( block[:, :, :], axis = (0,1)).flatten() 
                    
                    # Sum new block with existing data
                    try: 
                        sum_spectrum += new_sum_spectrum
                    except NameError:
                        sum_spectrum = np.zeros( shape = new_sum_spectrum.shape, dtype = np.float64 )
                        sum_spectrum += new_sum_spectrum
                    
                    # Keep only highest peaks from every block of data 
                    try: 
                        highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                    except NameError:
                        highest_spectrum = np.zeros( shape = new_highest_spectrum.shape , dtype = np.float64 )
                        highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                        
                    gc.collect()
                    
            # Find peaks from both the average spectrum intensity as well as the maximum spectrum intensity 
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
            appended_peaks = []
            search_width = 3
            peaks_mean, _ = scipy.signal.find_peaks(sum_spectrum, 
                                               prominence = (0.10*statistics.median( sum_spectrum ), None),
                                               distance = 5
                                               )
            
            peaks_max, _ = scipy.signal.find_peaks(highest_spectrum, 
                                               height = (np.percentile(highest_spectrum, 95), None ),
                                               prominence = 2,
                                               distance = 5
                                               )
            
            peaks_mean = list( peaks_mean )
            peaks_max = list( peaks_max)  
            
            # Sort candidate peaks by the x ray intensity. We want to add the strongest peaks first
            a = np.argsort(sum_spectrum[peaks_mean]) 
            peaks_mean[:] = [peaks_mean[i] for i in a][:len(peaks_mean)]             
            
            a = np.argsort(highest_spectrum[peaks_max]) 
            peaks_max[:] = [peaks_max[i] for i in a][:len(peaks_max)]      
            
            # Iterate through all candidate peaks that were found for the montage and include the peak if it is more than X bins from previously detected peaks 
            i = 0 
            while True:          
                
                try: 
                    peak_candidate = peaks_mean[i]
                
                    if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                        appended_peaks.append(peak_candidate)
                        peaks_mean.remove(peak_candidate)
                        i = 0
                    else:
                        i += 1 
                            
                except: 
                    break 

            i = 0 
            while True:          
                
                try: 
                    peak_candidate = peaks_max[i]
                
                    if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                        appended_peaks.append(peak_candidate)
                        peaks_max.remove(peak_candidate)
                        i = 0
                    else:
                        i += 1 
                            
                except: 
                    break 
             
        return appended_peaks, sum_spectrum, highest_spectrum 
        
    elif isinstance(file_list, str): 
        file_path = file_list
        # load all data with hyperspy 
        all_data = hs.load(file_path)
        
        # Find and load the EDS data 
        for i in range(len(all_data)):
            data = all_data[i]
            
            if data.metadata.General.title == 'EDX':
            
                try:       
                    block = data.data
                
                except: 
                    pass 

                save_file = h5py.File(montage_file, 'a') 

                try:
                    save_file.create_group('EDS')
                except:
                    pass 
            
                try: 
                    save_file['EDS'].create_dataset('Xray Spectrum', shape = (block.shape[0], block.shape[1], block.shape[2] ), chunks=(True, True, 50), dtype = 'int64')
                except: 
                    pass 
                
                try: 
                    save_file['EDS'].create_dataset('Xray Intensity', shape = (block.shape[0], block.shape[1]), chunks=True, dtype = 'int64')
                except: 
                    pass 
                
                save_file['EDS']['Xray Intensity'][:,:] = np.sum( block, axis = 2)
                save_file['EDS']['Xray Spectrum'][:,:,:] = block 
    
                save_file.close() 
                
                new_highest_spectrum =  np.max( block[:, :, :], axis = (0,1)).flatten() 
                new_sum_spectrum =  np.sum( block[:, :, :], axis = (0,1)).flatten() 
                
                # Sum new block with existing data
                try: 
                    sum_spectrum += new_sum_spectrum
                except NameError:
                    sum_spectrum = np.zeros( shape = new_sum_spectrum.shape, dtype = np.float64 )
                    sum_spectrum += new_sum_spectrum
                
                # Keep only highest peaks from every block of data 
                try: 
                    highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                except NameError:
                    highest_spectrum = np.zeros( shape = new_highest_spectrum.shape , dtype = np.float64 )
                    highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                    
                gc.collect()
                
                # Find peaks from both the average spectrum intensity as well as the maximum spectrum intensity 
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
                appended_peaks = []
                search_width = 3
                peaks_mean, _ = scipy.signal.find_peaks(sum_spectrum, 
                                                   prominence = (0.10*statistics.median( sum_spectrum ), None),
                                                   distance = 5
                                                   )
                
                peaks_max, _ = scipy.signal.find_peaks(highest_spectrum, 
                                                   height = (np.percentile(highest_spectrum, 95), None ),
                                                   prominence = 2,
                                                   distance = 5
                                                   )
                
                peaks_mean = list( peaks_mean )
                peaks_max = list( peaks_max)  
                
                # Sort candidate peaks by the x ray intensity. We want to add the strongest peaks first
                a = np.argsort(sum_spectrum[peaks_mean]) 
                peaks_mean[:] = [peaks_mean[i] for i in a][:len(peaks_mean)]             
                
                a = np.argsort(highest_spectrum[peaks_max]) 
                peaks_max[:] = [peaks_max[i] for i in a][:len(peaks_max)]      
                
                # Iterate through all candidate peaks that were found for the montage and include the peak if it is more than X bins from previously detected peaks 
                i = 0 
                while True:          
                    
                    try: 
                        peak_candidate = peaks_mean[i]
                    
                        if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                            appended_peaks.append(peak_candidate)
                            peaks_mean.remove(peak_candidate)
                            i = 0
                        else:
                            i += 1 
                                
                    except: 
                        break 
    
                i = 0 
                while True:          
                    
                    try: 
                        peak_candidate = peaks_max[i]
                    
                        if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                            appended_peaks.append(peak_candidate)
                            peaks_max.remove(peak_candidate)
                            i = 0
                        else:
                            i += 1 
                                
                    except: 
                        break 
                
                return appended_peaks, sum_spectrum, highest_spectrum 
            



def bruker_parse_metadata(file_list, montage_name): 
    # Input list of file addresses to parse for metadata. 
    # Returns a dataframe of metadata 
    # Note: Hyperspy does not parse Bruker EBSD data as of 6 June 2022
    #####################################################################################
    
    if isinstance(file_list, tuple): 
        
        initializing_flag = True
        for i, file_path in enumerate( tqdm(file_list, total = len(file_list) ) ): 
    
            # skip file_path file that are not bcf 
            if Path(file_path).suffix == '.bcf': 
                
                try: 
                    if initializing_flag: 
                        metadata = bruker_parse_bcf_metadata(montage_name, "Bruker", "Unknown", "Unknown", file_path)
                      
                        initializing_flag = False
                    else: 
                        metadata_placeholder = bruker_parse_bcf_metadata(montage_name, "Bruker", "Unknown", "Unknown", file_path)
                        
                        metadata = pd.concat( [metadata, metadata_placeholder], ignore_index = True )
                except:
                    pass
                    
        return metadata
    elif isinstance(file_list, str): 
        metadata = bruker_parse_bcf_metadata(montage_name, "Bruker", "Unknown", "Unknown", file_list)
        return metadata
        


def bruker_parse_bcf_metadata(montage_name, hardware_vendor, software_version, format_version, file_path):

    # load all data with hyperspy 
    all_data = hs.load(file_path)
    
    # Find and load the EDS data 
    for i in range(len(all_data)):
        data = all_data[i]
        
        if data.metadata.General.title == 'EDX':
            
            ##########
            # if available, collect EDS metadata
            try:   
                eds_project =                           np.nan
                eds_specimen =                          data.metadata.Sample.name 
                eds_site =                              np.nan
                eds_voltage =                           data.original_metadata.Microscope.HV  
                eds_magnification =                     data.original_metadata.Microscope.Mag
                eds_field =                             Path(str(file_path)).stem
                eds_binning =                           np.nan
                eds_bin_width =                         data.original_metadata.Spectrum.CalibLin*1_000        # eV bin width
                eds_start_channel =                     data.original_metadata.Spectrum.CalibAbs*1_000        # eV offset? 
                eds_averaged_frames =                   np.nan
                eds_process_time =                      np.nan
                eds_x_cells =                           data.original_metadata.DSP_Configuration.ImageWidth
                eds_y_cells =                           data.original_metadata.DSP_Configuration.ImageHeight
                eds_x_step =                            data.original_metadata.Microscope.DX                    # X axis um per pixel       
                eds_y_step =                            data.original_metadata.Microscope.DY                    # Y axis um per pixel 
                try: 
                    eds_real_time_sum =                     data.metadata.Acquisition_instrument.SEM.Detector.EDS.real_time
                    eds_live_time =                         np.nan
                except AttributeError:
                    eds_real_time_sum =                     data.metadata.Acquisition_instrument.TEM.Detector.EDS.real_time
                    eds_live_time =                         np.nan
                eds_unique_vendor_id =                  np.nan
                try: 
                    eds_detector_azimuth_radians =          data.metadata.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle
                    eds_detector_elevation_radians =        data.metadata.Acquisition_instrument.SEM.Detector.EDS.elevation_angle
                except AttributeError:
                    eds_detector_azimuth_radians =          data.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
                    eds_detector_elevation_radians =        data.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle
                eds_hardware_detector_serial_number =   np.nan
                try:
                    eds_hardware_detector_model =           data.metadata.Acquisition_instrument.SEM.Detector.EDS.detector_type # make, model 
                except AttributeError:
                    eds_hardware_detector_model =           data.metadata.Acquisition_instrument.TEM.Detector.EDS.detector_type # make, model 
                eds_drift_correction =                  np.nan
                eds_num_bins =                          data.original_metadata.Spectrum.ChannelCount
                eds_processor =                         np.nan
                eds_date =                              data.metadata.General.date  
                try:
                    eds_x_pos =                             data.original_metadata.Stage.X / 1_000.0                # stage position in um 
                    eds_y_pos =                             data.original_metadata.Stage.Y / 1_000.0                # stage position in um 
                    eds_z_pos =                             data.original_metadata.Stage.Z / 1_000.0                # stage position in um 
                    eds_tilt_radians  =                     data.original_metadata.Stage.Tilt                       # stage tile in 
                    eds_rotation_radians =                  data.original_metadata.Stage.Rotation                   # working distance in mm 
                except AttributeError:
                    eds_x_pos =                             np.nan                                                  # stage position in um 
                    eds_y_pos =                             np.nan                                                  # stage position in um 
                    eds_z_pos =                             np.nan                                                  # stage position in um 
                    eds_tilt_radians  =                     np.nan                                                  # stage tile in 
                    eds_rotation_radians =                  np.nan                                                  # working distance in mm 
                eds_strobe_area =                       np.nan
                eds_strobe_FWHM =                       np.nan
                eds_window_type =                       np.nan
                eds_working_distance =                  data.original_metadata.Microscope.WD                        # working distance in mm 
                
            except: 
                eds_project =                           np.nan 
                eds_specimen =                          np.nan 
                eds_site =                              np.nan 
                eds_voltage =                           np.nan 
                eds_magnification =                     np.nan 
                eds_field =                             np.nan 
                eds_binning =                           np.nan 
                eds_bin_width =                         np.nan 
                eds_start_channel =                     np.nan 
                eds_averaged_frames =                   np.nan 
                eds_process_time =                      np.nan 
                eds_x_cells =                           np.nan 
                eds_y_cells =                           np.nan 
                eds_x_step =                            np.nan 
                eds_y_step =                            np.nan 
                eds_real_time_sum =                     np.nan
                eds_live_time =                         np.nan
                eds_unique_vendor_id =                  np.nan
                eds_detector_azimuth_radians =          np.nan
                eds_detector_elevation_radians =        np.nan
                eds_hardware_detector_serial_number =   np.nan
                eds_hardware_detector_model =           np.nan
                eds_drift_correction =                  np.nan
                eds_num_bins =                          np.nan
                eds_processor =                         np.nan
                eds_date =                              np.nan
                eds_magnification =                     np.nan
                eds_x_pos =                             np.nan
                eds_y_pos =                             np.nan
                eds_z_pos =                             np.nan
                eds_tilt_radians  =                     np.nan
                eds_rotation_radians =                  np.nan
                eds_strobe_area =                       np.nan
                eds_strobe_FWHM =                       np.nan
                eds_window_type =                       np.nan
                eds_working_distance =                  np.nan
                
    ##########
    # if available, collect SE/BSE metadata
    try:   
        electron_image_project =                np.nan 
        electron_image_specimen =               data.metadata.Sample.name                              # sample name     
        electron_image_site =                   np.nan 
        electron_image_field =                  np.nan 
        electron_image_voltage =                data.original_metadata.Microscope.HV                   # KeV voltage 
        electron_image_magnification =          data.original_metadata.Microscope.Mag                  # magnification 
        electron_image_unique_vendor_id =       np.nan
        electron_image_drift_correction =       np.nan
        electron_image_dwell_time =             np.nan
        electron_image_average_frames =         np.nan
        electron_image_x_cells =                data.original_metadata.DSP_Configuration.ImageWidth
        electron_image_y_cells =                data.original_metadata.DSP_Configuration.ImageHeight
        electron_image_x_step =                 data.original_metadata.Microscope.DX                    # X axis um per pixel               
        electron_image_y_step =                 data.original_metadata.Microscope.DY                    # Y axis um per pixel 
        electron_image_date =                   data.metadata.General.date  
        electron_image_time =                   data.metadata.General.time                              #  
        try: 
            electron_image_x_pos =                  data.original_metadata.Stage.X                          # stage position in um 
            electron_image_y_pos =                  data.original_metadata.Stage.Y                          # stage position in um 
            electron_image_z_pos =                  data.original_metadata.Stage.Z                          # stage position in um 
            electron_image_tilt_radians  =          data.original_metadata.Stage.Tilt                       # stage tile in 
            electron_image_rotation_radians =       data.original_metadata.Stage.Rotation                   # working distance in mm 
        except AttributeError:
            electron_image_x_pos =                   np.nan                         # stage position in um 
            electron_image_y_pos =                   np.nan                         # stage position in um 
            electron_image_z_pos =                   np.nan                         # stage position in um 
            electron_image_tilt_radians  =           np.nan                       # stage tile in 
            electron_image_rotation_radians =        np.nan                   # working distance in mm 
        electron_image_working_distance =       data.original_metadata.Microscope.WD                    # working distance in mm 
    
    except: 
        electron_image_project =                np.nan 
        electron_image_specimen =               np.nan  
        electron_image_site =                   np.nan 
        electron_image_field =                  np.nan 
        electron_image_voltage =                np.nan 
        electron_image_magnification =          np.nan 
        electron_image_unique_vendor_id =       np.nan 
        electron_image_drift_correction =       np.nan 
        electron_image_dwell_time =             np.nan 
        electron_image_average_frames =         np.nan 
        electron_image_x_cells =                np.nan 
        electron_image_y_cells =                np.nan 
        electron_image_x_step =                 np.nan 
        electron_image_y_step =                 np.nan 
        electron_image_date =                   np.nan 
        electron_image_x_pos =                  np.nan 
        electron_image_y_pos =                  np.nan 
        electron_image_z_pos =                  np.nan 
        electron_image_tilt_radians  =          np.nan 
        electron_image_rotation_radians =       np.nan 
        electron_image_working_distance =       np.nan 
    
    """
    ##########
    # if available, collect EBSD metadata
    try: 
        ebsd_project =                          
        ebsd_specimen =                         
        ebsd_site =                             
        ebsd_voltage =                          
        ebsd_magnification =                    
        ebsd_field =                            
        ebsd_acq_date =                        
        ebsd_acq_speed =                       
        ebsd_acq_time =                       
        ebsd_unique_ID =                   
        ebsd_background_correction =           
        ebsd_band_detection_mode =           
        ebsd_bounding_box =                  
        ebsd_bounding_box_x 
        ebsd_bounding_box_y 
        ebsd_camera_binning_mode =             
        ebsd_camera_exposure_time =            
        ebsd_camera_gain =                      
        ebsd_detector_insertion_distance =    
        ebsd_detector_orientation_euler =     
        ebsd_detector_orientation_euler_a =    
        ebsd_detector_orientation_euler_b =    
        ebsd_detector_orientation_euler_c =    
        ebsd_drift_correction =              
        ebsd_hit_rate =                       
        ebsd_hough_resolution =              
        ebsd_indexing_mode =                
        ebsd_lens_distortion =               
        ebsd_lens_field_view =             
        ebsd_number_bands_detected =          
        ebsd_number_frames_averaged =         
        ebsd_pattern_height =                   
        ebsd_pattern_width =               
        ebsd_project_file =                   
        ebsd_project_notes =                  
        ebsd_relative_offset =                
        ebsd_relative_offset_x = 
        ebsd_relative_offset_y = 
        ebsd_relative_size =                  
        ebsd_relative_size_x = 
        ebsd_relative_size_y = 
        ebsd_scanning_rotation_angle =        
        ebsd_site_notes =                       
        ebsd_specimen_notes =                 
        ebsd_specimen_orientation =            
        ebsd_specimen_orientation_a =         
        ebsd_specimen_orientation_b =         
        ebsd_specimen_orientation_c =         
        ebsd_stage_x =                        
        ebsd_stage_y =                      
        ebsd_stage_z =                        
        ebsd_stage_rotation =                 
        ebsd_stage_tilt =                      
        ebsd_static_background_correction =    
        ebsd_tilt_angle =                       
        ebsd_tilt_axis =                       
        ebsd_working_distance =               
        ebsd_xcells =                        
        ebsd_ycells =                       
        ebsd_xstep =                       
        ebsd_ystep =                        
        
    except: 
        
    """
    
    ebsd_project =                          np.nan
    ebsd_specimen =                         np.nan
    ebsd_site =                             np.nan
    ebsd_voltage =                          np.nan
    ebsd_magnification =                    np.nan
    ebsd_field =                            np.nan
    ebsd_acq_date =                         np.nan
    ebsd_acq_speed =                        np.nan 
    ebsd_acq_time =                         np.nan
    ebsd_label =                            np.nan
    ebsd_unique_ID =                        np.nan 
    ebsd_background_correction =            np.nan
    ebsd_band_detection_mode =              np.nan
    ebsd_beam_voltage =                     np.nan
    ebsd_bounding_box =                     np.nan
    ebsd_bounding_box_x =                   np.nan
    ebsd_bounding_box_y =                   np.nan
    ebsd_camera_binning_mode =              np.nan
    ebsd_camera_exposure_time =             np.nan
    ebsd_camera_gain =                      np.nan
    ebsd_detector_insertion_distance =      np.nan
    ebsd_detector_orientation_euler =       np.nan
    ebsd_detector_orientation_euler_a =     np.nan
    ebsd_detector_orientation_euler_b =     np.nan
    ebsd_detector_orientation_euler_c =     np.nan 
    ebsd_drift_correction =                 np.nan
    ebsd_hit_rate =                         np.nan
    ebsd_hough_resolution =                 np.nan
    ebsd_indexing_mode =                    np.nan
    ebsd_lens_distortion =                  np.nan
    ebsd_lens_field_view =                  np.nan
    ebsd_magnification =                    np.nan
    ebsd_number_bands_detected =            np.nan
    ebsd_number_frames_averaged =           np.nan
    ebsd_pattern_height =                   np.nan
    ebsd_pattern_width =                    np.nan
    ebsd_project_file =                     np.nan
    ebsd_project_label =                    np.nan
    ebsd_project_notes =                    np.nan
    ebsd_relative_offset =                  np.nan
    ebsd_relative_offset_x =                np.nan
    ebsd_relative_offset_y =                np.nan
    ebsd_relative_size =                    np.nan
    ebsd_relative_size_x =                  np.nan
    ebsd_relative_size_y =                  np.nan
    ebsd_scanning_rotation_angle =          np.nan
    ebsd_site_label =                       np.nan
    ebsd_site_notes =                       np.nan
    ebsd_specimen_label =                   np.nan
    ebsd_specimen_notes =                   np.nan
    ebsd_specimen_orientation =             np.nan
    ebsd_specimen_orientation_a =           np.nan
    ebsd_specimen_orientation_b =           np.nan
    ebsd_specimen_orientation_c =           np.nan
    ebsd_stage_x =                          np.nan
    ebsd_stage_y =                          np.nan
    ebsd_stage_z =                          np.nan
    ebsd_stage_rotation =                   np.nan
    ebsd_stage_tilt =                       np.nan
    ebsd_static_background_correction =     np.nan
    ebsd_tilt_angle =                       np.nan
    ebsd_tilt_axis =                        np.nan
    ebsd_working_distance =                 np.nan
    ebsd_xcells =                           np.nan
    ebsd_ycells =                           np.nan
    ebsd_xstep =                            np.nan
    ebsd_ystep =                            np.nan
    
    ##########
    metadata = [] 
    
    metadata.append( (
        montage_name,
        hardware_vendor,
        software_version, 
        format_version, 
    
        electron_image_project,
        electron_image_specimen,
        electron_image_site,
        electron_image_field,
        electron_image_voltage,
        electron_image_magnification,
        electron_image_unique_vendor_id,
        electron_image_drift_correction,
        electron_image_dwell_time,
        electron_image_average_frames,
        electron_image_x_cells,
        electron_image_y_cells,
        electron_image_x_step,
        electron_image_y_step,
        electron_image_date,
        electron_image_x_pos,
        electron_image_y_pos,
        electron_image_z_pos,
        electron_image_tilt_radians,
        electron_image_rotation_radians,
        electron_image_working_distance,
        
        eds_project,
        eds_specimen,
        eds_site,
        eds_field,
        eds_voltage,
        eds_magnification,
        eds_binning,
        eds_bin_width,
        eds_start_channel,
        eds_averaged_frames,
        eds_process_time,
        eds_x_cells,
        eds_y_cells,
        eds_x_step,
        eds_y_step,
        eds_real_time_sum,
        eds_live_time,
        eds_unique_vendor_id,
        eds_detector_azimuth_radians,
        eds_detector_elevation_radians,
        eds_hardware_detector_serial_number,
        eds_hardware_detector_model,
        eds_drift_correction,
        eds_num_bins,
        eds_processor,
        eds_date,
        eds_x_pos,
        eds_y_pos,
        eds_z_pos,
        eds_tilt_radians,
        eds_rotation_radians,
        eds_strobe_area,
        eds_strobe_FWHM,
        eds_window_type,
        eds_working_distance,
        
        ebsd_project,
        ebsd_specimen,
        ebsd_site,
        ebsd_field,
        ebsd_voltage,
        ebsd_magnification,
        ebsd_acq_date,
        ebsd_acq_speed,
        ebsd_acq_time,
        ebsd_unique_ID,
        ebsd_background_correction,
        ebsd_band_detection_mode,
        ebsd_bounding_box_x,
        ebsd_bounding_box_y,
        ebsd_camera_binning_mode,
        ebsd_camera_exposure_time,
        ebsd_camera_gain,
        ebsd_detector_insertion_distance,
        ebsd_detector_orientation_euler_a,
        ebsd_detector_orientation_euler_b,
        ebsd_detector_orientation_euler_c,
        ebsd_drift_correction,
        ebsd_hit_rate,
        ebsd_hough_resolution,
        ebsd_indexing_mode,
        ebsd_lens_distortion,
        ebsd_lens_field_view,
        ebsd_magnification,
        ebsd_number_bands_detected,
        ebsd_number_frames_averaged,
        ebsd_pattern_height,
        ebsd_pattern_width,
        ebsd_project_file,
        ebsd_project_notes,
        ebsd_relative_offset_x,
        ebsd_relative_offset_y,      
        ebsd_relative_size_x,        
        ebsd_relative_size_y,
        ebsd_scanning_rotation_angle,
        ebsd_site_notes,
        ebsd_specimen_notes,
        ebsd_specimen_orientation_a,
        ebsd_specimen_orientation_b,
        ebsd_specimen_orientation_c,
        ebsd_stage_x,
        ebsd_stage_y ,
        ebsd_stage_z,
        ebsd_stage_rotation,
        ebsd_stage_tilt,
        ebsd_static_background_correction,
        ebsd_tilt_angle,
        ebsd_tilt_axis,
        ebsd_working_distance,
        ebsd_xcells,
        ebsd_ycells,
        ebsd_xstep,
        ebsd_ystep
        ) ) 
    
    metadata = np.asarray(metadata, dtype=object)
    
    ##########    
    cols = ["Montage Label",
            "Hardware Vendor",
            "Software Version", 
            "Format Version", 
    
            "SEM Project",
            "SEM Specimen",
            "SEM Site",
            "SEM Field",
            "SEM Voltage (KeV)",
            "SEM Magnification",            
            "SEM Vendor Unique ID",
            "SEM Drift Correction",
            "SEM Dwell Time (us)",
            "SEM Average Frames",
            "SEM Number of X Cells",
            "SEM Number of Y Cells",
            "SEM X Step Size (um)",
            "SEM Y Step Size (um)",
            "SEM Date",
            "SEM Stage X Position (mm)",
            "SEM Stage Y Position (mm)",
            "SEM Stage Z Position (mm)",
            "SEM Stage Tilt (rad)",
            "SEM Stage Rotation (rad)",
            "SEM Working Distance (mm)",
            
            "EDS Project",
            "EDS Specimen",
            "EDS Site",
            "EDS Field",
            "EDS Voltage (KeV)",
            "EDS Magnification",
            "EDS Binning Factor",
            "EDS Voltage Bin Width (eV)",
            "EDS Starting Bin Voltage (eV)",
            "EDS Number of Averaged Frames",
            "EDS Process Time",
            "EDS Number of X Cells",
            "EDS Number of Y Cells",
            "EDS X Step Size (um)",
            "EDS Y Step Size (um)",
            "EDS Real Time Sum (s)",
            "EDS Live Time (s)",
            "EDS Vendor Unique ID",
            "EDS Azimuth Angle (rad)",
            "EDS Detector Angle (rad)",
            "EDS Detector Serial Number",
            "EDS Detector Model Number",
            "EDS Drift Correction",
            "EDS Number of Channels",
            "EDS Processor Type",
            "EDS Date",
            "EDS Stage X Position (mm)",
            "EDS Stage Y Position (mm)",
            "EDS Stage Z Position (mm)",
            "EDS Stage Tilt (rad)",
            "EDS Stage Rotation (rad)",
            "EDS Strobe Area",
            "EDS Strobe FWHM (ev)",
            "EDS Window Type",
            "EDS Working Distance (mm)",
            
            "EBSD Project",
            "EBSD Specimen",
            "EBSD Site",
            "EBSD Field",
            "EBSD Voltage (KeV)",
            "EBSD Magnification",
            "EBSD Acquisition Date",
            "EBSD Acquisition Speed (Hz)",
            "EBSD Acquisition Time (s)",
            "EBSD Vendor Unique ID",
            "EBSD Auto Background Correction",
            "EBSD Band Detection Mode",
            "EBSD Bounding Box X (um)",
            "EBSD Bounding Box Y (um)",
            "EBSD Camera Binning Mode",
            "EBSD Camera Exposure Time (ms)",
            "EBSD Camera Gain",
            "EBSD Detector Insertion Distance (mm)",
            "EBSD Detector Orientation Euler A (rad)",
            "EBSD Detector Orientation Euler B (rad)",
            "EBSD Detector Orientation Euler C (rad)",
            "EBSD Drift Correction",
            "EBSD Hit Rate",
            "EBSD Hough Resolution",
            "EBSD Indexing Mode",
            "EBSD Lens Distortion",
            "EBSD Lens Field View (mm)",
            "EBSD Magnification",
            "EBSD Number Bands Detected",
            "EBSD Number Frames Averaged",
            "EBSD Pattern Height (px)",
            "EBSD Pattern Width (px)",
            "EBSD Project File",
            "EBSD Project Notes",
            "EBSD Relative Offset X",
            "EBSD Relative Offset Y",
            "EBSD Relative Size X",
            "EBSD Relative Size Y",
            "EBSD Scanning Rotation Angle (rad)",
            "EBSD Site Notes",
            "EBSD Specimen Notes",
            "EBSD Specimen Orientation A",
            "EBSD Specimen Orientation B",
            "EBSD Specimen Orientation C",
            "EBSD Stage X Position (mm)",
            "EBSD Stage Y Position (mm)" ,
            "EBSD Stage Z Position (mm)",
            "EBSD Stage Rotation (rad)",
            "EBSD Stage Tilt (rad)",
            "EBSD Static Background Correction",
            "EBSD Tilt Angle (rad)",
            "EBSD Tilt Axis",
            "EBSD Working Distance (mm)",
            "EBSD Number X Pixels",
            "EBSD Number Y Pixels",
            "EBSD X Step Size (um)",
            "EBSD Y Step Size (um)"]
    
    metadata = pd.DataFrame( data = metadata, columns = cols)
    
    return metadata
    ##########


def EDAX_parse(): 
    output_src = filedialog.askdirectory( title = "Select output directory")
    file_path = filedialog.askopenfilename( title = "Select HDF5 File", filetypes=[("H5 files", ".h5"), ("H5 files", ".h5oina")])
    file = h5py.File(os.path.join(file_path), "r")
    
    projects = list(file.keys())
    for project in projects: 
        samples = list(file[str(project)].keys())
        for sample in samples: 
            areas = list(file[str(project)][str(sample)].keys())
            for area in areas: 
                if "Area" in area: 
                    X = list( file[str(project)][str(sample)][str(area)].keys() )
                    for montage in X: 
                        if "Montage" in montage: 
       
                            Y = list(file[str(project)][str(sample)][str(area)][str(montage)].keys())
                            for field in tqdm(Y): 
                                if "Field" in field: 
                                    #raise Exception() 
                                    #print(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGEIPR']['MicronsPerPixelX'][0])
                                    
                                    
                                    #file[str(project)][str(sample)][str(area)][str(montage)]['MAPIMAGECOLLECTIONPARAMS'][...]
                                    
                                    
                                    dat = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['SPC'][...]
                                    
                                    
                                    """
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['SpotSize'][0]
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)].keys()
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS'][...]
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS'][...]
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['ELEMENTOVRLAYIMGCOLLECTIONPARAMS'][...]
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGEIPR'][...]
                                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS'][...]
                                    """
                                    
                                    
                                    hardware_vendor =   "EDAX" 
                                    software_version =  dat['AppVersion'][0]
                                    format_version =    np.nan
                                    project_name =      project
                                    site_name =         area      
                                    specimen_name =     sample      
                                    field_name =        field    
                                    montage_name =      montage
                                    
                                    electron_image_project =                project_name 
                                    electron_image_specimen =               specimen_name
                                    electron_image_site =                   site_name
                                    electron_image_field =                  field_name
                                    electron_image_voltage =                file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['KV'][0]
                                    electron_image_magnification =          file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Magnification'][0]
                                    electron_image_unique_vendor_id =       file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Id'][0] # unique id 
                                    electron_image_drift_correction =       np.nan 
                                    electron_image_dwell_time =             np.nan 
                                    electron_image_average_frames =         np.nan 
                                    electron_image_x_cells =                int(file[str(project)][str(sample)][str(area)]['FOVIMAGECOLLECTIONPARAMS']['Width'])
                                    electron_image_y_cells =                int(file[str(project)][str(sample)][str(area)]['FOVIMAGECOLLECTIONPARAMS']['Height'])
                                    electron_image_x_step =                 file[str(project)][str(sample)][str(area)][str(montage)]['FOVIMAGECOLLECTIONPARAMS']['MicronsPerPixelX'][0]
                                    electron_image_y_step =                 file[str(project)][str(sample)][str(area)][str(montage)]['FOVIMAGECOLLECTIONPARAMS']['MicronsPerPixelY'][0]
                                    electron_image_date =                   np.nan 
                                    electron_image_x_pos =                  file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageXPosition'][0]
                                    electron_image_y_pos =                  file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageYPosition'][0]
                                    electron_image_z_pos =                  file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageZPosition'][0] 
                                    electron_image_tilt_radians  =          math.radians(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Tilt'][0]) 
                                    electron_image_rotation_radians =       math.radians(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Rotation'][0])
                                    electron_image_working_distance =       file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['WD'][0]
                                    
                                    eds_project =                           project_name 
                                    eds_specimen =                          specimen_name
                                    eds_site =                              site_name
                                    eds_voltage =                           file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['KV'][0]
                                    eds_magnification =                     file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Magnification'][0]
                                    eds_field =                             field_name
                                    eds_binning =                           np.nan 
                                    eds_bin_width =                         dat['evPerChannel'][0]
                                    eds_start_channel =                     dat['StartEnergy'][0]
                                    eds_averaged_frames =                   file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGEIPR']['NFrames'][0]
                                    eds_process_time =                      np.nan 
                                    eds_x_cells =                           file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS']['Width'][0]                          
                                    eds_y_cells =                           file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS']['Height'][0]
                                    eds_x_step =                            file[str(project)][str(sample)][str(area)][str(montage)]['FOVIMAGECOLLECTIONPARAMS']['MicronsPerPixelX'][0]
                                    eds_y_step =                            file[str(project)][str(sample)][str(area)][str(montage)]['FOVIMAGECOLLECTIONPARAMS']['MicronsPerPixelY'][0]
                                    eds_real_time_sum =                     dat['DeadTime'][0] + dat['LiveTime'][0]
                                    eds_live_time =                         dat['LiveTime'][0]  # sum across image 
                                    eds_unique_vendor_id =                  file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Id'][0] # unique id 
                                    eds_detector_azimuth_radians =          math.radians(dat['AzimuthAngle'][0])
                                    eds_detector_elevation_radians =        math.radians(dat['ElevationAngle'][0])
                                    eds_hardware_detector_serial_number =   np.nan
                                    eds_hardware_detector_model =           np.nan
                                    eds_drift_correction =                  np.nan
                                    eds_num_bins =                          dat['NumberOfPoints'][0]
                                    eds_processor =                         np.nan
                                    eds_date =                              np.nan
                                                                            #dat['CollectDateTime']
                                                                            #dat['DataStart']
                                    eds_magnification =                     file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Magnification'][0]
                                    eds_x_pos =                             file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageXPosition'][0]
                                    eds_y_pos =                             file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageYPosition'][0]
                                    eds_z_pos =                             file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageZPosition'][0]
                                    eds_tilt_radians  =                     math.radians(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Tilt'][0]) 
                                    eds_rotation_radians =                  math.radians(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Rotation'][0])
                                    eds_strobe_area =                       np.nan
                                    eds_strobe_FWHM =                       np.nan
                                    eds_window_type =                       np.nan
                                    eds_working_distance =                  file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['WD'][0]
                                   
                                    ebsd_project =                          project_name
                                    ebsd_specimen =                         specimen_name
                                    ebsd_site =                             site_name
                                    ebsd_voltage =                          file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['KV'][0]
                                    ebsd_magnification =                    file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Magnification'][0]
                                    ebsd_field =                            field_name
                                    ebsd_acq_date =                         np.nan
                                    ebsd_acq_speed =                        np.nan 
                                    ebsd_acq_time =                         np.nan
                                    ebsd_label =                            np.nan
                                    ebsd_unique_ID =                        file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Id'][0] # unique id 
                                    ebsd_background_correction =            np.nan
                                    ebsd_band_detection_mode =              np.nan
                                    ebsd_beam_voltage =                     np.nan
                                    ebsd_bounding_box =                     np.nan
                                    ebsd_bounding_box_x =                   np.nan
                                    ebsd_bounding_box_y =                   np.nan
                                    ebsd_camera_binning_mode =              np.nan
                                    ebsd_camera_exposure_time =             np.nan
                                    ebsd_camera_gain =                      np.nan
                                    ebsd_detector_insertion_distance =      np.nan
                                    ebsd_detector_orientation_euler =       np.nan
                                    ebsd_detector_orientation_euler_a =     np.nan
                                    ebsd_detector_orientation_euler_b =     np.nan
                                    ebsd_detector_orientation_euler_c =     np.nan 
                                    ebsd_drift_correction =                 np.nan
                                    ebsd_hit_rate =                         np.nan
                                    ebsd_hough_resolution =                 np.nan
                                    ebsd_indexing_mode =                    np.nan
                                    ebsd_lens_distortion =                  np.nan
                                    ebsd_lens_field_view =                  np.nan
                                    ebsd_magnification =                    np.nan
                                    ebsd_number_bands_detected =            np.nan
                                    ebsd_number_frames_averaged =           np.nan
                                    ebsd_pattern_height =                   np.nan
                                    ebsd_pattern_width =                    np.nan
                                    ebsd_project_file =                     np.nan
                                    ebsd_project_label =                    np.nan
                                    ebsd_project_notes =                    np.nan
                                    ebsd_relative_offset =                  np.nan
                                    ebsd_relative_offset_x =                np.nan
                                    ebsd_relative_offset_y =                np.nan
                                    ebsd_relative_size =                    np.nan
                                    ebsd_relative_size_x =                  np.nan
                                    ebsd_relative_size_y =                  np.nan
                                    ebsd_scanning_rotation_angle =          np.nan
                                    ebsd_site_label =                       np.nan
                                    ebsd_site_notes =                       np.nan
                                    ebsd_specimen_label =                   np.nan
                                    ebsd_specimen_notes =                   np.nan
                                    ebsd_specimen_orientation =             np.nan
                                    ebsd_specimen_orientation_a =           np.nan
                                    ebsd_specimen_orientation_b =           np.nan
                                    ebsd_specimen_orientation_c =           np.nan
                                    ebsd_stage_x =                          file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageXPosition'][0]
                                    ebsd_stage_y =                          file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageYPosition'][0]
                                    ebsd_stage_z =                          file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageZPosition'][0]
                                    ebsd_stage_rotation =                   math.radians(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Rotation'][0])
                                    ebsd_stage_tilt =                       math.radians(file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['Tilt'][0]) 
                                    ebsd_static_background_correction =     np.nan
                                    ebsd_tilt_angle =                       np.nan
                                    ebsd_tilt_axis =                        np.nan
                                    ebsd_working_distance =                 file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['WD'][0]
                                    ebsd_xcells =                           int(file[str(project)][str(sample)][str(area)]['FOVIMAGECOLLECTIONPARAMS']['Width'])
                                    ebsd_ycells =                           int(file[str(project)][str(sample)][str(area)]['FOVIMAGECOLLECTIONPARAMS']['Height'])
                                    ebsd_xstep =                            file[str(project)][str(sample)][str(area)][str(montage)]['FOVIMAGECOLLECTIONPARAMS']['MicronsPerPixelX'][0]
                                    ebsd_ystep =                            file[str(project)][str(sample)][str(area)][str(montage)]['FOVIMAGECOLLECTIONPARAMS']['MicronsPerPixelY'][0]
                                       
                                    metadata_list = [] 
        
                                    metadata_list.append( (
                                        montage_name,
                                        hardware_vendor,
                                        software_version, 
                                        format_version, 
                                
                                        electron_image_project,
                                        electron_image_specimen,
                                        electron_image_site,
                                        electron_image_field,
                                        electron_image_voltage,
                                        electron_image_magnification,
                                        electron_image_unique_vendor_id,
                                        electron_image_drift_correction,
                                        electron_image_dwell_time,
                                        electron_image_average_frames,
                                        electron_image_x_cells,
                                        electron_image_y_cells,
                                        electron_image_x_step,
                                        electron_image_y_step,
                                        electron_image_date,
                                        electron_image_x_pos,
                                        electron_image_y_pos,
                                        electron_image_z_pos,
                                        electron_image_tilt_radians,
                                        electron_image_rotation_radians,
                                        electron_image_working_distance,
                                        
                                        eds_project,
                                        eds_specimen,
                                        eds_site,
                                        eds_field,
                                        eds_voltage,
                                        eds_magnification,
                                        eds_binning,
                                        eds_bin_width,
                                        eds_start_channel,
                                        eds_averaged_frames,
                                        eds_process_time,
                                        eds_x_cells,
                                        eds_y_cells,
                                        eds_x_step,
                                        eds_y_step,
                                        eds_real_time_sum,
                                        eds_live_time,
                                        eds_unique_vendor_id,
                                        eds_detector_azimuth_radians,
                                        eds_detector_elevation_radians,
                                        eds_hardware_detector_serial_number,
                                        eds_hardware_detector_model,
                                        eds_drift_correction,
                                        eds_num_bins,
                                        eds_processor,
                                        eds_date,
                                        eds_x_pos,
                                        eds_y_pos,
                                        eds_z_pos,
                                        eds_tilt_radians,
                                        eds_rotation_radians,
                                        eds_strobe_area,
                                        eds_strobe_FWHM,
                                        eds_window_type,
                                        eds_working_distance,
                                        
                                        ebsd_project,
                                        ebsd_specimen,
                                        ebsd_site,
                                        ebsd_field,
                                        ebsd_voltage,
                                        ebsd_magnification,
                                        ebsd_acq_date,
                                        ebsd_acq_speed,
                                        ebsd_acq_time,
                                        ebsd_unique_ID,
                                        ebsd_background_correction,
                                        ebsd_band_detection_mode,
                                        ebsd_bounding_box_x,
                                        ebsd_bounding_box_y,
                                        ebsd_camera_binning_mode,
                                        ebsd_camera_exposure_time,
                                        ebsd_camera_gain,
                                        ebsd_detector_insertion_distance,
                                        ebsd_detector_orientation_euler_a,
                                        ebsd_detector_orientation_euler_b,
                                        ebsd_detector_orientation_euler_c,
                                        ebsd_drift_correction,
                                        ebsd_hit_rate,
                                        ebsd_hough_resolution,
                                        ebsd_indexing_mode,
                                        ebsd_lens_distortion,
                                        ebsd_lens_field_view,
                                        ebsd_magnification,
                                        ebsd_number_bands_detected,
                                        ebsd_number_frames_averaged,
                                        ebsd_pattern_height,
                                        ebsd_pattern_width,
                                        ebsd_project_file,
                                        ebsd_project_notes,
                                        ebsd_relative_offset_x,
                                        ebsd_relative_offset_y,      
                                        ebsd_relative_size_x,        
                                        ebsd_relative_size_y,
                                        ebsd_scanning_rotation_angle,
                                        ebsd_site_notes,
                                        ebsd_specimen_notes,
                                        ebsd_specimen_orientation_a,
                                        ebsd_specimen_orientation_b,
                                        ebsd_specimen_orientation_c,
                                        ebsd_stage_x,
                                        ebsd_stage_y ,
                                        ebsd_stage_z,
                                        ebsd_stage_rotation,
                                        ebsd_stage_tilt,
                                        ebsd_static_background_correction,
                                        ebsd_tilt_angle,
                                        ebsd_tilt_axis,
                                        ebsd_working_distance,
                                        ebsd_xcells,
                                        ebsd_ycells,
                                        ebsd_xstep,
                                        ebsd_ystep
                                        ) ) 
                            
                                    metadata_list = np.asarray(metadata_list, dtype=object)
                                    
                                    ##########    
                                    cols = ["Montage Label",
                                            "Hardware Vendor",
                                            "Software Version", 
                                            "Format Version", 
                                
                                            "SEM Project",
                                            "SEM Specimen",
                                            "SEM Site",
                                            "SEM Field",
                                            "SEM Voltage (KeV)",
                                            "SEM Magnification",            
                                            "SEM Vendor Unique ID",
                                            "SEM Drift Correction",
                                            "SEM Dwell Time (us)",
                                            "SEM Average Frames",
                                            "SEM Number of X Cells",
                                            "SEM Number of Y Cells",
                                            "SEM X Step Size (um)",
                                            "SEM Y Step Size (um)",
                                            "SEM Date",
                                            "SEM Stage X Position (mm)",
                                            "SEM Stage Y Position (mm)",
                                            "SEM Stage Z Position (mm)",
                                            "SEM Stage Tilt (rad)",
                                            "SEM Stage Rotation (rad)",
                                            "SEM Working Distance (mm)",
                                            
                                            "EDS Project",
                                            "EDS Specimen",
                                            "EDS Site",
                                            "EDS Field",
                                            "EDS Voltage (KeV)",
                                            "EDS Magnification",
                                            "EDS Binning Factor",
                                            "EDS Voltage Bin Width (eV)",
                                            "EDS Starting Bin Voltage (eV)",
                                            "EDS Number of Averaged Frames",
                                            "EDS Process Time",
                                            "EDS Number of X Cells",
                                            "EDS Number of Y Cells",
                                            "EDS X Step Size (um)",
                                            "EDS Y Step Size (um)",
                                            "EDS Real Time Sum (s)",
                                            "EDS Live Time (s)",
                                            "EDS Vendor Unique ID",
                                            "EDS Azimuth Angle (rad)",
                                            "EDS Detector Angle (rad)",
                                            "EDS Detector Serial Number",
                                            "EDS Detector Model Number",
                                            "EDS Drift Correction",
                                            "EDS Number of Channels",
                                            "EDS Processor Type",
                                            "EDS Date",
                                            "EDS Stage X Position (mm)",
                                            "EDS Stage Y Position (mm)",
                                            "EDS Stage Z Position (mm)",
                                            "EDS Stage Tilt (rad)",
                                            "EDS Stage Rotation (rad)",
                                            "EDS Strobe Area",
                                            "EDS Strobe FWHM (ev)",
                                            "EDS Window Type",
                                            "EDS Working Distance (mm)",
                                            
                                            "EBSD Project",
                                            "EBSD Specimen",
                                            "EBSD Site",
                                            "EBSD Field",
                                            "EBSD Voltage (KeV)",
                                            "EBSD Magnification",
                                            "EBSD Acquisition Date",
                                            "EBSD Acquisition Speed (Hz)",
                                            "EBSD Acquisition Time (s)",
                                            "EBSD Vendor Unique ID",
                                            "EBSD Auto Background Correction",
                                            "EBSD Band Detection Mode",
                                            "EBSD Bounding Box X (um)",
                                            "EBSD Bounding Box Y (um)",
                                            "EBSD Camera Binning Mode",
                                            "EBSD Camera Exposure Time (ms)",
                                            "EBSD Camera Gain",
                                            "EBSD Detector Insertion Distance (mm)",
                                            "EBSD Detector Orientation Euler A (rad)",
                                            "EBSD Detector Orientation Euler B (rad)",
                                            "EBSD Detector Orientation Euler C (rad)",
                                            "EBSD Drift Correction",
                                            "EBSD Hit Rate",
                                            "EBSD Hough Resolution",
                                            "EBSD Indexing Mode",
                                            "EBSD Lens Distortion",
                                            "EBSD Lens Field View (mm)",
                                            "EBSD Magnification",
                                            "EBSD Number Bands Detected",
                                            "EBSD Number Frames Averaged",
                                            "EBSD Pattern Height (px)",
                                            "EBSD Pattern Width (px)",
                                            "EBSD Project File",
                                            "EBSD Project Notes",
                                            "EBSD Relative Offset X",
                                            "EBSD Relative Offset Y",
                                            "EBSD Relative Size X",
                                            "EBSD Relative Size Y",
                                            "EBSD Scanning Rotation Angle (rad)",
                                            "EBSD Site Notes",
                                            "EBSD Specimen Notes",
                                            "EBSD Specimen Orientation A",
                                            "EBSD Specimen Orientation B",
                                            "EBSD Specimen Orientation C",
                                            "EBSD Stage X Position (mm)",
                                            "EBSD Stage Y Position (mm)" ,
                                            "EBSD Stage Z Position (mm)",
                                            "EBSD Stage Rotation (rad)",
                                            "EBSD Stage Tilt (rad)",
                                            "EBSD Static Background Correction",
                                            "EBSD Tilt Angle (rad)",
                                            "EBSD Tilt Axis",
                                            "EBSD Working Distance (mm)",
                                            "EBSD Number X Pixels",
                                            "EBSD Number Y Pixels",
                                            "EBSD X Step Size (um)",
                                            "EBSD Y Step Size (um)"]
                                
                                    metadata_list = pd.DataFrame( data = metadata_list, columns = cols)
                                    
                                    try:
                                        metadata = pd.concat([metadata, metadata_list], sort=False, axis = 0, ignore_index = True)
                                    except NameError: 
                                        metadata = pd.DataFrame( data = metadata_list, columns = cols)
                                 
                            x_scale = np.unique( metadata['EDS X Step Size (um)'] )[0]
                            y_scale = np.unique( metadata['EDS Y Step Size (um)'] )[0]
                            
                            # get the XY coordinates for each field 
                            x_list =  list( np.unique( metadata['EDS Stage X Position (mm)'][metadata['Montage Label'] == montage] ) )
                            y_list =  list( np.unique( metadata['EDS Stage Y Position (mm)'][metadata['Montage Label'] == montage] ) ) 
                           
                            x_size = np.unique(metadata['EDS Number of X Cells'])
                            y_size = np.unique(metadata['EDS Number of Y Cells'])
                            
                            # find minimum coodinates so that we can create an appropriately sized array 
                            x_min = int( math.ceil(min(x_list)*1000.0/x_scale))
                            y_min = int( math.ceil(min(y_list)*1000.0/y_scale))
                            
                            # determine how large of an array is neeed 
                            x_range = int (math.ceil( ( 1000.0*(max(x_list) - min(x_list))/x_scale ) ) + x_size )
                            y_range = int( math.ceil( ( 1000.0*(max(y_list) - min(y_list))/y_scale ) ) + y_size )
                          
                            eds_layers = np.unique( metadata["EDS Number of Channels"] )
                            
                            output_path = os.path.join(output_src, "Montage " + str(sample) + "_" + str(area) + "_" + str(montage.replace("Montage", "", 1)) + ".h5" ) 
                            
                            try: 
                                save_file = h5py.File(output_path, 'a')
                            except: 
                                save_file = h5py.File(output_path, 'r+') 
            
                            #save_file.create_dataset('array', shape = (y_range, x_range, eds_layers), chunks=True, dtype = 'int32')
                            
                            for field in tqdm(Y): 
                                                                    
                                if "Field" in field: 
                                     
                                    #file[str(project)][str(sample)][str(area)][str(x)][str(field)]['MAPIMAGECOLLECTIONPARAMS'][...]
                                    height = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS']['Height'][0]
                                    width = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['MAPIMAGECOLLECTIONPARAMS']['Width'][0]
                                    # = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['SPC']['NumberOfPoints'][0]
                                    
                                    KV = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['KV'][0]
                                    
                                    
                                    evPch = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['SPC']['evPerChannel'][0]
                                    num_bins = int( (KV*1_000) / evPch )
    
                                    eds_x_pos = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageXPosition'][0]
                                    eds_y_pos = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['HOSTPARAMS']['StageYPosition'][0] 
        
                                    x_location = (eds_x_pos*1_000)/x_scale - x_min
                                    y_location = (eds_y_pos*1_000)/y_scale - y_min
                
                                    if x_location < 0:
                                        x_location = 0
                                    else:
                                        x_location = int(x_location)
                            
                                    if y_location < 0:
                                        y_location = 0
                                    else:
                                        y_location = int(y_location)
                                        
                                    #print(x_location)
                                    #print(y_location)
                                    ########
                                    block = file[str(project)][str(sample)][str(area)][str(montage)][str(field)]['SPD'][...].reshape(height, width, num_bins)
                                    block = np.flip(block, axis = 1)
                                    block = np.flip(block, axis = 0)
                                    
                                    try:
                                        save_file.create_group('EDS')
                                    except:
                                        pass 
                                
                                    try: 
                                        save_file['EDS'].create_dataset('Xray Spectrum', shape = (y_range, x_range, block.shape[2] ), chunks= True, dtype = 'int16')
                                    except: 
                                        pass 
                              
                                    try: 
                                        save_file['EDS'].create_dataset('Xray Intensity', shape = (y_range, x_range ), chunks=True, dtype = 'int64')
                                    except: 
                                        pass 
                                    
                                    save_file['EDS']['Xray Spectrum'][y_location:y_location + int(y_size), x_location:x_location + int(x_size), :] = block 
                                    save_file['EDS']['Xray Intensity'][y_location:y_location + int(y_size), x_location:x_location + int(x_size)] = np.sum( block, axis = 2)
                           
                                    new_highest_spectrum =  np.max( block[:, :, :], axis = (0,1)).flatten() 
                                    new_sum_spectrum =  np.sum( block[:, :, :], axis = (0,1)).flatten() 
                                    
                                    # Sum new block with existing data
                                    try: 
                                        sum_spectrum += new_sum_spectrum
                                    except NameError:
                                        sum_spectrum = np.zeros( shape = new_sum_spectrum.shape, dtype = np.float64 )
                                        sum_spectrum += new_sum_spectrum
                                    
                                    # Keep only highest peaks from every block of data 
                                    try: 
                                        highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                                    except NameError:
                                        highest_spectrum = np.zeros( shape = new_highest_spectrum.shape , dtype = np.float64 )
                                        highest_spectrum = np.maximum( np.asarray(highest_spectrum), np.asarray(new_highest_spectrum))
                                        
                                    gc.collect()
                
                            appended_peaks = []
                            search_width = 6
                            peaks_mean, _ = scipy.signal.find_peaks(sum_spectrum, 
                                                               prominence = (0.10*statistics.median( sum_spectrum ), None),
                                                               distance = 5
                                                               )
                            
                            peaks_max, _ = scipy.signal.find_peaks(highest_spectrum, 
                                                               height = (np.percentile(highest_spectrum, 95), None ),
                                                               prominence = 2,
                                                               distance = 5
                                                               )
                            
                            peaks_mean = list( peaks_mean )
                            peaks_max = list( peaks_max)  
                            
                            # Sort candidate peaks by the x ray intensity. We want to add the strongest peaks first
                            a = np.argsort(sum_spectrum[peaks_mean]) 
                            peaks_mean[:] = [peaks_mean[i] for i in a][:len(peaks_mean)]             
                            
                            a = np.argsort(highest_spectrum[peaks_max]) 
                            peaks_max[:] = [peaks_max[i] for i in a][:len(peaks_max)]      
                            
                            # Iterate through all candidate peaks that were found for the montage and include the peak if it is more than X bins from previously detected peaks 
                            i = 0 
                            while True:          
                                
                                try: 
                                    peak_candidate = peaks_mean[i]
                                
                                    if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                                        appended_peaks.append(peak_candidate)
                                        peaks_mean.remove(peak_candidate)
                                        i = 0
                                    else:
                                        i += 1 
                                            
                                except: 
                                    break 
                            
                            i = 0 
                            while True:          
                                
                                try: 
                                    peak_candidate = peaks_max[i]
                                
                                    if np.all( abs(appended_peaks - peak_candidate) > search_width ): 
                                        appended_peaks.append(peak_candidate)
                                        peaks_max.remove(peak_candidate)
                                        i = 0
                                    else:
                                        i += 1 
                                            
                                except: 
                                    break 
                                          
                            # Save autodetected peaks, sum of spectrum, and highest intensity spectrum for each montage 
                            save_h5(output_path, 'EDS', ['array', 'Autodetected Peak Bins'], np.array(appended_peaks ).astype(np.uint16) )
                            save_h5(output_path, 'EDS', ['array', 'Sum of Spectrum'], np.array(sum_spectrum ).astype(np.uint64) )
                            save_h5(output_path, 'EDS', ['array', 'Highest Intensity Spectrum'], np.array(highest_spectrum ).astype(np.uint64) )
                                    
                            for header in metadata.columns: 
                                try:
                                    save_file.create_dataset( 'Metadata/'+str(header) , data = metadata[str(header)][metadata['Montage Label'] == montage].astype('S'))
                                except ValueError:
                                    pass
                        
                            save_file.close() 
                            del metadata
                            
    file.close()
                            
    


def main(): 
    print("")
    print("This software parses proprietry electron microscope files into a common format for VBGMM analysis")
    print("Select a supported input format from the following options: ")
    print("")
    print("1 for Oxford SEM-EDS & SEM-EBSD")
    print("   Multiple montages from the same Oxford OIP file may be parsed in the same batch")
    print("   Montages from different OIP files must be parsed as seperate batches")
    print("   The Oxford parser inputs H5OINA, RPL, and RAW files")
    print("")
    print("2 for Bruker SEM-EDS")
    print("   The parser inputs BCF files from a single Bruker montage")
    print("   Multiple different montages must each be parsed seperately")
    print("")
    print("3 for Bruker STEM-EDS")
    print("   Each individual tile image is parsed seperately")
    print("")
    print("4 for EDAX SEM-EDS")
    print("")
    print("Press 'enter' to execute")
    # get analysis type input 
    file_type =  int( input("") )

    if file_type == 1: 
        oxford_parse()
    elif file_type == 2: 
        bruker_parse_SEM()
    elif file_type == 3: 
        bruker_parse_STEM()
    elif file_type == 4:
        EDAX_parse()
    else: 
        raise NameError("Error: Enter a number for a supported format type") 



if __name__ == "__main__":
    
    main() 
    













































