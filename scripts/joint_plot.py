# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:24:37 2023

@author: alexa
"""
import os
import matplotlib.pyplot as plt
from src.utils.plotting import plot_err_evol, process

from src.utils.files import data_from_file
from src.utils.mydataclasses import get_label

PROCESSING = {'classical': 'none',
                'canonical': 'none',
                'MLQAE':  'averaging', 
                'QAES': 'binning',
                'SQAE #2': "binning", 
                'SQAE #1': "binning", 
                'SQAE #0': "binning", 
                'FQAE': 'binning',
                'IQAE - chernoff': 'binning',
                'mIQAE - chernoff': 'binning',
                'BAE': 'binning'}

def plot_from_folder(folder):
    '''
    Plot together all the datasets in the folder 'folder' of the folder 
    'datasets'.
    '''
    fnlist = dataset_filenames_from_folder(folder)
    stats = ["mean", "median"]
    for stat in stats:
        estdatas = get_estdatas(fnlist, stat)
        plot_err_evol("RMSE", estdatas, stat)
        plt.show()

def dataset_filenames_from_folder(folder_name):
    '''
    Returns a list of the filenames for all files in the subfolder 'folder_name'
    of the folder 'datasets'.
    '''
    folder_path = os.path.join(os.getcwd(), "datasets", folder_name)
    filenames = [file for file in os.listdir(folder_path)]
    full_paths = [os.path.join(folder_path, filename) for filename in filenames]
    return full_paths

def get_estdatas(filename_list, stat):
    '''
    Get estimation data objects from the execution data objects in the files
    and process them.
    '''
    execdatas = [data_from_file(filename) for filename in filename_list]
    labels = [get_label(execdata) for execdata in execdatas]

    estdatas = [execdata.estdata for execdata in execdatas]
    estdatas = [process(estdata, stat, how = PROCESSING[label])
                for label, estdata in zip (labels, estdatas)]
    return estdatas

if __name__ == "__main__":
    plot_from_folder("noiseless")