#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:56:20 2018

@author: benmeradn
"""

import numpy as np
import os

def load_data(dataset_4x_1y_path):
    # Preparing the training dataset
    dataset_4x_1y = [str.lower(e) for e in os.listdir(dataset_4x_1y_path)]
    
    x_train = np.zeros((908,4,64,64), dtype=np.float32) # (908,1)
    y_train = np.zeros((908,1,64,64), dtype=np.float32) # (908,1)
    
    for file in dataset_4x_1y:
        file_path = dataset_4x_1y_path+str(file)
        file_info = str.split(file[:-4], '_')
        
        data = np.load(file_path)
        data = data.astype('float32')
        if(file_info[0] == 'tensor'):
            x_train[int(file_info[1])] = data
        elif(file_info[0] == 'target'):
            y_train[int(file_info[1])] = data
        else:
            print('Warning : Bad file naming \'{}\''.format(file))
    print('Dataset loaded')
    return x_train, y_train
   
from numpy import random
def build_dataset_4X_1Y(dataframe, path):
    '''Function to read only 4x_1Y database.
    Dataframe: Pandas dataframe of indexes
    Path: Path to database files
    Returns:
    x, y: numpy arrays
    '''
    x = []
    y = []
    
    list_items = range(dataframe.count()[0])[0:5000]
    list_items_rndm = random.choice(list_items,1000)
    for row_idx in list_items:
        # Building tensor from indexes dataframe
        tensor = []
        for map_idx in dataframe.drop('Y', axis=1).iloc[row_idx]:
            # Map index to filename
            file_name = path + 'target_' + str(map_idx) + '.npy'
            # Read file
            data = np.load(file_name).astype('float32')
            tensor.append(data)
        x.append(np.array(tensor))
        # Building target
        target_idx = dataframe['Y'].iloc[row_idx]
        target_name = path + 'target_' + str(target_idx) + '.npy'
        target = np.load(target_name).astype('float32')
        target = np.expand_dims(target, axis=0)
        y.append(target)
    return np.array(x), np.array(y)
    
def build_dataset_4X_6Y(dataframe, path):
    '''Function to reads only 4x_1Y data.
    Dataframe: Pandas dataframe of indexes
    Path: Path to database files
    Returns:
    x, y: numpy arrays
    '''
    x = []
    y = []
    for row_idx in tqdm(range(dataframe.count()[0])):
        # Building tensor from indexes dataframe
        tensor = []
        for map_idx in dataframe.drop(columns='Y1 Y2 Y3 Y4 Y5 Y6'.split()).iloc[row_idx].values:
            # Map index to filename
            file_name = path + 'target_' + str(map_idx) + '.npy'
            # Read file 
            data = np.load(file_name).astype('float32')
            tensor.append(data)
        x.append(np.array(tensor))
        # Building target
        targets = []
        i =0
        for target_idx in dataframe['Y1 Y2 Y3 Y4 Y5 Y6'.split()].iloc[row_idx].values:
            target_name = path + 'target_' + str(target_idx) + '.npy'
            target = np.load(target_name).astype('float32')
            targets.append(target)
        y.append(np.array(targets))
    return np.array(x), np.array(y)
