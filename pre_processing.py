import os
import numpy as np
import pandas as pd
from tqdm import tqdm 
import cv2
import matplotlib.pyplot as plt

data_idx = pd.read_csv('training_samples_X4_1Y.csv')
data_idx.drop(columns='Unnamed: 0', inplace=True)
data_idx.head()

data_idx.describe()

def build_target(target, path, file_list,i):
    # Get target id
    target_folder = int(target/288)
    target_id = target % 288    
    # Open folder
    file_name = file_list[target_folder]
    file_name = str(file_name).split("'")[1]
    maps = np.load(path+str(file_name))['arr_0']
    target_map = np.array(maps[target_id])  
    # Normalization
    max_value = 1
    target_map = target_map/max_value
    # Resize target map
    target_map = cv2.resize(target_map.astype('float64'), (64, 64), interpolation = cv2.INTER_CUBIC)
    # Save target map
    np.save('/home/fernandezn/Bureau/SoftConvDeconv/RadarNet/Data/4x_1y_corrected/target_{}'.format(i),np.array(target_map))
    return  np.array(target_map)

def get_sequence_idx(sequence):
    file_id = [int(x / 288) for x in sequence]
    seq_id = [x % 288 for x in sequence]
    file_id = np.unique(file_id)
    return file_id, np.array(seq_id)

def get_sequence_maps(file_id, seq_id, path, file_list):  
    map_seq = []
    # Load always the first map of the sequence
    file_name = file_list[file_id[0]]
    file_name = str(file_name).split("'")[1]
    maps = np.load(path+str(file_name))['arr_0']
    map_seq.append(maps[seq_id[0]])   
    # Check continuity
    for i in 0,1,2:
        if (seq_id[i] + 1 == seq_id[i+1]):
            # Use the first file opened
            map_seq.append(maps[seq_id[i+1]])
        else:
            print('Warning: One map is in a following file')
            # Open next file
            file_name = file_list[file_id[1]]
            file_name = str(file_name).split("'")[1]
            # Open maps and keep only the needed
            maps = np.load(path+str(file_name))['arr_0']
            map_seq.append(maps[seq_id[i+1]])

    return np.array(map_seq)

def build_tensor(x,path,files,i):
    """Function to compute resized and normalized input tensor.
    The result is saves in harcoded folder.
    """
    # Get id's
    file_id, seq_id = get_sequence_idx(x)
    # Get sequence
    seq = get_sequence_maps(file_id, seq_id, path, files)
    # Normalization
    max_value = 1
    seq = seq/max_value
    # Build tensor
    tensor = [cv2.resize(x.astype('float64'), (64, 64), interpolation = cv2.INTER_CUBIC) for x in seq] 
    # Save tensor
    np.save('/home/fernandezn/Bureau/SoftConvDeconv/RadarNet/Data/4x_1y_corrected/tensor_{}'.format(i),np.array(tensor))
    return tensor

# Get filenames
path = '/home/fernandezn/Bureau/SoftConvDeconv/Rain rates/'

# Sort 
npz_files = [str.lower(e) for e in os.listdir(path)]
files = pd.DataFrame(npz_files).sort_values(by=[0])
files = files.values

# Processing dataset
prev_index = -1 
row = 0
for index in tqdm(data_idx['Y'].values):
    if prev_index + 1 == index:
        maps = build_target(index,path,files,index)
    else:
        for index in data_idx.iloc[row].values:
            maps = build_target(index,path,files,index)
    prev_index = index
    row = row + 1



