import matplotlib.pyplot as plt
import visdom
import numpy as np
import os
import sys
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn as nn

sys.path.append('Architectures/')
sys.path.append('Ops/Utils')
import soft_conv_deconv as scd_estimator

sys.path.append('Ops/')
import warp as warp
import dataset as dataset
import losses as losses
import norms as norms

viz = visdom.Visdom(server='http://132.227.204.165', env='viz_seqs')

dataset_4x_1y_path = 'Data/4x_1y_dataset_non-norm/'
models_path = 'Models/'

parser = argparse.ArgumentParser(description='Visualializing some pretrained model predictions')
parser.add_argument('--architecture', default='scd')#, choices=estimator_names, help='Estimator architecture: ' ' | '.join(estimator_names))
parser.add_argument('--norm', default='None')#, choices=norm_type, help='Normalization type: ' ' | '.join(norm_type))
parser.add_argument('--viz', default=1, type=int)#, choices=norm_type, help='Normalization type: ' ' | '.join(norm_type))
parser.add_argument('-s', '--start', default=0, type=int)
parser.add_argument('-c', '--count', default=4, type=int)
parser.add_argument('-seq', '--seq', default=0, type=int)

args = parser.parse_args()

if(args.architecture == 'scd'):
    model = scd_estimator.SoftConvDeconvEstimator()
    print('Using a SoftConvDeconv architecture')
else:
    print('Bad architecture name')

if torch.cuda.is_available():
    print('Using CUDA')
    model = model.cuda()
else:
    print('Running on CPU :')


#if(args.viz):
    # To load weights
    #model.load_state_dict(torch.load(models_path+'SoftConvDeconv'))
    #OP-Adam_BS-64_N-reduce_LR-0.0001
    #OP-Adam_BS-128_N-reduce_LR-0.0001
#    model.load_state_dict(torch.load(models_path+'OP-Adam_BS-128_N-reduce_LR-0.0001'))
#    print('Using a SoftConvDeconv architecture... Model loaded')	

x_train, y_train = dataset.load_data(dataset_4x_1y_path)

# Normalize x and y
max_val = np.max(x_train)
mean_val = np.mean(x_train)
std_val = np.std(x_train)

if(args.norm == 'reduce'):
    x = norms.norm_reduce(x_train, mean_val, std_val)
    y = norms.norm_reduce(y_train, mean_val, std_val)

if(args.norm == 'log'):
    x = norms.norm_log(x_train)
    y = norms.norm_log(y_train) 
else:
    x = x_train
    y = y_train

# Init warp schema
warp = warp.BilinearWarpingScheme()
      
def get_predicted_image(model,X,warp):
    ''' Functix_trainon to compute and plot a real map.
    data: input X numpy tensor
    warp: initialized warp class objectnorm
    Returns:
    img: numpy array image
    '''
    if torch.cuda.is_available:
        output = model(Variable(torch.from_numpy(X)).cuda().unsqueeze(0))
        X = Variable(torch.from_numpy(X).cuda(async=True).unsqueeze(0))
        img = warp(X[:, -1].unsqueeze(1), output)
        predicted = img.data.cpu().numpy().reshape(64,64)
    else:
        print('Error: CUDA is not avaliable')
        return 0
    return np.expand_dims(predicted, axis=0), output

import flowlib as flow
import plot as flow_plot
import time
from scipy.misc import imresize

# color vector field
def flow_to_im(w):
    img = np.flip(flow_plot.flow_color(w), 0)
    img = imresize(img, (64 * 4, 64 * 4)).transpose(2, 0, 1)
    return img

results_path = 'Results/'
for mdl in [e for e in os.listdir(models_path)]:
    #win_real = viz.heatmap(y_train[0][0])
    #win_pred = viz.heatmap(y_train[0][0])
    
    
    model.load_state_dict(torch.load(models_path+mdl))
    
    
    if args.seq == 1:
        # Generate a sequence of plots
        start_sequence = args.start
        sequence_count = args.count
        
        for i in range(0, sequence_count):
            viz.heatmap(y_train[start_sequence + i][0], opts=dict(title='Real {} | {}'.format(i + start_sequence, mdl)))
            np.savez_compressed(results_path+'{}_{}'.format('Real', i + start_sequence), y_train[start_sequence + i][0])
            
        X, Y, W = x[start_sequence], [[x[start_sequence][-1]]], []
        
        for i in range(0, sequence_count):
            predicted, w = get_predicted_image(model,X,warp)
            Y.append(predicted) # Save the output in for the end sequence     
            W.append(w)
            X = np.concatenate((X, np.array(predicted))) # Push the output to the top of the next input
            X = np.delete(X, 0, 0) # Remove the first input map
        
        errors = []
        
        for i in range(0, sequence_count): # 1
            #viz.heatmap(y_train[start_sequence + i][0], win=win_real, opts=dict(title='Real {} | {}'.format(i + start_sequence, mdl)))
            #viz.heatmap(Y[i][0], win=win_pred, opts=dict(title='Pred {} | {}'.format(i + start_sequence, mdl)))
            viz.heatmap(Y[i][0], opts=dict(title='Pred {} | {}'.format(i + start_sequence, mdl)))
            w = W[i].data.cpu().numpy()
            #print('w : ', np.shape(w))
            flow_clr = flow_to_im(w[0])
            viz.image(flow_clr, 
                      opts=dict(title='Warp {} | {}'.format(i + start_sequence, mdl)))
            
            #viz.image(W[i],# win=str(2 * len(imgs) + 1),
            #                  opts=dict(title='Warp {}'.format(i + start_sequence)))
            errors.append(losses.rmse(Y[i][0],y_train[i][0]))
            #time.sleep(1)
            
            np.savez_compressed(results_path+'{}_{}'.format(str.split(mdl,'-')[2][:-2], i + start_sequence), Y[i][0])
        
        # Validation on a sequence of 
        print('mean sequence error = ', np.mean(np.array(errors))) 
    
    else:
        # Generate a sequence of plotsS
        start_sequence = args.start
        sequence_count = args.count
        
        X, Y, W = x[start_sequence: sequence_count + start_sequence], [], []
        
        for i in range(0, sequence_count):
            predicted, w = get_predicted_image(model,X[i],warp)
            Y.append(predicted) # Save the output in for the end sequence     
            W.append(w)
        
        errors = []
        
        import time
        for i in range(1, sequence_count):
            viz.image(y_train[start_sequence + i][0], win=win_real, opts=dict(title='Real {} | {}'.format(i + start_sequence, mdl)))
            
            viz.image(Y[i][0], win=win_pred, opts=dict(title='Pred {} | {}'.format(i + start_sequence, mdl)))
            #viz.image(W[i],# win=str(2 * len(imgs) + 1),
            #                  opts=dict(title='Warp {}'.format(i + start_sequence)))
            errors.append(losses.rmse(Y[i][0],y_train[i][0]))
            
            time.sleep(2)
        
        # Validation on a sequence of 
        print('mean sequence error = ', np.mean(np.array(errors))) 