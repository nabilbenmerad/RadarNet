import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import sys
#from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn as nn

sys.path.append('Architectures/')
sys.path.append('Ops/')
sys.path.append('Ops/Utils')

import soft_conv_deconv as scd_estimator
import warp as warp
import dataset as dataset
import losses as mylosses
import norms as norms
import visdom
import flowlib as flow
import plot as flow_plot
from scipy.misc import imresize

viz = visdom.Visdom(server='http://132.227.204.165', env='last_lap')

#dataset_4x_1y_path = 'Data/4x_1y_dataset_corrected/'
dataset_4x_1y_path = '/home/benmeradn/Bureau/Workspace/RadarNet/Data/4x_1y_corrected/'

models_path = '/home/benmeradn/Bureau/Workspace/RadarNet/Models/'

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--architecture', default='scd')#, choices=estimator_names, help='Estimator architecture: ' ' | '.join(estimator_names))
parser.add_argument('--norm', default='None')#, choices=norm_type, help='Normalization type: ' ' | '.join(norm_type))
parser.add_argument('-e', '--epochs', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-sl', '--sequence_lenght', default=6, type=int)
parser.add_argument('-tt', '--train_type', default='normal')
parser.add_argument('-op', '--optimizer', default='Adam')

args = parser.parse_args()

if(args.architecture == 'scd'):
    model = scd_estimator.SoftConvDeconvEstimator()
    print('Using a SoftConvDeconv architecture')
else:
    print('Bad architecture name')

if torch.cuda.is_available():
    print('Using CUDA !')
    model = model.cuda()
else:
    print('Running on CPU :')
	
# Preparing the training dataset
#x_denorm, y_denorm = dataset.load_data(dataset_4x_1y_path)   

data_idx = pd.read_csv('training_samples_X4_1Y.csv')
data_idx.drop('Unnamed: 0', inplace=True, axis=1)
print('Loading dataset...')
x_denorm, y_denorm = dataset.build_dataset_4X_1Y(data_idx, dataset_4x_1y_path)

print('Dataset Loaded...')
print('X shape = ', np.shape(x_denorm))
print('Y shape = ', np.shape(y_denorm))

# Normalize x and y
max_val = np.max(x_denorm)
mean_val = np.mean(x_denorm)
std_val = np.std(x_denorm)

if(args.norm == 'reduce'):
    x = norms.norm_reduce(x_denorm, mean_val, std_val)
    y = norms.norm_reduce(y_denorm, mean_val, std_val)

if(args.norm == 'log'):
    x = norms.norm_log(x_denorm)
    y = norms.norm_log(y_denorm) 
else:
    x = x_denorm
    y = y_denorm

# Training, test and validation split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)
x_train, x_val, y_train, y_val  = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

# Casting to Torch tensors
x_train_tensors = torch.from_numpy(x_train)
y_train_tensors = torch.from_numpy(y_train)

x_test_tensors = torch.from_numpy(x_test)
y_test_tensors = torch.from_numpy(y_test)

x_val_tensors = torch.from_numpy(x_val)
y_val_tensors = torch.from_numpy(y_val)

print('Training, test and validation partitions created...')

batch_size=args.batch_size
epochs = args.epochs

test = data_utils.TensorDataset(x_test_tensors, y_test_tensors)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

val = data_utils.TensorDataset(x_val_tensors, y_val_tensors)
val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=True)

warp = warp.BilinearWarpingScheme()

lr = 0.0001
momentum = 0.9
beta= 0.999
weight_decay = 4e-4

if(args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(momentum, beta), weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum , weight_decay=weight_decay)
    

criterion = nn.MSELoss()

train = data_utils.TensorDataset(x_train_tensors, y_train_tensors)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

def get_predicted_image(model,X,warp):
    ''' Function to compute and plot a real map.
    data: input X numpy tensor
    warp: initialized warp class object
    Returns:
    img: numpy array image
    '''
    if torch.cuda.is_available:
        output = model(Variable(torch.from_numpy(X)).cuda().unsqueeze(0))
        X = Variable(torch.from_numpy(X).cuda().unsqueeze(0))
        img = warp(X[:, -1].unsqueeze(1), output)
        predicted = img.data.cpu().numpy().reshape(64,64)
    else:
        print('Error: CUDA is not avaliable')
    return np.expand_dims(predicted, axis=0)

# color vector field
def flow_to_im(w):
    img = np.flip(flow_plot.flow_color(w), 0)
    img = imresize(img, (64 * 4, 64 * 4)).transpose(2, 0, 1)
    return img


def adjust_learning_rate(optimizer, epoch, lr):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = lr * (0.1 *(epoch/50))
    for param_group in optimizer.param_groups:
    
        param_group['lr'] = lr
        return lr


# Start training
losses = []
total_val_loss = []
warp_maps = []
radar_maps = []
def train(epoch):
    #losses_by_batch=[]
    cum_loss = 0
    model.train()
    # losses_by_batch = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to cuda tensors if cuda flag is true
        if torch.cuda.is_available:
            data, target = data.cuda(), target.cuda()
        
        data, targets = Variable(data), Variable(target)
        optimizer.zero_grad()
        # Autoregression loop
        if(args.train_type == 'autoreg'):
            sequence_loss = 0    
            for step in range(args.sequence_lenght):
                w = model(data)
                imgs = warp(data[:, -1].unsqueeze(1), w)
                data = torch.cat([data[:, 1:], imgs], 1)
                current_loss = criterion(imgs, targets)
                sequence_loss += current_loss
            loss = sequence_loss / args.sequence_lenght
        else:
            #print('data', np.shape(data))
            # Input tensor to get the warp
            w = model(data)
            #print('w', np.shape(w))
            # Get transformed image using the ward schema
            imgs = warp(data[:, -1].unsqueeze(1), w)
            #print('data[:, -1]', np.shape(data[:, -1]))
            #print('data[:, -1].unsqueeze(1)', np.shape(data[:, -1].unsqueeze(1)))
            #print('imgs', np.shape(imgs))
            # Compute MSE loss
            loss = criterion(imgs, targets)
	    cum_loss += loss.data[0]
        
        #print('Batch : {}  |  loss : {}   |   cum_loss : {}'.format(batch_idx, loss.data[0], cum_loss))
	    
        # Set gradients to zero and backpropagate

        loss.backward()
        optimizer.step()
        # Printing 
        if batch_idx % 50 == 0:
            #losses_by_batch.append(loss.data[0])
            # Plot random warping map on wisdom	
            r = np.random.randint(0, len(data))
            w = w[r].data.cpu().numpy()
            #flow_clr = flow_to_im(w)
            #viz.image(flow_clr, win=str(2 * len(imgs) + 1),
            #          opts=dict(title='output w {}'.format(1)))

            # Plot real radar image and its prediction on visdom
            #img = imgs[r, 0].data.cpu()
            #viz.heatmap(img, win=str(
            #             + 1), opts=dict(title='output image {}'.format(1)))

            #img_target = targets[r, 0].data.cpu()
            #viz.heatmap(img_target, win=str(
            #            len(imgs) + 1), opts=dict(title='target image {}'.format(1)))

    #print('Average train epoch rmse =',cum_loss)
    losses.append(cum_loss/((batch_idx+1)))
    # Saving maps and images (one per epoch)    
    warp_maps.append(w)
    radar_maps.append(imgs)
	
    # Compute validation loss
    epoch_vloss = 0
    for batch_idx, (data, target) in enumerate(val_loader):
  	#convert to cuda tensors if cuda flag is true
        if torch.cuda.is_available:
            data, target = data.cuda(), target.cuda()  
        data, target = Variable(data), Variable(target)
        	# Compute validation error of each epoch
           	    # Input tensor to get the warp
        w = model(data)
        	# Get transformed image using the ward schema
        img = warp(data[:, -1].unsqueeze(1), w)
        	# Compute MSE loss
        val_loss = criterion(img, target)
        epoch_vloss += val_loss.data[0]
        
    total_val_loss.append(epoch_vloss/((batch_idx+1)))
   
    #print('Val Batch : {}  |  loss : {}   |   cum_loss : {}'.format(batch_idx, val_loss.data[0], epoch_vloss))

if(args.train_type == 'autoreg'):
    model.load_state_dict(torch.load(models_path+'SoftConvDeconv'))
    print('Using a SoftConvDeconv architecture... Model loaded')	

win_loss_val = viz.line(
        X=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
        Y=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
        opts=dict(title='Loss_OP-{}_BS-{}_LR-{}'.format(args.optimizer, args.batch_size, 50.0001))
)
# Main training loop
for epoch in range(0, epochs):
    train(epoch)
    # Learning rate scheduler
    adjust_learning_rate(optimizer, epoch, lr)
    # loss line updates
    viz.line(
        X=np.column_stack((np.arange(0, len(losses)), np.arange(0, len(total_val_loss)))),
        Y=np.column_stack((np.array(losses), np.array(total_val_loss))),
        win=win_loss_val,
        update='replace',# could be 'append'
    )

# Compute model test error
test_error=[]
for i in range(len(y_test)):	
    predicted = get_predicted_image(model, x_test[i],warp)
    if(args.norm == 'reduce'):
        	predicted = norms.denorm_reduce(predicted,mean_val, std_val)
        	error = mylosses.rmse(predicted, norms.denorm_reduce(y_test[i],mean_val, std_val)) 
        	test_error.append(error)
    if(args.norm == 'log'):    
        predicted = norms.denorm_log(predicted)
        error = mylosses.rmse(predicted, norms.denorm_log(y_test[i])) 
    else:
        error = mylosses.rmse(predicted, y_test[i]) 

print('Average test rmse=',np.mean(np.array(test_error)))


# Save the model
torch.save(model.state_dict(), models_path+'OP-{}_BS-{}EP-{}_LR-{}'.format(args.optimizer, args.batch_size, args.epochs, 50.0001))

# Plot error
plt.plot(losses,label='Train loss')
plt.plot(total_val_loss,label='Validation loss')
plt.title('Training loss vs epochs')
plt.legend()
plt.show()

print('Finished Training')
