## RadarNet
Deep Learning Convolutional-Deconvolutional framework to short-term climate forecasting


## Introduction
RadarNet provides a 

Axionaut is intended for rapid experimentation, use the built-in Deep Learning architectures and start driving!


## Code style
PEP 8 -- Style Guide for Python Code.


## Screenshot
![alt text](/Photos/real_sequence_example.png)

## Tech/framework used

<b>Built using:</b>
- [PyTorch](http://pytorch.org)


## Features

1. <strong>Training mode:</strong> Easy model training with built-in database.
2. <strong>Visualization:</strong> Real time visualization of training Losses.
3. <strong>Forecasting:</strong> Built in forecasting routine to produce 50 min forecast sequences.


## API

Create a new vehicle and set it to self-driving mode is extremely easy:

	#Load self-driving pre trained model
    model, graph = load_autopilot('autopilot.hdf5')

    # Create Axionaut car with default settings
    axionaut = vehicles.Axionaut()

    # Configure PDW control commands as default
    axionaut.commands = get_commands(path=None, default=True)

    # Test camera position
    axionaut.camera_test()

    # Set vehicle to auto pilot mode 
    axionaut.autopilot(model, graph)

    # Start car   
    axionaut.start()

## Code Exemple

The following commands are avaliable when using the main.py example:

<strong>Start vehicle on self-driving mode:</strong>
`python main.py mode self_driving`

<strong>Start on recording mode:</strong>
`python main.py mode record`

<strong>Start on free ride mode:</strong>
`python main.py mode free`

<strong>To train your own driving model:</strong>
`python main.py mode train architecture ConvNets epochs 100 batch size 300 optimizer Adam`

Feel free to explore and set your prefered training hyperparameters!


## Installation
<strong>Clone repository to your laptop:</strong>
`git clone https://github.com/Axionable/AxionautV1`

<strong>Install packages:</strong>
`pip install -r laptop_requirements.txt`


## Status

RadarNet is currently under active developement.

## Contribute

RadarNet is totally free and open for everyone to use, please feel free to contribute!



