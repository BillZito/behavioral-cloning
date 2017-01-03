# The Car Drives Itself #

![alt tag](./sdc1.gif)

## Overview ##
Deep neural network to determine car steering angles, copied from Comma.ai's https://github.com/commaai/research/blob/master/train_steering_model.py and trained on simulator data.

1. Images recorded of car correctly steering on track and recovering from sides of track
1. The model, a mix of convolutional2d layers, ELU, and fully connected layers, is trained (see model section for more details)
1. Drive.py sends the predicted steering angle while the simulator runs

## Files ##
1. model.py - The script used to create and train the model.
1. drive.py - Sends information to the simulator about throttle and steering angle.
1. model.json - The model architecture.
1. model.h5 - The model weights.

## Model ##
1. convolutional2d layers add many more features (~300k) in pyramid fashion
1. Exponential Linear Units find relationships between conv layers
1. Fully connected layer added at end and matched to steering angles
1. Using adam optimizer, mean squared error minimized (distance from predicted steering angle and actual)
1. trained for 6 epochs with batch size 128

