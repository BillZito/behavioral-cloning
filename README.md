# The Car Drives Itself #

![alt tag](./sdc.gif)

## Overview ##
Deep neural network to drive a car in a simulator. For a deeper look at my process, check out [my article on Medium](https://medium.com/@billzito/my-first-self-driving-car-e9cd5c04f0f2#.d4ww3vea7). 

Model largely copied from [Nvidia's paper](images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Trained via 8k images on the track, which were augmented to become 48k images. 

All made possible by Udacity's Self-driving Car Nanodegree.

1. Images recorded of car correctly steering on track and recovering from sides of track
1. The model, a mix of convolutional2d layers, relu activations, and fully connected layers, is trained for ~16 epochs on 45k training images (see model section for more details)
1. Drive.py sends the predicted steering angle while the simulator runs

## Files ##
1. model.py - The script used to create and train the model.
1. drive.py - Sends information to the simulator about throttle and steering angle.
1. process_data.py - Resizes and saves the images and steering angles, as well as augmenting them in various ways.
1. model.json - The model architecture.
1. model.h5 - The model weights.
1. successful_models - A list of the models and weights for models that get around the track successfully.

## Model ##
![alt tag](./Nvidia_model.png)

1. 5 convolutional2d layers increase the feature depth.
1. Relu activations find non-linear relationships between layers.
1. Three fully connected layers added at end, eventually outputing a steering angle.
1. Using adam optimizer, mean squared error minimized (distance from predicted steering angle and actual).
1. 48k images (after processing) trained for 15 epochs with batch size 128 at .001, then for a couple more at .0001


## Acknowledgements ##
1. Thank you to Udacity for selecting me for the nanodegree and helping me meet like-minded people.
1. I drew inspiration for each step of the process from Nvidia's paper and  Comma.ai's model.
1. Several other students wrote good Medium pieces explaining their methodolgies. In particular, I am grateful for pieces by [Vivek](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.zh7bo8734), [Mengxi](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.df7dce6ih), and [Denise](https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184#.fsprdy8ok) 
