'''
Trains model for determining steering angle
'''
import cv2
import json
import random
import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dense, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from process_data import crop_images, resize_images, show_images, change_brightness, flip_half, flip_X, flip_y

np_dir = 'data/np_data/'
model_dir = 'models/'

'''
create a model to train the img data with
*why need time_len = 1?
'''
def make_model(time_len=1):
  #our data, 3 color channels, 64 by 64
  row, col, ch = 64, 64, 3
  start_shape = (row, col, ch)

  #set up sequential linear model (stacked on top of eachother)
  model = Sequential()

  #normalize pixels from 255 to be out of 1
  model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(start_shape), output_shape=(start_shape)))

  #convolutional 16, 8, 8 with subsample aka stridelength (4, 4), same padding means that it doesn't lose part on end? 
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same'))

  #ELU aka exponential linear unit(similar to RELU)
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same'))

  #flatten and dropout .2
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())

  # dense (aka normal fully connected) -- outputs to size 512 (and doesnt matter input)
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())

  #dense 1 -- outputs to 1 and then can 1-hot it?
  model.add(Dense(1)) 

  #compile model and return. setting loss to mean standard error because regression
  #no metrics
  adam = Adam(lr=0.0001)
  model.compile(optimizer=adam, loss='mse')

  return model

'''
create generator to create augmented images
'''
def my_generator(X, y, batch_size, num_per_epoch):

  print('generator starting')
  #preprocess image

  while True:
    # print('range is', int(num_per_epoch/batch_size))
    smaller = min(len(X), num_per_epoch)
    iterations = int(smaller/batch_size)
    for i in range(iterations):
      X, y = shuffle(X, y)
      start, end = i * batch_size, (i + 1) * batch_size
      half_flip_X, half_flip_y = flip_half(X_train[start: end], y_train[start: end])
      brightness_adjusted_imgs = change_brightness(half_flip_X)
      # cropped_imgs = crop_images(brightness_adjusted_imgs, 60, 140)
      # resized_imgs = resize_images(cropped_imgs, 64, 64)
      yield (brightness_adjusted_imgs, half_flip_y)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Model to train steering angles')
  parser.add_argument('--batch', type=int, default=256, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=20000, help='How many images per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='?multiple path out.')
  parser.add_argument('--features', type=str, default=np_dir + 'lrc_combo_images.npy', help='File where features .npy found.')
  parser.add_argument('--labels', type=str, default=np_dir + 'lrc_combo_angles.npy', help='File where labels .npy found.')
  parser.add_argument('--destfile', type=str, default=model_dir + 'generator_20', help='File where model found')

  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  orig_features = np.load(args.features)
  orig_labels = np.load(args.labels)

  '''
  split into training, validation, and set to right type
  '''
  X_train, X_val, y_train, y_val = train_test_split(orig_features, orig_labels, test_size=.1, random_state=0)
  y_train = y_train.astype(np.float)
  y_val = y_val.astype(np.float)
  print('X_train and y_train', X_train.shape, y_train.shape)

  '''
  fit model to generated data
  '''
  model = make_model()
  model.fit_generator(
    my_generator(X=X_train, y=y_train, batch_size=args.batch, num_per_epoch=args.epochsize),
    nb_epoch=args.epoch, 
    samples_per_epoch=args.epochsize,
    validation_data=my_generator(X=X_val, y=y_val, batch_size=args.batch, num_per_epoch=args.epochsize),
    nb_val_samples=800)

  #save the model
  print('saving model as', args.destfile)

  model.save_weights(args.destfile + '.h5', True)
  # save weights as json
  with open(args.destfile + '.json', 'w') as outfile: 
    json.dump(model.to_json(), outfile)
