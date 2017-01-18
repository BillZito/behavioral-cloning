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
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, Conv2D, ELU, Flatten, Dense, Dropout, Lambda, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from process_data import crop_images, resize_images, show_images, change_brightness, flip_half, flip_X, flip_y, translate

np_dir = 'data/np_data/'
model_dir = 'models/'

'''
'''
def nvidia_model():
  row, col, depth = 66, 200, 3
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col, depth), output_shape=(row, col, depth)))
  
  #valid border mode should get rid of a couple each way, whereas same keeps
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))

  model.add(Flatten())
  model.add(Dropout(.5))
  model.add(Dense(100))
  ######
  #consider adding elu between dense layers
  #my guess at what's right
  model.add(Dense(50))
  #and again
  # model.add(Dropout(.3))
  # model.add(Dense(10))
  model.add(Dense(1))

  #compile and return
  model.compile(loss='mse', optimizer='adam')
  model.summary()
  return model

def comma_model():
  row, col, depth = 66, 200, 3
  shape = (row, col, depth)

  model = Sequential()

  model.add(Lambda(lambda x: x/255 -.5, input_shape=shape, output_shape=shape))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='elu'))
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', activation='elu'))
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same'))

  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dropout(.5))
  model.add(ELU())

  #the fully connected layer accounts for huge % of parameters (50+)
  model.add(Dense(200))
  model.add(Dense(100))
  model.add(Dense(1))

  model.compile(loss='mse', optimizer='adam')
  model.summary()
  return model

'''
add in validation generator
'''
def val_generator(X, y, batch_size, num_per_epoch):
  while True:
    # X, y = shuffle(X, y)
    smaller = min(len(X), num_per_epoch)
    iterations = int(smaller/batch_size)
    for i in range(iterations):
      start, end = i * batch_size, (i + 1) * batch_size
      yield X[start:end], y[start:end]

'''
create generator to create augmented images
'''
def my_generator(X, y, batch_size, num_per_epoch, n_t):

  # print('norm thresh', n_t)
  #preprocess image

  # curr_epoch += 1
  # print('curr epoch', epoch)

  while True:
    # X, y = shuffle(X, y)
    # print('range is', int(num_per_epoch/batch_size))
    smaller = min(len(X), num_per_epoch)
    iterations = int(smaller/batch_size)
    for i in range(iterations):
      start, end = i * batch_size, (i + 1) * batch_size

      # make x/y have only a certain amount of 0's by checking y vals
      # count = 1
      # new_y = y[start].reshape((1,) + y[start].shape)
      # new_X = X[start].reshape((1,) + X[start].shape)
      # while new_y.shape[0] < batch_size:
      #   random_int = random.randint(1, 100)
      #   y_val = y[count % y.shape[0]]
      #   # print('y val is', y_val)
      #   if abs(y_val) > 0 or random_int > n_t:
      #     # if random_int < 28 + 8 * epoch
      #     next_y = np.array([y[count % y.shape[0]]])
      #     next_X = np.array([X[count % X.shape[0]]])
      #     new_y = np.append(new_y, next_y, axis=0)
      #     new_X = np.append(new_X, next_X, axis=0)
      #   count += 1
      # print('y after while', new_y.shape[0])
      # print('x after while', new_X.shape[0])

      # half_flip_X, half_flip_y = flip_half(new_X, new_y)
      # translated_X, translated_y = translate(half_flip_X, half_flip_y)
      # brightness_adjusted_X = change_brightness(half_flip_X)
      # cropped_X = crop_images(translated_X, 40, 135)
      # resized_X = resize_images(cropped_X, 64, 64, batch_size)
      # translated_X, translated_y = translate(brightness_adjusted_X, half_flip_y)
      half_flip_X, half_flip_y = flip_half(X[start: end], y[start: end])
      # yield(translated_X, translated_y)
      # yield (translated_X, translated_y)
      yield(half_flip_X, half_flip_y)
      # yield(brightness_adjusted_X, half_flip_y)
      # yield X[start:end], y[start:end]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Model to train steering angles')
  parser.add_argument('--batch', type=int, default=128, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=15, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=43394, help='How many images per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='?multiple path out.')
  parser.add_argument('--features', type=str, default=np_dir + 'udacity_final_images.npy', help='File where features .npy found.')
  parser.add_argument('--labels', type=str, default=np_dir + 'udacity_angles.npy', help='File where labels .npy found.')
  parser.add_argument('--destfile', type=str, default=model_dir + 'comma_1', help='File where model found')

  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  orig_features = np.load(args.features).astype(np.float)
  orig_labels = np.load(args.labels).astype(np.float)

  '''
  double data for mini-model tessting
  python docs say ::-1 should read it backwards--- doesnt make sense how that would reverse image
  '''
  # orig_features = np.append(orig_features, orig_features[:, :,::-1], axis=0)
  # orig_labels = np.append(orig_labels, -orig_labels, axis=0)

  '''
  split into training, validation, and set to right type
  '''
  orig_features, orig_labels = shuffle(orig_features, orig_labels)
  X_train, X_val, y_train, y_val = train_test_split(orig_features, orig_labels, test_size=.1, random_state=0)
  print('X_train and y_train', X_train.shape, y_train.shape)
  print('X_val shape', X_val.shape)

  '''
  for minimodel: give depth of 1 (when only convert it to image.. but still at 3 for my data)
  '''
  # X_train = X_train.reshape(X_train.shape + (1,))
  # X_val = X_val.reshape(X_val.shape + (1,))
  # print('reshaped', X_train.shape, X_val.shape)

  # # will have to change early stopping to make it work with my unique model
  # earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
  # '''
  # fit model to generated data
  # '''
  top_val = 1
  # model = nvidia_model()
  model = comma_model()

  # with open('models/nvidia_3_15.json', 'r') as jfile:
  #       model = model_from_json(json.load(jfile))

  # model.compile("adam", "mse")
  # #weights file doesnt exist yet... google this
  # weights_file = 'models/nvidia_3_15.h5'
  # #load weights into model
  # model.load_weights(weights_file)

  # history = model.fit(X_train, y_train, batch_size=args.batch, verbose=1, validation_data=(X_val, y_val))
  for i in range(0,  args.epoch):
    print('epoch ', i)
    norm_threshold = 100 * 1.0/(1 + i)
    score = model.fit_generator(
      my_generator(X=X_train, y=y_train, batch_size=args.batch, num_per_epoch=args.epochsize, n_t=norm_threshold),
      nb_epoch=1, 
      samples_per_epoch=args.epochsize,
      validation_data=val_generator(X=X_val, y=y_val, batch_size=args.batch, num_per_epoch=args.epochsize),
      nb_val_samples=800)
    
    epoch = str(i + 1)
  # epoch = '10'
    model.save_weights(args.destfile + '_' + epoch +'.h5', True)
  # save weights as json
    with open(args.destfile + '_' + epoch + '.json', 'w') as outfile: 
      json.dump(model.to_json(), outfile)
  # print('score is', score.history)
  # curr_val = score.history['val_loss'][0]
  # if curr_val < top_val:
  #   top_val = curr_val
  #   print('best score', top_val)

    #set high score to whatever is highest

  #save the model
  print('images', args.features, 'labels', args.labels)
  print('saved model as', args.destfile)

  # model.save_weights(args.destfile + '.h5', True)
  # # save weights as json
  # with open(args.destfile + '.json', 'w') as outfile: 
  #   json.dump(model.to_json(), outfile)
