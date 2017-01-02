import os
import random
import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dense, Dropout, Lambda

img_dir = 'IMG'
img_list = os.listdir(img_dir)
img_combo = []

'''
read in 10 random imagezs and plot them so we can see data
'''
def show_images():
  fig = plt.figure()

  for img_num in range(0, 9):
    # print('now on img', img_num)
    random_num = random.randint(0, len(img_list))
    # print('random num is', random_num)
    img_name = img_list[random_num]
    print('image name is ', img_name)
    img = misc.imread(img_dir + '/' + img_name)

    fig.add_subplot(3, 3, img_num + 1)
    plt.imshow(img)

# show_images()
# plt.show()

'''
save all images to file
'''
def save_images():
  #add each to img_combo
  for img_name in img_list:
    if not img_name.startswith('.'):
      img = misc.imread(img_dir + '/' + img_name)
      img_combo.append(img)
    
  #cast to numpy array and save to file
  all_images = np.array(img_combo)
  print('all_images shape is', all_images.shape)
  np.save('images.npy', all_images)

# save_images()
test = np.load('images.npy')
print('test.shape', test.shape)
'''
create a model to train the img data with
*why need time_len = 1?
'''
def make_model(time_len=1):
  #our data, 3 color channels, 160 by 320
  ch, row, col = 3, 160, 320
  start_shape = (ch, row, col)

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
  model.compile(optimizer='adam', loss='mse')

  return model



# if __name == "__main__":
#   # set up arg parser so that we can call python file with diff args
#   parser = argparse.ArgumentParser(description='Model to train steering angles')
#   #didn't include port options since dont need to run on server
#   parser.add_argument('--batch', type=int, default=128, help='Batch size.')
#   parser.add_argument('--epoch', type=int, default=19, help='Number of epochs.')
#   parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
#   #confused by help--just skips validation when fit model right?
#   parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='?multiple path out.')
#   parser.set_defaults(skipvalidate=False)
#   parser.set_defaults(loadweights=False)
#   args = parser.parse_args()

#   model = make_model()
#   print('model is', model)

#   #try to fit model normally without generator... 
#   model.fit(nb_epcoh=args.epoch)

#   #if haven't saved ouput yet, save now
#   #turn to json