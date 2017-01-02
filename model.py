from scipy import misc
import matplotlib.pyplot as plt
import os
import random
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dense, Dropout, Lambda

img_dir = 'IMG'
img_list = os.listdir(img_dir)

#read in 10 random imagezs and plot them so we can see data
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

  plt.show()

# show_images()

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

# model = make_model()
# print('model is', model)