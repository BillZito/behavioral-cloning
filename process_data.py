import os
import csv
import cv2
import random
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


'''
##########################################################################
1. Save labels and images to .npy files
##########################################################################
'''

'''
save all images (not just center) to file
'''
def save_images(img_dir, dest_file):
  img_list = os.listdir(img_dir)
  img_combo = []

  print('starting to save ' + str(len(img_list)) + ' images')

  count = 0
  for img_name in img_list:
    # can change this line to img_name.startswith('center') for center imgs
    if not img_name.startswith('.'):

      if count % 500 == 0:
        print('count is', count)

      img = misc.imread(img_dir + '/' + img_name)
      img_combo.append(img)
      count += 1

  #cast to numpy array and save to file
  all_images = np.array(img_combo)
  print('images shape', all_images.shape)
  np.save(dest_file, all_images)


'''
save csv contents to a file
'''
def save_csv(csv_dir, dest_file):
  reader = csv.reader(open(csv_dir), delimiter=',')
  
  # split the first value based on value right after center
  all_angles = []
  for row in reader: 
    steering_angle = row[3]
    all_angles.append(steering_angle)

  np_angles = np.array(all_angles)
  print('angles shape', np_angles.shape)
  np.save(dest_file, np_angles)


'''
save csv with left and right
'''
def save_csv_lrc(csv_dir, dest_file, correction):
  reader = csv.reader(open(csv_dir), delimiter=',')
  
  # split the first value based on value right after center
  all_angles = []
  for row in reader: 
    steering_angle = float(row[3])
    all_angles.append(steering_angle)
  print('done with center', len(all_angles))
  
  #left (based on order of images in logs)
  reader = csv.reader(open(csv_dir), delimiter=',')
  for row in reader:
    steering_angle = float(row[3]) + correction
    # print('steering angle is', steering_angle)
    all_angles.append(steering_angle)
  print('done with left', len(all_angles))

  #right
  reader = csv.reader(open(csv_dir), delimiter=',')
  for row in reader: 
    steering_angle = float(row[3]) - correction
    all_angles.append(steering_angle)

  np_angles = np.array(all_angles)
  print('angles shape', np_angles.shape)
  # udacity_angles.npy

  np.save(dest_file, np_angles)

'''
combine two files
'''
def combine_images(first_src, second_src, dest_file):
  first_imgs = np.load(first_src)
  second_imgs = np.load(second_src) 
  combined = np.append(first_imgs, second_imgs, axis=0)
  print('img destination:', dest_file)
  print('combined images shape', combined.shape)
  np.save(dest_file, combined)

'''
combine two sets of images from numpy files
'''
def combine_lrc_images(first_src, second_src, dest_file):
  first_imgs = np.load(first_src)
  length_of_first = first_imgs.shape[0]
  third_length_of_first = int(length_of_first / 3)

  second_imgs = np.load(second_src) 
  length_of_second = second_imgs.shape[0]
  third_length_of_second = int(length_of_second / 3)

  center_combined = np.append(first_imgs[0 : third_length_of_first], second_imgs[0 : third_length_of_second], axis=0)
  left_combined = np.append(first_imgs[third_length_of_first : 2*third_length_of_first], second_imgs[third_length_of_second : 2*third_length_of_second], axis=0)
  right_combined = np.append(first_imgs[2*third_length_of_first : length_of_first], second_imgs[2*third_length_of_second : length_of_second], axis=0)
  
  cl_combo = np.append(center_combined, left_combined, axis=0)
  combined = np.append(cl_combo, right_combined, axis=0)
  
  print('img destination:', dest_file)
  print('combined images shape', combined.shape)
  np.save(dest_file, combined)

'''
given the length of shape, for each set of images, find correct file and combine
'''
def combine_all(np_dir, img_prefix, dest_name, length):
  print('length is', length)
  last_start = 4000
  start_str = '0_2000_'
  second_str = '2000_4000_'
  combine_images(np_dir + start_str + img_prefix, np_dir + second_str + img_prefix, np_dir + dest_name)
  
  for i in range(6000, length, 2000):
    print('combining', np_dir + str(last_start) + '_' + str(i) + '_' + img_prefix, 'range', last_start, i)
    combine_images(np_dir + dest_name, np_dir + str(last_start) + '_' + str(i)  + '_' + img_prefix, np_dir + dest_name)
    last_start = i

  #combine last set
  print('length is ', length, 'last start is', last_start)
  if (length - last_start != 2000):
    print('combining', np_dir + str(last_start) + '_' + str(length) + '_' + img_prefix, 'range', last_start, length)
    combine_images(np_dir + dest_name, np_dir + str(last_start) + '_' + str(length)  + '_' + img_prefix, np_dir + dest_name)
  print('combined all')


'''
combine two sets of labels rom numpy files 
'''
def combine_labels(first_src, second_src, dest_file):
  my_labels = np.load(first_src)
  udacity_labels = np.load(second_src)
  combo_angles = np.append(my_labels, udacity_labels, axis=0)
  print('angle destination:', dest_file)
  print('combined labels shape', combo_angles.shape)
  np.save(dest_file, combo_angles)

'''
combine left right center
'''
def combine_lrc_labels(first_src, second_src, dest_file):
  first_labels = np.load(first_src)
  second_labels = np.load(second_src)

  length_of_first = first_labels.shape[0]
  third_length_of_first = int(length_of_first / 3)

  length_of_second = second_labels.shape[0]
  third_length_of_second = int(length_of_second / 3)

  center_combined = np.append(first_labels[0 : third_length_of_first], second_labels[0 : third_length_of_second], axis=0)
  left_combined = np.append(first_labels[third_length_of_first : 2*third_length_of_first], second_labels[third_length_of_second : 2*third_length_of_second], axis=0)
  right_combined = np.append(first_labels[2*third_length_of_first : length_of_first], second_labels[2*third_length_of_second : length_of_second], axis=0)
  
  cl_combo = np.append(center_combined, left_combined, axis=0)
  combo_angles = np.append(cl_combo, right_combined, axis=0)
  print('angle destination:', dest_file)
  print('combined labels shape', combo_angles.shape)
  np.save(dest_file, combo_angles)


'''
##########################################################################
2. Display images
Show images from directoy, .npy files, and from numpy arrays
Plot labels to see their distribution
##########################################################################
'''

'''
read in 9 random imagezs from img file and visualize them
'''
def show_file_images(filename, img_list):
  fig = plt.figure()

  #for 9 random images, print them 
  for img_num in range(0, 9):
    random_num = random.randint(0, len(img_list))
    img_name = img_list[random_num]
    print('image name is ', img_name)
    img = misc.imread(filename + img_name)
    np_img = np.array(img)
    flipped_img = np.fliplr(np_img)[60:160]

    # print('img is ', img)
    img = img[60:160]
    fig.add_subplot(5, 5, img_num * 2 + 1)
    plt.imshow(img)
    fig.add_subplot(5, 5, img_num * 2 + 2)
    plt.imshow(flipped_img)
  
  plt.show()

'''
show images to test that flipping correct
'''
def show_images(img_arr):
  fig = plt.figure()
  print('shape', img_arr.shape)
  
  for img_num in range(1, min(len(img_arr), 10)):
    print('img num is', img_num)
    img = img_arr[img_num]
    fig.add_subplot(3, 3, img_num)
    plt.imshow(img)

  plt.show()

'''
show images to test that flipping correct
'''
def show_images_angles(img_arr, img_angles, mode=0):
  if mode == 1:
    img_arr = np.load(img_arr)
    img_angles = np.load(img_angles)

  fig = plt.figure()
  print('shape', img_arr.shape)
  
  for img_num in range(1, min(len(img_arr), 10)):
    print('img num is', img_num)
    img = img_arr[img_num]
    fig.add_subplot(3, 3, img_num)
    plt.title(round(img_angles[img_num], 2))
    plt.imshow(img)

  plt.show()

'''
show images to test that flipping correct
'''
def show_lrc_images_angles(img_arr, img_angles, mode=0):
  if mode == 1:
    img_arr = np.load(img_arr)
    img_angles = np.load(img_angles)

  third_length = int(img_arr.shape[0] / 3)
  print('third_length', third_length)
  fig = plt.figure()
  
  for img_num in range(0, min(len(img_arr), 3)):
    rand_num = random.randint(0, third_length - 1)

    l_num = third_length + rand_num
    print('left num is ', l_num)
    l_img = img_arr[l_num]
    fig.add_subplot(3, 3, img_num * 3 + 1)
    plt.title(str(round(img_angles[l_num], 2)))
    plt.imshow(l_img, cmap='gray')

    print('img num is', img_num)
    img = img_arr[rand_num]
    fig.add_subplot(3, 3, img_num * 3 + 2)
    plt.title(str(round(img_angles[rand_num], 2)))
    plt.imshow(img, cmap='gray')

    r_num = third_length * 2 + rand_num
    print('right num is', r_num)
    r_img = img_arr[r_num]
    fig.add_subplot(3, 3, img_num * 3 + 3)
    plt.title(str(round(img_angles[r_num], 2)))
    plt.imshow(r_img, cmap='gray')

  plt.show()

'''
show a single image
'''
def show_image(img):
  plt.imshow(img)
  plt.show()


'''
plot labels to understand their distribution
'''
def plot_labels(src_file):
  labels = np.load(src_file).astype(float)
  labels = np.multiply(labels, 100)
  # print('as int, labels are', labels.astype(int))
  plt.hist(x=labels.astype(int), range=(-100, 100), bins=201)
  plt.show()


'''
ouptut how many of l/r/or center images
'''
def count_images(img_dir):
  #add each to img_combo
  img_list = os.listdir(img_dir)
  l_count = 0
  c_count = 0
  r_count =0
  for img_name in img_list:
    if img_name.startswith('center'):
      c_count += 1
    elif img_name.startswith('left'):
      l_count += 1
    elif img_name.startswith('right'):
      r_count +=1
      # img = misc.imread(img_dir + '/' + img_name)
      # img_combo.append(img)
  print('counts l, c, r:', l_count, c_count, r_count)


'''
##########################################################################
3. Process images
Process images to crop out unnecessary parts
Zero-normalize labels to reduce the bias towards 0
Flip left/right axis for images and labels (make negative)
##########################################################################
'''
'''
remove 3/4 of zero values since zeros are 50x larger than other data points currently
'''
def zero_normalize(angles_src, images_src, angles_dest_file, images_dest_file, start=0):
  # load files
  labels = np.load(angles_src)
  images = np.load(images_src)
  print('initial shapes', labels.shape, images.shape)
  normalized_labels = np.array([labels[0]])
  normalized_images = np.array([images[0]])

  # for each value, randomly remove ~3/4 of 0's and save to new array
  index = 0
  deleted_count = 0
  for index in range(start, start + images.shape[0]): 
    val = labels[index]
    # print('val', val)
    if index % 500 == 0: 
      print('now on index', index)

    random_num = random.randint(1, 100)

    if val != 0 or random_num < 25:
      #all angles given since array is relatively small, whereas only range of images given to function
      #thereore, the correct image index is the index - the start value of the range
      normalized_labels = np.append(normalized_labels, np.array([val]), axis=0)
      normalized_images = np.append(normalized_images, np.array([images[index - start]]), axis=0)
      if normalized_labels.shape[0] % 500 == 0:
        print('now labels', normalized_labels.shape[0])
    else:
      deleted_count += 1
      if deleted_count % 500 == 0:
        print('deleted count now', deleted_count)

  #save the images
  print('0s deleted', deleted_count)
  print('total vals now', normalized_labels.shape, normalized_images.shape)
  normalized_labels = np.delete(normalized_labels, 0, 0)
  normalized_images = np.delete(normalized_images, 0, 0)
  np.save(angles_dest_file, normalized_labels)
  np.save(images_dest_file, normalized_images)


'''
flip images horizontally
'''
def flip_X(images):
  # initialize with correct size
  # print('flip x called', images.shape)
  flipped_imgs = np.array([images[0]])
  for i in range(len(images)):
    flip = np.fliplr(images[i])
    flipped_imgs = np.append(flipped_imgs, flip.reshape((1,) + flip.shape), axis=0)
    # print('flipped imgs appended', i)

  # remove first image which was just there to initialize size
  flipped_imgs = np.delete(flipped_imgs, 0, 0)
  return flipped_imgs


'''
flip labels to negative
'''
def flip_y(labels): 
  # print('flip y called', labels.shape)
  for i in range(len(labels)):
    labels[i] = labels[i] * -1
  return labels


'''
for half of images and labels given, flip them, then return
'''
def flip_half(X, y):
  shuffled_X, shuffled_y = shuffle(X, y)
  half = int(len(X) / 2)
  end = len(X)

  half_flipped_X = flip_X(shuffled_X[0:half])
  modified_X = np.concatenate([half_flipped_X, shuffled_X[half:end]])

  half_flipped_y = flip_y(shuffled_y[0:half])
  modified_y = np.concatenate([half_flipped_y, shuffled_y[half:end]])
  # print('modified shapes', modified_X.shape, modified_y.shape)
  return modified_X, modified_y


'''
change the brightness for each img in array
'''
def change_brightness(img_arr):
  # print('change brightness called')
  adjusted_imgs = np.array([img_arr[0]])
  for img_num in range(0, len(img_arr)):
    img = img_arr[img_num]
    # print('array access')
    # show_image(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
    # print('rgb2hsv')
    # show_image(hsv)
    rando = np.random.uniform()
    # print('rando is', rando)
    hsv[:,:, 2] = hsv[:,:, 2] * (.25 + rando)
    
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # print('hsv2rgb')
    # show_image(new_img)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # show_images(img.reshape((1,) + img.shape), new_img.reshape((1,) + new_img.shape))
    adjusted_imgs = np.append(adjusted_imgs, new_img.reshape((1,) + new_img.shape), axis=0)

  adjusted_imgs = np.delete(adjusted_imgs, 0, 0)
  return adjusted_imgs

'''
change brightness of one image
'''
def change_one(img):
  print('before')
  show_image(img)
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
  print('rgb2hsv')
  show_image(hsv)
  rando = np.random.uniform()
  # print('rando is', rando)
  hsv[:,:, 2] = hsv[:,:, 2] * (.25 + rando)
  
  new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  print('hsv2rgb')
  show_image(new_img)

'''
resize given images to 64x64-- reducing fidelity improves model speed and performance?
'''
def resize_images(img_arr, width, height, end=2000):
  # print('resized_imgs shape', resized_imgs.shape)
  if end == 0:
    end = img_arr.shape[0]

  for i in range(0, end):
    img = img_arr[i]
    resized = cv2.resize(img, (width, height))
    resized = resized.reshape((1,) + resized.shape)

    if i == 0:
      resized_imgs = resized
    else: 
      resized_imgs = np.append(resized_imgs, resized, axis=0)
    
  # print('resized_imgs size', resized_imgs.shape)
  return resized_imgs


'''
resize given images to 64x64-- reducing fidelity improves model speed and performance?
(from file)
'''
def resize_file_images(img_src, dest_file, width, height, start=0, end=0):
  # print('started')
  img_arr = np.load(img_src)
  # print('resized_imgs shape', resized_imgs.shape)
  
  if end == 0:
    end = img_arr.shape[0]

  for i in range(start, end):
    if i % 500 == 0:
      print('index is', i)

    img = img_arr[i]
    ################################################################################
    #remove the color change when dont want that
    resized = cv2.resize(img, (width, height))
    # resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1], (width, height))
    # cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(32,16))
    resized = resized.reshape((1,) + resized.shape)

    if i == start:
      resized_imgs = resized
    else: 
      resized_imgs = np.append(resized_imgs, resized, axis=0)

  np.save(dest_file, resized_imgs)
  print('final shape', resized_imgs.shape, 'saved to', dest_file)


'''
for every 2000 images, resize and save
unfortunately doesnt do the last 100 messages because less than 2000
'''
def resize_all(src_file, np_dir, dest_name, width, height):
  imgs = np.load(src_file)
  end = imgs.shape[0]
  last_start = 0

  for i in range(2000, end, 2000):
    print('resizing to', np_dir + str(last_start) + '_' + str(i) + '_' + dest_name, 'range', last_start, i)
    resize_file_images(src_file, np_dir + str(last_start) + '_' + str(i) + '_' + dest_name, width, height, last_start, i)
    last_start = i

  #combine last set
  print('end is ', end, 'last start is', last_start)
  if (end - last_start != 2000):
    print('resizing to', np_dir + str(last_start) + '_' + str(end) + '_' + dest_name, 'range', last_start, end)
    resize_file_images(src_file, np_dir + str(last_start) + '_' + str(end) + '_' + dest_name, width, height, last_start, end)
  print('resized all')


'''
crop images to remove content above horizon and hood of car
'''
def crop_images(img_arr, low_bound, top_bound):
  cropped_images = []
  # count = 0
  for i in range(0, len(img_arr)):
    img = img_arr[i]
    # if count < 10:
    img = img[low_bound:top_bound]
    # print('i is', count)
    # print('image shape', img.shape)
    cropped_images.append(img)
    # count += 1
  np_cropped = np.array(cropped_images)
  # print('cropped images is', np_cropped.shape)
  # np.save(dest_file, np_cropped)
  # print('dest file is', dest_file)
  return np_cropped


'''
crop images to remove content above horizon and hood of car (from file)
'''
def crop_file_images(img_src, dest_file, low_bound, top_bound):
  my_images = np.load(img_src)
  cropped_images = []
  # count = 0
  for img in my_images:
    # if count < 10:
    img = img[low_bound:top_bound]
    # print('i is', count)
    # print('image shape', img.shape)
    cropped_images.append(img)
    # count += 1
  np_cropped = np.array(cropped_images)
  print('cropped images is', np_cropped.shape)
  np.save(dest_file, np_cropped)
  print('dest file is', dest_file)


'''
given an image, provide a small translation and a small change of angle
'''
def translate(X, y):
  # print('starting shape', X.shape, y.shape)
  #intialize new image set 
  translated_images = np.array([X[0]])
  #and new angle set
  translated_labels = []
  # for each image
  for img_num in range(X.shape[0]):
    # get a random number between 0 and 1/5 of width of image
    random_num = random.randint(-16, 16)
    # change image by random num
    t_image = translate_image(X[img_num], random_num)
    # append image to new image set
    translated_images = np.append(translated_images, t_image.reshape((1,) + t_image.shape), axis=0)
    # change angle by random num * .002
    translation = random_num * .0106
    new_val = y[img_num] + translation
    # print('random num', random_num, 'translation', translation, 'prev val', y[img_num], 'new val', new_val)
    # show_image(t_image)
    # append new angles
    translated_labels.append(new_val)

  translated_images = np.delete(translated_images, 0, 0)
  translated_labels = np.array(translated_labels)
  # return new image set and new angle set
  # print('ending shape', translated_images.shape, translated_labels.shape)
  return translated_images, translated_labels

'''
translate one images
'''
def translate_image(old_img, amount):
  # print('shape is', old_img.shape)
  rows,cols, depth = old_img.shape
  new_shape = np.float32([[1, 0, amount], [0, 1, 0]])
  t_image = cv2.warpAffine(old_img, new_shape, (cols, rows))
  # show_image(t_image)
  return t_image
  # for a given image, provide translation based on given val
  # return translated_images

# def translate_angle(angle, )


if __name__ == '__main__':
  img_dir = 'data/images/udacity_IMG'
  csv_dir = 'data/logs/udacity_driving_log.csv'
  np_dir = 'data/np_data/'

  ##########################################################################################
  '''
  my workflow
  1. load in images
  2. load in lrc with .3 diff
  3. resize images to 40-80 (no color change for now)
  '''
  # #1. load in images
  # save_images(img_dir, np_dir + 'udacity_images.npy')

  # #2. load in lrc with .3 diff
  # save_csv_lrc(csv_dir, np_dir + 'udacity_angles.npy', .3)

  # #visualize
  # show_lrc_images_angles(img_arr=np_dir + 'udacity_images.npy', img_angles=np_dir + 'udacity_angles.npy', mode=1)
  # plot_labels(np_dir + 'udacity_angles.npy')

  #2.5 crop off top 20 pixels from top
  # crop_file_images(np_dir + 'udacity_images.npy', np_dir + 'udacity_cropped2_images.npy', 40, 160)
  # show_lrc_images_angles(img_arr=np_dir + 'udacity_cropped2_images.npy', img_angles=np_dir + 'udacity_angles.npy', mode=1)

  # #3. resize images to 40-80 (no color change for now) and combine
  # resize_all(np_dir + 'udacity_cropped_images.npy', np_dir, 'udacity_r_images.npy', 100, 33)
  # angles = np.load(np_dir + 'udacity_test_angles.npy')
  # length = angles.shape[0]
  # combine_all(np_dir=np_dir, img_prefix='udacity_r_images.npy', dest_name='udacity_c1_final_images.npy', length=length)

  # # #visualize
  # show_lrc_images_angles(img_arr=np_dir + 'udacity_c1_final_images.npy', img_angles=np_dir + 'udacity_angles.npy', mode=1)
  # plot_labels(np_dir + 'udacity_angles.npy')
