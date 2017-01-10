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
Combine images and save them to .npy file.
Combine labels and same them .npy file.
##########################################################################
'''

'''
save all images to file
'''
def save_center_images(img_dir, dest_file):
  img_list = os.listdir(img_dir)
  #add each to img_combo
  for img_name in img_list:
    if img_name.startswith('center'):
      img = misc.imread(img_dir + '/' + img_name)
      img_combo.append(img)

  #cast to numpy array and save to file
  all_center_images = np.array(img_combo)
  print('images shape', all_center_images.shape)
  #udacity_center_images.npy
  np.save(dest_file, all_center_images)

'''
save csv contents to a file
'''
def save_csv(dest_file):
  reader = csv.reader(open(csv_dir), delimiter=',')
  
  # split the first value based on value right after center
  all_angles = []
  for row in reader: 
    # title = row[0].split('center_')[1]
    # print('newval', title)
    steering_angle = row[3]
    all_angles.append(steering_angle)

  np_angles = np.array(all_angles)
  print('angles shape', np_angles.shape)
  # udacity_angles.npy

  np.save(dest_file, np_angles)

'''
combine my images and udacity images from numpy files
'''
def combine_images(first_src, second_src, dest_file):
  my_images = np.load(first_src)
  udacity_images = np.load(second_src)
  combined = np.append(my_images, udacity_images, axis=0)
  print('img destination:', dest_file)
  print('combined images shape', combined.shape)
  np.save(dest_file, combined)


'''
combine my labels and udacity labels from numpy files (originally from csv files) 
'''
def combine_labels(first_src, second_src, dest_file):
  my_labels = np.load(first_src)
  udacity_labels = np.load(second_src)
  combo_angles = np.append(my_labels, udacity_labels, axis=0)
  print('angle destination:', dest_file)
  print('combined labels shape', combo_angles.shape)
  np.save(dest_file, combo_angles)

'''
combine left, center, and right images and save to .npy file
'''
def lr_augment(src_dir, dest_file):
  print('starting at i')
  img_list = os.listdir(src_dir)

  l_count = 0
  # for each image, start with left and concat to center and right
  for img_name in img_list:
    if img_name.startswith('left'):
      
      img = concat(img_name, src_dir)
      
      if l_count == 0: 
        concatted_imgs = img.reshape((1,) + img.shape)
      elif l_count % 500 == 1:
        print('l_count', l_count)
        save_concat(concatted_imgs, dest_file)
        concatted_imgs = img.reshape((1,) + img.shape)
      else:
        concatted_imgs = np.append(concatted_imgs, img.reshape((1,) + img.shape), axis=0)
        if l_count % 500 == 2:
          #remove the first one that is only there as default
          concatted_imgs = np.delete(concatted_imgs, 0, 0)

      l_count += 1

  save_concat(concatted_imgs, dest_file)
  print('saved all imgs')

'''
concats left, center, and right images
'''
def concat(img_name, src_dir):
  center_name = img_name.replace('left', 'center')
  right_name = img_name.replace('left', 'right')

  left_img = misc.imread(src_dir + '/' + img_name)
  center_img = misc.imread(src_dir + '/' + center_name)
  right_img = misc.imread(src_dir + '/' + right_name)

  img = np.concatenate((left_img, center_img, right_img), axis=1)
  return img

'''
saves passed in images to end of current list of concatted files
'''
def save_concat(concatted_imgs, dest_file):
  try:
    prev_images = np.load(dest_file)
    print('found old file', prev_images.shape)
    now_images = np.append(prev_images, concatted_imgs, axis=0)
    np.save(dest_file, now_images)
    print('saved new file', now_images.shape)
  except IOError:
    print('no images found, saving to file', concatted_imgs.shape)
    np.save(dest_file, concatted_imgs)

'''
##########################################################################
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
read in 9 random images from numpy file and visualize
'''
def show_npfile_images(src_file, second_src):
  fig = plt.figure()
  img_arr = np.load(src_file)
  flipped_arr = np.load(second_src)
  print('imgarr size', img_arr.shape, flipped_arr.shape)
  #for 9 random images, print them 
  for img_num in range(1, 5):
    random_num = random.randint(0, img_arr.shape[0] - 1)
    print('img num', img_num, 'image is at', random_num )

    img = img_arr[random_num]
    fig.add_subplot(3, 3, img_num * 2 - 1)
    plt.imshow(img)

    flipped_img = flipped_arr[random_num]
    fig.add_subplot(3, 3, img_num * 2)
    plt.imshow(flipped_img)
  
  plt.show()

'''
show images to test that flipping correct
'''
def show_images(img_arr):
  fig = plt.figure()
  print('shape', img_arr.shape)
  
  for img_num in range(0, min(len(img_arr), 3)):
    print('img num is', img_num)
    img = img_arr[img_num]
    fig.add_subplot(3, 3, img_num * 2 + 1)
    plt.imshow(img)
    # flipped_img = flipped_arr[img_num]
    # fig.add_subplot(3, 3, img_num * 2 + 2)
    # plt.imshow(flipped_img)

  plt.show()


'''
plot labels to understand their distribution
'''
def plot_labels(src_file):
  labels = np.load(src_file)
  # print('lables start', labels)
  labels = np.multiply(labels.astype(float), 100)
  # print('after mult, labels are', labels)
  # print('as int, labels are', labels.astype(int))
  plt.hist(x=labels.astype(int), range=(-50, 50), bins=101)
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
-Process images to crop out unnecessary parts
-Zero-normalize labels to reduce the bias towards 0
-Flip left/right axis for images and labels (make negative)
##########################################################################
'''
'''
remove 3/4 of zero values since zeros are 50x larger than other data points currently
'''
def zero_normalize(angles_src, images_src, angles_dest_file, images_dest_file, start=0):
  labels = np.load(angles_src).astype(float)
  images = np.load(images_src)
  print('initial shapes', labels.shape, images.shape)
  normalized_labels = np.array([labels[0]])
  normalized_images = np.array([images[0]])

  index = 0
  deleted_count = 0
  for index in range(start, images.shape[0]): 
    val = labels[index]
    # print('index is', index, 'val', val)
    if index % 100 == 0: 
      print('now on index', index)

    random_num = random.randint(1, 100)

    if val != 0 or random_num < 25:
      normalized_labels = np.append(normalized_labels, np.array([val]), axis=0)
      normalized_images = np.append(normalized_images, np.array([images[index]]), axis=0)
      if normalized_labels.shape[0] % 100 == 0:
        print('now labels', normalized_labels.shape[0])
    else:
      deleted_count += 1
      if deleted_count % 100 == 0:
        print('deleted count now', deleted_count)

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
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
    rando = np.random.uniform()
    # print('rando is', rando)
    hsv[:,:, 2] = hsv[:,:, 2].astype('float64') * (.25 + rando)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # show_images(img.reshape((1,) + img.shape), new_img.reshape((1,) + new_img.shape))
    adjusted_imgs = np.append(adjusted_imgs, new_img.reshape((1,) + new_img.shape), axis=0)

  adjusted_imgs = np.delete(adjusted_imgs, 0, 0)
  return adjusted_imgs

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
def resize_file_images(img_src, dest_file, size, start=0, end=0):
  # print('started')
  img_arr = np.load(img_src)
  # print('resized_imgs shape', resized_imgs.shape)
  
  if end == 0:
    end = img_arr.shape[0]

  for i in range(start, end):
    if i % 100 == 0:
      print('index is', i)

    img = img_arr[i]
    resized = cv2.resize(img, (size, size))
    resized = resized.reshape((1,) + resized.shape)

    if i == start:
      resized_imgs = resized
    else: 
      resized_imgs = np.append(resized_imgs, resized, axis=0)

  np.save(dest_file, resized_imgs)
  print('final shape', resized_imgs.shape, 'saved to', dest_file)

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


if __name__ == '__main__':
  img_dir = 'data/images/udacity_IMG'
  csv_dir = 'data/logs/udacity_driving_logs.csv'
  np_dir = 'data/np_data/'
  # show_npfile_images(np_dir + 'udacity_images.npy', np_dir + 'udacity_final_lr_images.npy')
  # crop_file_images(np_dir + 'udacity_final_lr_images.npy', np_dir + 'new_cropped_udacity_images.npy', 60, 140)
  # resize_file_images(np_dir + 'new_cropped_udacity_images.npy', np_dir + 'first_resized_udacity_images.npy', 64, 0, 2000)
  # resize_file_images(np_dir + 'new_cropped_udacity_images.npy', np_dir + 'second_resized_udacity_images.npy', 64, 2000, 4000)
  # resize_file_images(np_dir + 'new_cropped_udacity_images.npy', np_dir + 'third_resized_udacity_images.npy', 64, 4000, 6000)
  # resize_file_images(np_dir + 'new_cropped_udacity_images.npy', np_dir + 'fourth_resized_udacity_images.npy', 64, 6000)
  # show_npfile_images(np_dir + 'third_resized_udacity_images.npy', np_dir + 'fourth_resized_udacity_images.npy')
  
  # zero_normalize(np_dir + 'udacity_angles.npy', np_dir + 'first_resized_udacity_images.npy', np_dir + 'first_udacity_norm_angles.npy', np_dir + 'first_udacity_norm_images.npy')
  # zero_normalize(np_dir + 'udacity_angles.npy', np_dir + 'second_resized_udacity_images.npy', np_dir + 'second_udacity_norm_angles.npy', np_dir + 'second_udacity_norm_images.npy')
  # zero_normalize(np_dir + 'udacity_angles.npy', np_dir + 'third_resized_udacity_images.npy', np_dir + 'third_udacity_norm_angles.npy', np_dir + 'third_udacity_norm_images.npy')
  # zero_normalize(np_dir + 'udacity_angles.npy', np_dir + 'fourth_resized_udacity_images.npy', np_dir + 'fourth_udacity_norm_angles.npy', np_dir + 'fourth_udacity_norm_images.npy')
  # show_npfile_images(np_dir + 'first_udacity_norm_images.npy', np_dir + 'fourth_udacity_norm_images.npy')
  # show_npfile_images(np_dir + 'first_udacity_norm_images.npy', np_dir + 'second_udacity_norm_images.npy')
  # combo_images = lr_augment(img_dir, np_dir + 'test_images.npy')
  # show_images(combo_images)
  # combine_images(np_dir + 'first_udacity_norm_images.npy', np_dir + 'second_udacity_norm_images.npy', np_dir + 'udacity_combo_images.npy')
  # show_npfile_images(np_dir + 'udacity_combo_images.npy', np_dir + 'udacity_combo_images.npy')
  # combine_labels(np_dir + 'first_udacity_norm_angles.npy', np_dir + 'second_udacity_norm_angles.npy', np_dir + 'udacity_combo_angles.npy')
  # combine_labels(np_dir + 'udacity_combo_angles.npy', np_dir + 'third_udacity_norm_angles.npy', np_dir + 'udacity_combo_angles.npy')
  # combine_labels(np_dir + 'udacity_combo_angles.npy', np_dir + 'fourth_udacity_norm_angles.npy', np_dir + 'udacity_combo_angles.npy')



