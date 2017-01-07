import os
import csv
import cv2
import random
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


img_dir = 'data/bridge_recovery_2_IMG'
csv_dir = 'data/bridge_recovery_2_driving_log.csv'
np_dir = 'data/np_data/'
img_list = os.listdir(img_dir)
img_combo = []

'''
read in 10 random imagezs and plot them so we can see data
'''
def show_images(filename, img_list):
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

# show_images(img_dir + '/', img_list)


'''
save all images to file
'''
def save_images(filename):
  #add each to img_combo
  for img_name in img_list:
    if img_name.startswith('center'):
      img = misc.imread(img_dir + '/' + img_name)
      img_combo.append(img)

  #cast to numpy array and save to file
  all_center_images = np.array(img_combo)
  print('images shape', all_center_images.shape)
  #udacity_center_images.npy
  np.save(filename, all_center_images)

# save_images(np_dir + '1_3_bridge_recovery_2_images.npy') 


'''
save csv contents to a file
'''
def save_csv(filename):
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
  np.save(filename, np_angles)

# save_csv(np_dir + '1_3_bridge_recovery_2_angles.npy')

'''
combine my images and udacity images from numpy files
'''
def combine_images(first_file, second_file, dest_file):
  my_images = np.load(first_file)
  udacity_images = np.load(second_file)
  combined = np.append(my_images, udacity_images, axis=0)
  print('img destination:', dest_file)
  print('combined images shape', combined.shape)
  np.save(dest_file, combined)

# combine_images(np_dir + '1_3_bridge_recovery_2_images.npy', np_dir + '1_3_combo_images_night_3rd.npy', np_dir + '1_3_combo_images_night_4th.npy')

'''
combine my labels and udacity labels from numpy files (originally from csv files) 
'''
def combine_labels(first_file, second_file, dest_file):
  my_labels = np.load(first_file)
  udacity_labels = np.load(second_file)
  combo_angles = np.append(my_labels, udacity_labels, axis=0)
  print('angle destination:', dest_file)
  print('combined labels shape', combo_angles.shape)
  np.save(dest_file, combo_angles)

# combine_labels(np_dir + '1_3_bridge_recovery_2_angles.npy', np_dir + '1_3_combo_angles_night_3rd.npy', np_dir + '1_3_combo_angles_night_4th.npy')

def crop_images(image_file, dest_file):
  my_images = np.load(image_file)
  cropped_images = []
  # count = 0
  for img in my_images:
    # if count < 10:
    img = img[60:160]
    # print('i is', count)
    # print('image shape', img.shape)
    cropped_images.append(img)
    # count += 1
  np_cropped = np.array(cropped_images)
  print('cropped images is', np_cropped.shape)
  np.save(dest_file, np_cropped)
  print('dest file is', dest_file)

# crop_images(np_dir + '1_3_uncropped_normalized_images.npy', np_dir + '1_3_normalized_images.npy')

def plot_images(filename):
  labels = np.load(filename)
  print('lables start', labels)
  labels = np.multiply(labels.astype(float), 100)
  print('after mult, labels are', labels)
  print('as int, labels are', labels.astype(int))
  plt.hist(x=labels.astype(int), range=(-50, 50), bins=101)
  plt.show()

# plot_images(np_dir + '1_3_combo_angles_night_4th.npy')

def zero_normalize(angles_filename, images_filename, angles_dest_file, images_dest_file):
  labels = np.load(angles_filename)
  images = np.load(images_filename)
  print('initial shapes', labels.shape, images.shape)
  normalized_labels = np.array([labels[0]])
  normalized_images = np.array([images[0]])

  deleted_count = 0
  for index, val in np.ndenumerate(labels.astype(float)): 
    # print('index is', index[0], 'val', val)
    if index[0] % 100 == 0: 
      print('now on index', index)

    random_num = random.randint(1, 100)

    if val != 0 or random_num < 25:
      normalized_labels = np.append(normalized_labels, np.array([labels[index]]), axis=0)
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

# zero_normalize(np_dir + '1_3_combo_angles_4th.npy', np_dir + 'cropped_1_3_combo_images_4th.npy', np_dir + 'normalized_angles.npy', np_dir + 'normalized_images.npy')
# zero_normalize(np_dir + 'udacity_angles.npy', np_dir + 'cropped_udacity_images.npy', np_dir + 'udacity_normalized_angles.npy', np_dir + 'udacity_normalized_images.npy')

def make_64(filename, dest_file):
  img_arr = np.load(filename)
  resized_imgs = np.array([img_arr[0]])
  count = 0
  for np_img in img_arr:
    if count > 10:
      img = cv2.imread(np_img)
      cv2.resize(img, (64, 64))
      print
      resized = 


make_64(np_dir + 'cropped_udacity_images.npy', np_dir + 'small_cropped_udacity_images.npy')
