import os
import csv
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
def show_images():
  fig = plt.figure()

  #for 9 random images, print them 
  for img_num in range(0, 9):
    random_num = random.randint(0, len(img_list))
    img_name = img_list[random_num]
    print('image name is ', img_name)
    img = misc.imread(img_dir + '/' + img_name)
    # print('img is ', img)
    img = img[60:160]
    fig.add_subplot(3, 3, img_num + 1)
    plt.imshow(img)
  
  plt.show()

# show_images()


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

crop_images(np_dir + 'udacity_images.npy', np_dir + 'cropped_udacity_images.npy')


