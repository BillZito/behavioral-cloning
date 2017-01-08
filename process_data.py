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
def show_np_images(src_file):
  fig = plt.figure()
  img_arr = np.load(src_file)
  print('imgarr size', img_arr.shape)
  #for 9 random images, print them 
  for img_num in range(1, 10):
    print('img arr shape', img_arr.shape)
    random_num = random.randint(0, img_arr.shape[0] - 1)
    img = img_arr[random_num]
    # print('img is', img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('image is at', random_num )
    fig.add_subplot(3, 3, img_num)
    plt.imshow(img)
  
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
save all images to file
'''
def save_images(dest_file):
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
crop images to remove content above horizon and hood of car
'''
def crop_images(img_src, dest_file, low_bound, top_bound):
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
remove 3/4 of zero values since zeros are 50x larger than other data points currently
'''
def zero_normalize(angles_src, images_src, angles_dest_file, images_dest_file):
  labels = np.load(angles_src)
  images = np.load(images_src)
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

'''
resize given images to 64x64-- reducing fidelity improves model speed and performance?
'''
def resize_img(img_src, dest_file, size, end=0):
  # print('started')
  img_arr = np.load(img_src)
  resized_imgs = np.zeros([1, 80, 320, 3])
  # print('resized_imgs shape', resized_imgs.shape)
  
  if end == 0:
    end = img_arr.shape[0]
  print('end is ', end)

  # count = 0
  for img in img_arr[0:end]:
    print('img[0]', img.shape)
    # if count % 100 == 0:
    # print('count is', count)
    # img = cv2.imread(np_img)
    resized = cv2.imread(img)
    resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resized = cv2.cvtColor(resized, cv2.COLOR_HSV2RGB)
    # resized = cv2.resize(img, (size, size))
    # print('resized is', resized.shape)
    new_item = resized.reshape(1, 80, 320, 3)
    resized_imgs = np.concatenate((resized_imgs, new_item), axis=0)
    # count += 1

  resized_imgs = np.delete(resized_imgs, 0, 0)
  print('resized_imgs size', resized_imgs.shape)
  np.save(dest_file, resized_imgs)
  # print('final shape', resized_imgs.shape)

if __name__ == '__main__':
  # crop_images(img_src=np_dir + 'cropped_1_3_combo_images_night_4th.npy', dest_file=np_dir + 'dcropped_1_3_combo.npy', 0, 80)
  # show_np_images(src_file=np_dir + 'dcropped_1_3_combo_images.npy')
  resize_img(img_src=np_dir + 'dcropped_1_3_combo_images.npy', dest_file=np_dir + 'deleteme.npy', size=64, end=10)
  show_np_images(src_file=np_dir + 'deleteme.npy')
  # plot_labels(np_dir + 'deleteme.npy')

  # zero_normalize(np_dir + '1_3_combo_angles_night_4th.npy', np_dir + 'cropped_1_3_combo_images_night_4th.npy', np_dir + 'normalized_angles.npy', np_dir + 'normalized_images.npy')
  # zero_normalize(np_dir + 'udacity_angles.npy', np_dir + 'cropped_udacity_images.npy', np_dir + 'udacity_normalized_angles.npy', np_dir + 'udacity_normalized_images.npy')
  # show_file_images(img_dir + '/', img_list)
  # save_images(np_dir + '1_3_bridge_recovery_2_images.npy') 
  # save_csv(np_dir + '1_3_bridge_recovery_2_angles.npy')
  # combine_labels(np_dir + '1_3_bridge_recovery_2_angles.npy', np_dir + '1_3_combo_angles_night_3rd.npy', np_dir + '1_3_combo_angles_night_4th.npy')
  # combine_images(np_dir + '1_3_bridge_recovery_2_images.npy', np_dir + '1_3_combo_images_night_3rd.npy', np_dir + '1_3_combo_images_night_4th.npy')
