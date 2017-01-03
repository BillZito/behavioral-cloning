import os
import csv
import random
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


img_dir = 'data/IMG'
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

    fig.add_subplot(3, 3, img_num + 1)
    plt.imshow(img)
  
  plt.show()

# show_images()


'''
save all images to file
'''
def save_images():
  #add each to img_combo
  for img_name in img_list:
    if img_name.startswith('center'):
      img = misc.imread(img_dir + '/' + img_name)
      img_combo.append(img)

  #cast to numpy array and save to file
  all_center_images = np.array(img_combo)
  print('all_images shape is', all_center_images.shape)
  np.save('udacity_center_images.npy', all_center_images)

# save_images()
# test = np.load('center_images.npy')
# print('test.shape', test.shape)


'''
save csv contents to a file
'''
def save_csv():
  reader = csv.reader(open('data/driving_log.csv'), delimiter=',')
  
  # split the first value based on value right after center
  all_angles = []
  for row in reader: 
    # title = row[0].split('center_')[1]
    # print('newval', title)
    steering_angle = row[3]
    all_angles.append(steering_angle)

  np_angles = np.array(all_angles)
  print('all angles', np_angles.shape)
  np.save('udacity_angles.npy', np_angles)

# save_csv()

def combine_images():
  my_images = np.load('center_images.npy')
  udacity_images = np.load('udacity_center_images.npy')
  combined = np.append(my_images, udacity_images, axis=0)
  print('myimages shape', combined.shape)
  np.save('combined_images.npy', combined)

# combine_images()

def combine_labels():
  my_labels = np.load('angles.npy')
  udacity_labels = np.load('udacity_angles.npy')
  combo_angles = np.append(my_labels, udacity_labels, axis=0)
  print('mylabels shape', combo_angles.shape)
  np.save('combined_angles.npy', combo_angles)

combine_labels()