from scipy import misc
import matplotlib.pyplot as plt
import os
import random
#read in 10 random images and plot them so we can see data

img_dir = 'IMG'
img_list = os.listdir(img_dir)

fig = plt.figure()

for img_num in range(0, 16):
  print('now on img', img_num)
  random_num = random.randint(0, len(img_list))
  print('random num is', random_num)
  img_name = img_list[random_num]
  print('image name is ', img_name)
  img = misc.imread(img_dir + '/' + img_name)
  fig.add_subplot(4, 4, img_num + 1)
  plt.imshow(img)

print('checked all iamges')
plt.show()

# load in images -- they're all 320 by 160 jpg images
# img1 = misc.imread(file1)
# img2 = misc.imread(file2)
# plt.figure()
# fig.add_subplot(1, 2, 1)
# plt.imshow(img1)
# # plt.figure()
# fig.add_subplot(1, 2, 2)
# plt.imshow(img2)  
# plt.show()

# print (i have an image)
# make it show up in some plot
# 