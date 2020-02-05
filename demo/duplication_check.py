"""
This is to example how many images are duplicated in the test train ApolloScape 3D set:
https://github.com/idealo/imagededup
PHash:  http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
"""
import os
import matplotlib.pylab as plt
import imagededupSt
from imagededup.methods import PHash
from imagededup.utils import plot_duplicates

PATH = 'E:\DATASET\pku-autonomous-driving'

test_img_dir = os.path.join(PATH, 'test_images')
# Find similar images

# __Note:__ `max_distance_threshold` defines the threshold of differences between two images to consider them similar,
# the higher the value, the more tolerant it is in differences.
#
# Below we list the first 15 images found having similar content according to imagededup.
# To get the full list, you have to display the content of variable `duplicates`.
phasher = PHash()
duplicates = phasher.find_duplicates(image_dir=test_img_dir, scores=True, max_distance_threshold=3)

print('There are', len([x for x in duplicates if duplicates[x] != []]), 'images with similar images over', len(duplicates), 'images.')
# There are 429 images with similar images over 2021 images.

plt.figure(figsize=(20, 20))
plot_duplicates(image_dir=test_img_dir, duplicate_map=duplicates, filename='ID_5bf531cf3.jpg')