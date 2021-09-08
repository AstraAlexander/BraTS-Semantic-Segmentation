# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:40:52 2021

@author: Alexander
"""

"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators. 
No image processing operations are performed here, just load data from local directory
in batches. 
"""

#from tifffile import imsave, imread
import os
import numpy as np


def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        #if you wanna load like .tif-files, change 'npy' to 'tif' and use another loading-command like from cv2
        if (image_name.split('.')[1] == 'npy'):
            #if my data is a .npy data, use np.load to get access to it
            image = np.load(img_dir+image_name)
            #appending the loaded image to the list         
            images.append(image)
    images = np.array(images) #Converting it to an numpy Array
    
    return(images)




def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])      #Loading an image
            Y = load_img(mask_dir, mask_list[batch_start:limit])    #Loading a mask
            
            #not returning, but yielding X and Y
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            #Counter
            batch_start += batch_size   
            batch_end += batch_size

############################################

#Test the generator

from matplotlib import pyplot as plt
import random

train_img_dir = r"C:/Users/Alexander/Desktop/GitHub/BraTS-Semantic_Segmentation/BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = r"C:/Users/Alexander/Desktop/GitHub/BraTS-Semantic_Segmentation/BraTS2020_TrainingData/input_data_128/train/masks/"
#you can use some kind of sorted() function so sort things up
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

#Verify generator.... 
img, msk = train_img_datagen.__next__()


img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()