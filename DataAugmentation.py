import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.utils import img_to_array, load_img
import os

os.chdir("E:\ML\DL DanceClassification\dataset")    #Change working directory

datagenerator = ImageDataGenerator(
        rescale=1. / 255, 
        rotation_range=30,  
        zoom_range = 0.15,  
        width_shift_range=0.10,  
        height_shift_range=0.10,  
        horizontal_flip=True,  
        vertical_flip=True) 

def check_dir(directory):
''' Checks if the target directory exists or not.
    If it does not exist it will create one'''
        if not os.path.exists(directory):
                os.makedirs(directory)


def augmentation(x):
''' Here the actual data augmentation takes place'''
        x = img_to_array(x)
        x = x.reshape((1,) + x.shape) 
        i=0

        for batch in datagenerator.flow(x, batch_size=1,    
                                save_to_dir='preview',    #Target directory/folder
                                save_prefix='images',     #Name of augmented images
                                save_format='jpeg'):      #Extension of target images
                i += 1
                if i > 5:        #No of target images
                        break


x = load_img("F:/PV/canon eos 1500d/20180121213556_IMG_2535 (4).JPG")   #Loading input image

check_dir('preview')        #Target directory

augmentation(x)
