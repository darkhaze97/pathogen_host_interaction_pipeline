import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use('fivethirtyeight')
import cv2
from skimage import measure
from skimage.segmentation import clear_border
import pandas as pd

def main():
    cellImages = []
    nucleiImages = []
    pathogenImages = []
    
    # Scan through the nuclei, and correct...
    aList = label_nuclei_images('Images/Nuclei/')
    bList = label_pathogen_images('Images/Pathogen/')
    cList = label_cell_images('Images/Cell')

# This function is to help segment the nuclei images. It takes in a path to the
# nuclei images.
# Arguments:
#   - path (string): The path to the directory with the nuclei images.
# returns:
#   - label_images_otsu(path): A list of the labelled images.
def label_nuclei_images(path):
    return label_images_otsu(path)

# TODO
def label_pathogen_images(path):
    return label_images_otsu(path)

# TODO
def label_cell_images(path):
    return label_images_otsu(path)

# The function below takes in a path, and scans through the images to
# then return a list of the labelled images. This function is made to reduce
# repetition in nuclei and pathogen segmentation.
# arguments:
#   - path (string): The path to the directory that will have the images that are to be
#                    corrected.
# returns:
#   - images (list): The list of labelled images
# Resources used: 
#   - https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-3-otsu-thresholding/
#   - https://www.youtube.com/watch?v=qfUJHY3ku9k
def label_images_otsu(path):
    images = []
    # Obtain the file names in the directory dictated by path
    fileNames = obtain_file_names(path)
    for fileName in fileNames:
        # Append fileName onto path to form the path to the image
        imagePath = os.path.join(path, fileName)
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, alteredImg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alteredImg = clear_border(alteredImg)
        
        # Place labels on segmented nuclei
        labelImg = measure.label(alteredImg, connectivity=image.ndim)
        
        # Measure properties of the nuclei labels, such as area, diameter, etc.
        props = measure.regionprops_table(labelImg, image,
                                          properties=['label', 'area'])
        # Move this data into a pandas dataframe.           
        dframe = pd.DataFrame(props)
        
        # Remove nuclei labels that are small (these are most likely invalid)
        dframe = dframe[dframe['area'] > 100]
        print(dframe.head())
        
        plt.imshow(labelImg)
        plt.show()
        images.append(alteredImg)
    return images


# The function below takes in a path, and simply obtains the file names in that path.
# This is for reusability - other functions can use this function without having to write
# a try-except block for each of them.
# Arguments:
#   - path (string): The path to a directory
# Returns:
#   - fileNames (list<string>): List of file names
def obtain_file_names(path):
    fileNames = []
    try:
        for fileName in os.listdir(path):
            fileNames.append(fileName)
    except:
        file_exception_checking(path)
    return fileNames

# TODO
def file_exception_checking(path):
    if not (os.path.exists(path)):
        print('The directory path does not exist.')

if __name__ == '__main__':
    main()
