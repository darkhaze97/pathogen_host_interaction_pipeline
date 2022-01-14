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
from skimage import measure, segmentation
from skimage.segmentation import clear_border
import pandas as pd
import PIL



from Huang_Thresholding import process_image, get_positions

def main():
    cellImages = []
    nucleiImages = []
    pathogenImages = []
    
    # Scan through the nuclei, and correct...
    aList = label_nuclei_images('Images/Nuclei/')
    bList = label_pathogen_images('Images/Pathogen/')
    cList = label_cell_images('Images/Cell')
    
    find_intersection(aList[1], cList[1])
    
    # segj = np.in1d(aList[0], cList[0]).reshape(aList[0].shape)
    # print(segj)
    # aList[0][segj] = 0
    # plt.imshow(aList[0])
    # plt.show()
    # segj = segmentation.join_segmentations(aList[0], cList[0])
    # print(type(segj))
    
    # for i in range(0, len(aList[0])):
    #     arr2 = np.array([])
    #     for j in range(0, len(aList[0][i])):
    #         if (aList[0][i][j] == 0):
    #             np.append(arr2, [0])
    #             #arr2.append(0)
    #         elif (cList[0][i][j] == 0):
    #             np.append(arr2, [0])
    #             #arr2.append(0)
    #         else:
    #             print('Hi')
    #             np.append(arr2, [1])
    #             #arr2.append(1)
    # # np.reshape(arr, (len(aList[0]), len(aList[0][0])))
    
    # plt.imshow(segj)
    # plt.show()

# This function is to help segment the nuclei images. It takes in a path to the
# nuclei images.
# Arguments:
#   - path (string): The path to the directory with the nuclei images.
# returns:
#   - label_images_otsu(path): A list of the labelled images.
def label_nuclei_images(path):
    return label_images_otsu(path, False)

# TODO
def label_pathogen_images(path):
    return label_images_otsu(path, False)

# This function is to help segment the cell images. It takes in a path to the
# cell images.
# Arguments:
#   - path (string): The path to the directory with the cell images.
# returns:
#   - label_images_huang(path): A list of the labelled cell images.
def label_cell_images(path):
    return label_images_otsu(path, True)

# The function below takes in a path, and scans through the images to
# then return a list of the labelled images. This function is made to reduce
# repetition in nuclei and pathogen Otsu segmentation.
# arguments:
#   - path (string): The path to the directory that will have the images that are to be
#                    segmented.
# returns:
#   - images (list): The list of labelled images
# Resources used: 
#   - https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-3-otsu-thresholding/
#   - https://www.youtube.com/watch?v=qfUJHY3ku9k
def label_images_otsu(path, isCell):
    images = []
    # Obtain the file names in the directory dictated by path
    pathNames = obtain_file_names(path)
    for imagePath in pathNames:
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Before applying segmentation, check if this is a cell image. If it is, use
        # Contrast Limited Adaptive historgram equalisation (CLAHE) to improve contrast
        # and therefore improve segmentation.
        if (isCell):
            image = apply_clahe(image)
        
        ret, alteredImg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove labels that are at the edges, since they have segments outside of the field of view
        # that we cannot analyse.
        alteredImg = clear_border(alteredImg)
        
        # Place labels on segmented areas
        labelImg = measure.label(alteredImg, connectivity=image.ndim)
        
        # Measure properties of the labels, such as area, diameter, etc.
        props = measure.regionprops_table(labelImg, image,
                                          properties=['label', 'area'])
        # Move this data into a pandas dataframe.
        dframe = pd.DataFrame(props)
        
        # Remove labels that are small (these are most likely invalid)
        dframe = dframe[dframe['area'] > 100]
        print(dframe.head())
        
        plt.imshow(labelImg)
        plt.show()
        images.append((labelImg, alteredImg))
    return images

# The function below applies contrast limited adaptive histogram equalisation (CLAHE)
# and returns the edited image.
# arguments:
#   - img: The image to be edited
# returns:
#   - the edited image.
# Resources: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html 
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

# The function below takes in a path, and scans through the images to
# then return a list of the labelled images.
# arguments:
#   - path (string): The path to the directory that will have the cell images that are
#                    to be segmented.
# returns:
#   - images (list): The list of labelled cell images
# def label_images_huang(path):
#     images = []
#     # Obtain the file names in the directory dictated by path
#     pathNames = obtain_file_names(path)
#     for imagePath in pathNames:
#         image = cv2.imread(imagePath)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         cl1 = clahe.apply(image)
#         ret, alteredImg = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         plt.imshow(alteredImg)
#         plt.show()

def find_intersection(nucleiImages, cellImages):
    segj = segmentation.join_segmentations(cellImages[1], nucleiImages[1])
    plt.imshow(segj)
    plt.show()
    
    


# The function below takes in a path, and simply obtains the file names in that path.
# This is for reusability - other functions can use this function without having to write
# a try-except block for each of them.
# Arguments:
#   - path (string): The path to a directory
# Returns:
#   - fileNames (list<string>): List of file names
def obtain_file_names(path):
    pathNames = []
    try:
        for fileName in os.listdir(path):
            # Append fileName onto path to form the path to the image
            imagePath = os.path.join(path, fileName)
            pathNames.append(imagePath)
    except:
        file_exception_checking(path)
    return pathNames

# TODO
def file_exception_checking(path):
    if not (os.path.exists(path)):
        print('The directory path does not exist.')

if __name__ == '__main__':
    main()
