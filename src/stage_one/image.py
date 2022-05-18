import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, segmentation, morphology
from skimage.segmentation import clear_border
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import pandas as pd
import cProfile

plt.style.use('fivethirtyeight')

from correction import correct_segmentation
from intersect import get_intersection_information
from readout1 import readout
from helper import obtain_file_names

# The function below is to coordinate the analysis of the images. It first labels the
# images, then finds the intersection of the pathogens with the cell labels. 
# Arguments:
#   - nucleiPath: Path to the nuclei images
#   - pathogenPath: Path to the pathogen images
#   - cellPath: Path to the cell images
#   - threshold: The custom threshold value set by the user. Can also be None
# Returns:
#   - A dictionary containing the images, as well as information about the pathogens
#     that are intersecting with the cells. There is also cell information as well.
def image_analysis(nucleiPath, pathogenPath, cellPath, threshold, savePath):
    
    # Scan through the nuclei, and correct...
    nucleiImages = label_nuclei_images(nucleiPath, threshold)
    pathogenImages = label_pathogen_images(pathogenPath, threshold)
    cellImages = label_cell_images(cellPath, nucleiImages, threshold)
    
    intersection_info = get_intersection_information(pathogenImages, cellImages, nucleiImages, savePath)
    # Generate information about whether the cells are infected, and the number of
    # pathogens in the cell. In addition, generate information on the pathogens.
    # cProfile.runctx('get_intersection_information(pathogenImages, cellImages)', {'get_intersection_information': get_intersection_information}, {'pathogenImages': pathogenImages, 'cellImages': cellImages}, filename='report.txt' )
    
    # Perform stage 1 readouts.
    readout1 = readout(intersection_info)
    print(readout1)
    
    # Now, obtain the intracellular images, and 0 pad them, to prepare them to run through
    # the CNN.
    pathogenCNN = []
    for i in range(0, len(intersection_info['pathogenInfo']['image'])):
        imageNum = intersection_info['pathogenInfo']['image'][i]
        bb = intersection_info['pathogenInfo']['bounding_box'][i]
        cropImg = (pathogenImages[imageNum][1])[bb[0]:bb[2], bb[1]:bb[3]]
        ogShape = np.shape(cropImg)
        padImg = np.zeros((100, 100))
        padImg[:ogShape[0], :ogShape[1]] = cropImg
        pathogenCNN.append(padImg)
    
    # Run the CNN here...
    
    # Return the labelled images of the nuclei, pathogen and cells. In addition, return
    # information about the intersection between the pathogens and cells.
    return {
        'nucleiImages': nucleiImages,
        'pathogenImages': pathogenImages,
        'cellImages': cellImages,
        **intersection_info,
        'readout1': readout1
    }
    

# This function is to help segment the nuclei images. It takes in a path to the
# nuclei images.
# Arguments:
#   - path (string): The path to the directory with the nuclei images.
# returns:
#   - label_images_otsu(path): A list of the labelled images.
def label_nuclei_images(path, threshold):
    return label_images_otsu(path, None, threshold)

# This function is to help segment the pathogen images. It takes in a path to the
# pathogen images.
# Arguments:
#   - path (string): The path to the directory with the pathogen images.
# returns:
#   - label_images_otsu(path): A list of the labelled images.
def label_pathogen_images(path, threshold):
    return label_images_otsu(path, None, threshold)

# This function is to help segment the cell images. It takes in a path to the
# cell images and nuclei images. The nuclei images help fill in holes in the cell.
# Note that for this case, the nuclei segmentation must have occurred first.
# Arguments:
#   - path (string): The path to the directory with the cell images.
#   - nucleiImages list<(labelled images (list), images (list))>: The path to the
#                                                                 directory with the
#                                                                 nuclei images.
# returns:
#   - label_images_huang(path): A list of the labelled cell images.
def label_cell_images(path, nucleiImages, threshold):
    return label_images_otsu(path, nucleiImages, threshold)

# The function below takes in a path, and scans through the images to
# then return a list of the labelled images. This function is made to reduce
# repetition in nuclei and pathogen Otsu segmentation.
# arguments:
#   - path (string): The path to the directory that will have the images that are to be
#                    segmented.
# returns:
#   - list<(labelled images (list), images (list), imagePath (string))>: A list with tuples of labelled
#                                                        images, and segmented images
# Resources used: 
#   - https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-3-otsu-thresholding/
#   - https://www.youtube.com/watch?v=qfUJHY3ku9k
def label_images_otsu(path, nucleiImages, threshold):
    images = []
    # i is the index to calculate which nucleiImage we are up to (if we are currently
    # attempting to label the cell images.)
    # Obtain the file names in the directory dictated by path
    pathNames = obtain_file_names(path)
    for i, imagePath in enumerate(pathNames):
        image = cv2.imread(imagePath)
        greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Before applying segmentation, check if this is a cell image. If it is, use
        # Contrast Limited Adaptive historgram equalisation (CLAHE) to improve contrast
        # and therefore improve segmentation.
        if (not nucleiImages == None):
            greyImage = apply_clahe(greyImage)

        ret, alteredImg = cv2.threshold(greyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) \
                          if (threshold == None) else \
                          cv2.threshold(src=greyImage, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Cleanup the image.
        alteredImg = segment_cleanup(alteredImg, greyImage, not nucleiImages == None)

        # Place unique labels on segmented areas
        labelImg = measure.label(alteredImg, connectivity=greyImage.ndim) if (nucleiImages == None) \
                   else process_cell(alteredImg, nucleiImages[i][0], greyImage)
            
        # plt.imshow(labelImg)
        # plt.show()

        # For the pathogen and nuclei images, we do not need to access the 3rd index
        # of the tuple.
        images.append((labelImg, greyImage, imagePath))
    return images

# The function below cleans up the segmentation. It removes all labels that are on the
# border, removes labelled regions that are too small, and removes small holes
# within a label. 
# Arguments:
#   - alteredImg (numpy array): The image to clean up
#   - greyImage (numpy array): An image to help form the connectivity of the labelled image.
#   - isCell (boolean): Boolean for whether the image is of a cell or not.
# Returns:
#   - alteredImg (numpy array): The cleaned up image.
def segment_cleanup(alteredImg, greyImage, isCell):
    # Remove elements on the border. 
    alteredImg = clear_border(alteredImg)
    
    # We label the image here, so that we can correctly use
    # remove_small_objects and remove_small_holes
    alteredImg = measure.label(alteredImg, connectivity=greyImage.ndim)
    
    # Remove small objects in the altered image. The minimal size will be based on if the
    # image was a pathogen, cell or nucleus. A larger minimal value is needed for
    # the cell, since there can be a lot of noise in these photos compared to the nuclei
    # or pathogens.
    alteredImg = morphology.remove_small_objects(alteredImg, 100) if (not isCell) else \
                 morphology.remove_small_objects(alteredImg, 5000)

    # Then, remove holes that are within labels.
    alteredImg = morphology.remove_small_holes(alteredImg, 1000)
    
    return alteredImg

# The function below simply combines the labelling of the nuclei and the cells.
# This is to fill in the 'hole' in cell images, where the nuclei is not visible.
# Arguments:
#   - cellLabel: The labelled image of the cell
#   - nucleiLabel: The labelled image of the nuclei
# Returns:
#   - A numpy array with values 0 representing the background, and 1 representing the
#     foreground.
def combine_cell_nuclei_label(cellLabel, nucleiLabel):
    # =================== TODO =====================
    # After combining, only accept nuclei that are adjacent to the cell labels
    relabelNuclei = (nucleiLabel > 0).astype(int)
    cellLabel = (cellLabel > 0).astype(int)
    # relabelCell = (labelImg == True).astype(int)
    # Join the labels of the cells and nuclei.
    joinedLabels = segmentation.join_segmentations(relabelNuclei, cellLabel)
    
    # Make the background (0) and other labels be (1) again, and return.
    return (joinedLabels > 0).astype(int)

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

# This function is to process cell images. It performs pre-processing steps, to allow
# correct_segmentation to be called. It will return the new cell image that has been
# processed. For more information, read the information for correct_segmentation.
# Arguments:
#   - alteredImg (numpy array): The cell image.
#   - nucleiImage (numpy array): Used to combine the cell and nuclei labels, to fill in 
#                                holes in the cell labels.
#   - greyImage (numpy array): An image to help form the connectivity when labelling alteredImg
# Returns:
#   - A labelled image of the edited cell image.
def process_cell(alteredImg, nucleiImage, greyImage):
    origCellImg = np.copy(alteredImg)
    alteredImg = combine_cell_nuclei_label(alteredImg, nucleiImage)
    alteredImg = measure.label(alteredImg, connectivity=greyImage.ndim)
    
    # A new labelled image also has to be created, as the nuclei labels are being removed.
    dim = np.shape(alteredImg)
    newLabelImg = np.zeros((dim[0], dim[1]))
    
    correct_segmentation(alteredImg, origCellImg, nucleiImage, newLabelImg)
    
    return measure.label(newLabelImg)

if __name__ == '__main__':
    image_analysis("../Images/Nuclei", "../Images/Pathogen", "../Images/Cell", 10000, './temp')