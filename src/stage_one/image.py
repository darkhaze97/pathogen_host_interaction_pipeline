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

from intersect import get_intersection_information
from voronoi import voronoi_seg

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
    i = 0
    # Obtain the file names in the directory dictated by path
    pathNames = obtain_file_names(path)
    for imagePath in pathNames:
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
        
        # Remove elements on the border. 
        alteredImg = clear_border(alteredImg)
        
        alteredImg = measure.label(alteredImg, connectivity=greyImage.ndim)
        
        # Remove small objects in the altered image. The minimal size will be based on if the
        # image was a pathogen, cell or nucleus. A larger minimal value is needed for
        # the cell, since there can be a lot of noise in these photos compared to the nuclei
        # or pathogens.
        alteredImg = morphology.remove_small_objects(alteredImg, 5000) if (not nucleiImages == None) \
                else morphology.remove_small_objects(alteredImg, 100)

        # origCellImg will hold the original cell image. It will be updated when we are 
        # scanning for cell images.
        origCellImg = None
        # If nucleiImages are provided, combine the labelImg with the nucleiLabel.
        # Note that nucleiImage[0] corresponds to the nucleiLabel that was accessed first.
        if (not nucleiImages == None):
            origCellImg = np.copy(alteredImg)
            alteredImg = combine_cell_nuclei_label(alteredImg, nucleiImages[i][0])

        # Then, remove holes that are within labels.
        alteredImg = morphology.remove_small_holes(alteredImg, 1000)

        # Place unique labels on segmented areas
        labelImg = measure.label(alteredImg, connectivity=greyImage.ndim)
        
        # Now, we perform more processing if we are currently analysing cell images.
        if (not nucleiImages == None):
            # A new labelled image also has to be created, as the nuclei labels are being removed.
            dim = np.shape(labelImg)
            newLabelImg = np.zeros((dim[0], dim[1]))
            
            process_cell(labelImg, origCellImg, nucleiImages[i][0], newLabelImg)
            
            labelImg = measure.label(newLabelImg)
            plt.imshow(labelImg)
            plt.show()
            i = i + 1
            
        
        # plt.imshow(labelImg)
        # plt.show()
        
        # # Scan through each label, and try to change the image! Just doing random.
        # if (not nucleiImages == None):
        #     info = measure.regionprops(labelImg)
        #     for label in info:
        #         # Get the bounds of the label.
        #         bounds = label.bbox
        #         plt.imshow(labelImg[bounds[0]:bounds[2], bounds[1]:bounds[3]])
        #         plt.show()
        #         yes = labelImg[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        #         yes = np.where(yes == 0, 10, yes + 1)
        #         plt.imshow(yes)
        #         plt.show()

        # For the pathogen and nuclei images, we do not need to access the 3rd index
        # of the tuple.
        images.append((labelImg, greyImage, imagePath, origCellImg))
    return images

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


def process_cell(labelImg, origCellImg, nucleiImage, newLabelImg):
    # First, we need to remove any nuclei only labels. Perform a regionprops
    # to obtain information about each cell label.
    cellProps = measure.regionprops(labelImg)

    # Then, scan through each region, and apply a regionprops to each cell,
    # and use the original Cell image as an intensity image to find out if the
    # current label is a cell label.
    for cell in cellProps:
        bound = cell.bbox
        # cellImg = (cell['image'] == True).astype(int)
        cellImg = np.where(cell['image'] == True, cell.label, 0)
        intensityImg = np.where(cell['image'] == True, 1, 0)
        assertCell = measure.regionprops(cellImg,
                                         intensity_image=origCellImg[bound[0]:bound[2], bound[1]:bound[3]])
        # If statement below only considers 'true' cell labels. E.g. when we combined the
        # nuclei and the cell labels, if there were any fake cell labels (nuclei) by themselves,
        # they would not have any intensity with the original cell image.
        if (assertCell[0]['intensity_mean'] == 0):
            continue
        # If we reach here, we are in a true cell label.
        # First, check if we need to separate cells that are touching each other.
        # If there is more than one nuclei in the cell, then voronoi segmentation
        # needs to be performed.
        # Take the bounding box from the nuclei images, and find the nuclei that overlap
        # with the cell.
        nucleiBox = nucleiImage[bound[0]:bound[2], bound[1]:bound[3]]
        nucleiProps = measure.regionprops(nucleiBox,
                                          intensity_image=intensityImg
                                         )
        
        # Obtain the centroids of the nuclei that have a 100% intensity
        # overlap with cellImg.
        centroidList = []
        for nucleus in nucleiProps:
            if (nucleus['intensity_mean'] == 1):
                centroidList.append(list(nucleus.centroid))
        
        # If the size of nucleiProps is not at least 1, then continue.
        if (len(centroidList) == 0):
            continue

        # Separate below into a separate voronoi function

        # Else, we have the number of nuclei in the cell. If the number of nuclei
        # exceeds 1, perform a voronoi segmentation.
        # BIG NOTE: FOR 2 NUCLEI, PRETTY SIMPLE TO DO. JUST DO A PERP
        # LINE THROUGH THE LINE BETWEEN THE POINTS
        if (len(centroidList) == 2):
            print('')
        elif (len(centroidList) > 2):
            plt.imshow(cellImg)
            plt.show()
            voronoi_seg(centroidList, np.shape(cellImg))
        
        # Add the label to the new labelled image.
        boundBox = newLabelImg[bound[0]:bound[2], bound[1]:bound[3]]
        # If a specific pixel in the newLabelImg is of label 0, we can add freely to it. However,
        # if it is not 0, this means that this region can be potentially lost. Therefore,
        # the corresponding pixels from the region have been added back in.
        newLabelImg[bound[0]:bound[2], bound[1]:bound[3]] = np.where(boundBox == 0, cellImg, boundBox)
        
        # NOTE: MAYBE SCAN THROUGH EACH CELL LABEL, THEN TAKE THE NUCLEI IMAGES
        # WITH THE SAME BOUNDING BOX, AND THEN OBTAIN THE NUCLEI
        # THAT ARE OVERLAPPING WITH THE CELL LABEL. THEN PERFORM VORONOI SEGMENTATION
        # ON THE CENTROIDS. OVERLAP REGIONS, AND THE SECTIONS OF THE CELLS IN THE DIFF
        # REGIONS CORRESPOND TO DIFF CELLS. NOTE THAT THIS SHOULD NOT RUN ON A CELL
        # WITH ONE NUCLEUS.
    

# The function below is to simply return a dictionary with the readouts for stage 1. It uses all
# information calculated from previous steps to produce this.
# Arguments:
#   info: The dictionary that contains all the measurements we have done.
# Returns:
#   A dictionary containing all of the readouts for stage 1.
def readout(info):
    # Here, info['pathogenInfo']['area']'s length will be used to determine how many vacuoles
    # there are. info['pathogenInfo']['pathogens_in_vacuole'] will be used to determine
    # the number of pathogens in each vacuole.
    # info['cellInfo']['vacuole_number'] will be used to determine how many
    # cells there are.
    # Calculate % infected cells: n(infected)/n(non-infected)
    # Use info['cellInfo']['vacuole_number']
    vacNum = sum(info['cellInfo']['vacuole_number'])
    cellNum = len(info['cellInfo']['vacuole_number'])
    patNum = sum(info['pathogenInfo']['pathogens_in_vacuole'])
    
    percentInf = len([elem for elem in info['cellInfo']['vacuole_number'] if elem > 0])\
                    /cellNum if not cellNum == 0 else 0
    # Calculate Vacuole : Cells ratio
    vacCellRat = len(info['pathogenInfo']['area'])/cellNum if not cellNum == 0 else 0
    # Calculate pathogen load.
    patLoad = patNum/cellNum if not cellNum == 0 else 0
    # Calculate infection levels
    infectLevel = {
        '0': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 0])\
                    /cellNum if not cellNum == 0 else 0,
        '1': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 1])\
                    /cellNum if not cellNum == 0 else 0,
        '2': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 2])\
                    /cellNum if not cellNum == 0 else 0,
        '3': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 3])\
                    /cellNum if not cellNum == 0 else 0,
        '4': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 4])\
                    /cellNum if not cellNum == 0 else 0,
        '5': len([elem for elem in info['cellInfo']['vacuole_number'] if elem == 5])\
                    /cellNum if not cellNum == 0 else 0,
        '5+': len([elem for elem in info['cellInfo']['vacuole_number'] if elem > 5])\
                    /cellNum if not cellNum == 0 else 0,
    }
    # Calculate mean pathogen size
    meanPatSize = sum(info['pathogenInfo']['area'])/vacNum if not vacNum == 0 else 0
    # Calculate the mean vacuole position.
    vacPosition = sum(info['pathogenInfo']['dist_nuclear_centroid'])/vacNum if not vacNum == 0\
                  else 0
    # Calculate number of vacuoles that have replicating pathogens.
    percentRep = len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem > 1])\
                    /vacNum if not vacNum == 0 else 0
    # Calculate the replication distribution. I.e. how many vacuoles have one pathogen,
    # how many vacuoles have two pathogens, etc.
    repDist = {
        '1': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem == 1])\
                    /vacNum if not vacNum == 0 else 0,
        '2': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem == 2])\
                    /vacNum if not vacNum == 0 else 0,
        '4': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem == 4])\
                    /vacNum if not vacNum == 0 else 0,
        '4+': len([elem for elem in info['pathogenInfo']['pathogens_in_vacuole'] if elem > 4])\
                    /vacNum if not vacNum == 0 else 0,
    }
    return {
        'percent_infected': percentInf,
        'vacuole_to_cell_ratio': vacCellRat,
        'pathogen_load': patLoad,
        'infection_levels': infectLevel,
        'mean_pathogen_size': meanPatSize,
        'vacuole_position': vacPosition,
        'percent_replicating_pathogens': percentRep,
        'replication_distribution': repDist
    }

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
            imagePath = path + f'/{fileName}'
            imagePath = os.path.abspath(imagePath)
            # Convert backwards slash to forward slash
            imagePath = '/'.join(imagePath.split('\\'))
            pathNames.append(imagePath)
    except:
        file_exception_checking(path)
    return pathNames

# The function is simply a function to check if there exists a path to a user-defined directory.
# If there is not one, then it will print an error.
# arguments:
#   - path (string): The path to the directory
# returns:
#   - None. Simply prints/outputs an error.
def file_exception_checking(path):
    if not (os.path.exists(path)):
        print('The directory path does not exist.')

if __name__ == '__main__':
    image_analysis("../Images/Nuclei", "../Images/Pathogen", "../Images/Cell", 10000, './temp')