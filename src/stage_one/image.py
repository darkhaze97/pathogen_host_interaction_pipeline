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
from readout1 import readout
from helper import obtain_file_names, filter_zero_mean_intensity

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

# This function performs the brunt of the cell image correction. It removes nuclei only labels,
# which are a by-product of combining cell and nuclei labels. In addition, it separates
# cells that are touching each other, by calling functions to perform voronoi segmentation.
# Arguments:
#   - labelImg (numpy array): The cell image that has combined with nuclei.
#   - origCellImg (numpy array): The original cell image that has not combined with nucleiImages yet.
#                                This will be used to remove nuclei only labels.
#   - nucleiImage (numpy array): The nuclei images. This will be used to remove nuclei only
#                                labels.
#   - newLabelImg (numpy array): This will contain the edited cell image, and will be the 'return'
# Returns:
#   - None, however, newLabelImg will be the return by reference.
def correct_segmentation(labelImg, origCellImg, nucleiImage, newLabelImg):
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
            segmentedImg = voronoi_seg(centroidList, cellImg)
            correct_voronoi(segmentedImg, nucleiBox, cellImg)
                
        # Add the label to the new labelled image.
        boundBox = newLabelImg[bound[0]:bound[2], bound[1]:bound[3]]
        # If a specific pixel in the newLabelImg is of label 0, we can add the cell label
        # bounded by bound[0]:bound[2] and bound[1]:bound[3] freely to it. However,
        # if there is a pixel in this box that is not originally 0 and is not the label
        # of the current cell being analysed, this means that this 
        # region can be potentially lost. Therefore, the corresponding pixels from the 
        # region have been added back in.
        newLabelImg[bound[0]:bound[2], bound[1]:bound[3]] = np.where(boundBox == 0, cellImg, boundBox)
        
        # NOTE: MAYBE SCAN THROUGH EACH CELL LABEL, THEN TAKE THE NUCLEI IMAGES
        # WITH THE SAME BOUNDING BOX, AND THEN OBTAIN THE NUCLEI
        # THAT ARE OVERLAPPING WITH THE CELL LABEL. THEN PERFORM VORONOI SEGMENTATION
        # ON THE CENTROIDS. OVERLAP REGIONS, AND THE SECTIONS OF THE CELLS IN THE DIFF
        # REGIONS CORRESPOND TO DIFF CELLS. NOTE THAT THIS SHOULD NOT RUN ON A CELL
        # WITH ONE NUCLEUS.

def correct_voronoi(segmentedImg, nucleiBox, cellImg):
    # Then, if there are regions with the same label that are not adjacent,
    # try to find the region it was originally connected to.
    # First, distinguish the regions that should not be part of the same label
    # i.e. labels that are not connected to the nucleus.
    # Then, connect these regions to any other voronoi region, by obtaining
    # their regions in a numpy array. Also collect the label '-1', as this
    # is the voronoi ridge separating the regions.
    # Then set all of these regions to the same label, and use measure.label
    # once again. If the number of resulting regions is 1, this indicates
    # that the regions are connecting.
    
    # Now, correct the voronoi segmentation. There will be some labels that
    # have regions that are not connected spatially. Therefore, the correction
    # will attempt to find the 'correct' connections for these regions.
    
    # First, obtain the regions that are not meant to be in this voronoi region.
    # i.e. regions that are not connected to the nuclei.
    # First, for each voronoi region, create an intensity image. Then, the
    # corresponding nucleus for this voronoi region was obtained.

    # Obtain nuclei that are completely within a cell... Obtain the nucleus with the
    # highest intensity.
    #   - Obtain the intensity image of segmentedImg
    # Then obtain the regions that do not include this nuclus
    #   - Done by: Removing the region that contains this nuclei (> 0 intensity)
    newImg = np.zeros(segmentedImg.shape)
    # Extract parts of the voronoi ridges that overlap with the cells. This will be used
    # later to rejoin labels from different voronoi regions.
    vorRidgeOverlap = measure.label(np.where(segmentedImg == -1, cellImg, 0))
    # Assign each ridge in the segmentedImg with the label of an adjacent region.
    plt.imshow(vorRidgeOverlap)
    plt.show()
    vorRegions = measure.regionprops(segmentedImg)
    for r in vorRegions:
        # Below obtains the nucleus for this voronoi region. It will then be used
        # as an intensity image to determine the regions that are not connected to this
        # nucleus, to find the correct region that they are connected to.
        intensityImage = (segmentedImg == r['label']).astype(int)
        nucleiProps = measure.regionprops(nucleiBox,
                                  intensity_image=intensityImage
                                 )
        # Obtain the nucleus with the highest intensity.
        n = max(nucleiProps, key=lambda n:n.intensity_mean)
        
        # rLabel is the labelling of the region defined by r.
        # It will differentially label regions that are not connected.
        # The shape of rLabel is segmentedImg.shape, so that we can record the exact
        # positions of each label, and connect with other labels correctly.
        rLabel = np.zeros(segmentedImg.shape)
        rLabel[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]] = r.image
        rLabel = measure.label(rLabel)
        
        # intensityImage will contain an image of the nucleus for the region defined
        # by r. This will be used to extract out the regions that are not connected to this
        # nucleus.
        # Once again, we use nucleiBox.shape, so that we can use this as an intensity image
        # with rLabel, noting that nucleiBox and segmentedImg have the same shape.
        intensityImage = np.zeros(nucleiBox.shape)
        intensityImage[n.bbox[0]:n.bbox[2], n.bbox[1]:n.bbox[3]] = n.image

        # The section below simply filters out the labelled region that contains the
        # nucleus. The regions that are not connected to the cell (and therefore,
        # not connected to the nucleus) remain. These regions are connected to a label
        # in another voronoi region.
        cellProps = measure.regionprops(rLabel, intensity_image=intensityImage)
        cellProps = list(filter(filter_zero_mean_intensity, cellProps))
        
        # Now, connect each of cellProps to a different labelled region, and
        # include the -1 label (voronoi ridges).
        # First, obtain the -1 regions that overlap with the original cell image.
        # Then label each of these -1 regions again. (DONE)
        # Then, insert the two labelled regions into the image with size
        # nucleiBox.shape or segmentedImg.shape, and attempt to add
        # each labelled -1 region to the image, and label. If the label results in
        # one single region, then the region has been reformed. Label the region with
        # the label of the other region in the other voronoi region in segmentedImg.
        
        
        # Simply 'remove' this section of the voronoi ridge from segmentedImg, by filling in
        # where the ridge was before. To do this, simply take the connected regions, and 
        # colour over the section of the ridge between these connected regions (maybe
        # use something like np.where. This is so that we don't remove the entire ridge, in
        # case the ridge also separates two cells)
        # Also remove this section of the ridge from vorRidgeOverlap, and 
        # use measure.label the sections of ridges again.
        # Then after all has been completed, relabel the entire segmentedImg
        
        
        
        

        

if __name__ == '__main__':
    image_analysis("../Images/Nuclei", "../Images/Pathogen", "../Images/Cell", 10000, './temp')