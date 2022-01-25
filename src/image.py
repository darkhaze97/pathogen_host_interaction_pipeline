import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, segmentation, morphology, future
from skimage.segmentation import clear_border
import pandas as pd
plt.style.use('fivethirtyeight')


# The function below is to coordinate the analysis of the images. It first labels the
# images, then finds the intersection of the pathogens with the cell labels. 
# Arguments:
#   - nucleiPath: Path to the nuclei images
#   - pathogenPath: Path to the pathogen images
#   - cellPath: Path to the cell images
# Returns:
#   ==========TODO===========
def image_analysis(nucleiPath, pathogenPath, cellPath):
    
    # Scan through the nuclei, and correct...
    nucleiImages = label_nuclei_images(nucleiPath)
    pathogenImages = label_pathogen_images(pathogenPath)
    cellImages = label_cell_images(cellPath)
    
    # Generate information about whether the cells are infected, and the number of
    # pathogens in the cell. In addition, generate information on the pathogens.
    intersection_info = get_intersection_information(pathogenImages, cellImages)
    
    # Return the labelled images of the nuclei, pathogen and cells. In addition, return
    # information about the intersection between the pathogens and cells.
    return {
        'nucleiImages': nucleiImages,
        'pathogenImages': pathogenImages,
        'cellImages': cellImages,
        **intersection_info
    }
    

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
#   - list<(labelled images (list), images (list))>: A list with tuples of labelled
#                                                        images, and segmented images
# Resources used: 
#   - https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-3-otsu-thresholding/
#   - https://www.youtube.com/watch?v=qfUJHY3ku9k
def label_images_otsu(path, isCell):
    images = []
    # Obtain the file names in the directory dictated by path
    pathNames = obtain_file_names(path)
    for imagePath in pathNames:
        image = cv2.imread(imagePath)
        greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Before applying segmentation, check if this is a cell image. If it is, use
        # Contrast Limited Adaptive historgram equalisation (CLAHE) to improve contrast
        # and therefore improve segmentation.
        if (isCell):
            greyImage = apply_clahe(greyImage)

        ret, alteredImg = cv2.threshold(greyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove labels that are at the edges, since they have segments outside of the field of view
        # that we cannot analyse.
        alteredImg = clear_border(alteredImg)
        
        # Remove small objects in the altered image. The minimal size will be based on if the
        # image was a pathogen, cell or nucleus. A larger minimal value is needed for
        # the cell, since there can be a lot of noise in these photos compared to the nuclei
        # or pathogens.
        labelImg = alteredImg > 0
        labelImg = morphology.remove_small_objects(labelImg, 2000) if isCell \
                        else morphology.remove_small_objects(labelImg, 100)
        
        # Then, remove holes that are within labels.
        labelImg = morphology.remove_small_holes(labelImg, 1000000)
        
        # Place labels on segmented areas
        labelImg = measure.label(labelImg, connectivity=greyImage.ndim) 

        # plt.imshow(labelImg)
        # plt.show()
        images.append((labelImg, image))
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

# The function below takes in tuples of pathogen images and cell images (These tuples are generated
# by label_images_otsu). Overall, it filters out extracellular pathogens, and calculates
# the number of cells that are infected and not infected.
# arguments:
#   - pathogenImages: A tuple of the labelled image, and the black and white image of the pathogen
#   - cellImages: A tuple of the labelled image, and the black and white image of the cell
# returns:
#   - A dictionary which has the keys 'pathogenInfo' and 'cellInfo'. These map to
#     values about pathogens and cells. Note that cellInfo['area'][2] and
#     cellInfo['pathogen_number'][2] are referring to the same cell.
def get_intersection_information(pathogenImages, cellImages):
    # Relabel all labels to 0 (background) or 1 in both the cell and pathogen images.
    labelledPathogen = []
    labelledCell = []
    for pathogen in pathogenImages:
        newLabelPathogen = (pathogen[0] > 0).astype(int)
        labelledPathogen.append(newLabelPathogen)
    for cell in cellImages:
        newLabelCell = (cell[0] > 0).astype(int)
        labelledCell.append(newLabelCell)
    
    # Join the labels for each corresponding index. I.e. join labelledPathogen[1]
    # with labelledCell[1]. Then, obtain the intracellular pathogens only. The intracellular
    # pathogens will be labelled 3, since it overlaps with cell labels.
    intracellularPathogens = []
    for i in range(0, len(labelledPathogen)):
        joinedLabels = segmentation.join_segmentations(labelledPathogen[i], labelledCell[i])
        
        filtered = (joinedLabels == 3).astype(int)
        intracellularPathogens.append(filtered)

    # Join intracellular pathogens with the original pathogen labels, and then find regions with
    # mean_intensity == 1.0 (i.e. 100% overlap). Record the labels that have mean intensity
    # 1.0 in an array, and filter these into a new image.
    for i in range(0, len(labelledCell)):
        props = measure.regionprops(pathogenImages[i][0], intensity_image=intracellularPathogens[i])
        fullyIntracellular = []
        for prop in props:
            if (prop['intensity_mean'] == 1.0):
                fullyIntracellular.append(prop['label'])
        # Now, filter out the fully intracellular labels from pathogenImages[i][0], and
        # then convert each image to having only two labels; the background (0), and
        # pathogen labels (1)
        truthTable = (np.in1d(pathogenImages[i][0], fullyIntracellular)).reshape(
                        len(pathogenImages[i][0]),
                        len(pathogenImages[i][0][0])
                     )
        intracellularPathogens[i] = (truthTable == True).astype(int)

    # Now, rejoin the intracellularPathogens with the corresponding cell labels, so that
    # we can begin counting the number of infected cells and uninfected cells.
    cellPathogenLabels = []
    for i in range(0, len(intracellularPathogens)):
        joinedLabels = segmentation.join_segmentations(intracellularPathogens[i], labelledCell[i])
        cellPathogenLabel = measure.label(joinedLabels)
        cellPathogenLabels.append(cellPathogenLabel)
        # plt.imshow(cellPathogenLabels[i])
        # plt.show()

    pathogenInfo = {'bounding_box': [], 'area': [], 'image': []}
    cellInfo = {'bounding_box': [], 'area': [], 'image': [], 'pathogen_number': []}

    # Analyse the properties of each label using regionprops
    for i in range(0, len(cellPathogenLabels)):
        # We will obtain the region properties, and then scan through these
        props = measure.regionprops(cellPathogenLabels[i])

        # Form a region adjacency graph (from the original cellPathogenLabels[i] image),
        # then scan through each label. Place each label in the adjacency graph, and check
        # their neighbours. 
        # If there are multiple neighbours (i.e. more than one), then
        # we either have a pathogen or an infected cell. This is because the pathogen
        # may be on the edge of the cell label, and is touching both the cell label and
        # the background.
        #       If current label's area is bigger than all the adjacent labels, then the current label is an
        #       infected cell.
        #       Else, we have a pathogen.
        # Else, if there is only one neighbour, then
        #       If the neighbour is 0, then the current label is a non-infected cell
        #       Else, if the neighbour is not 0, then we have an intracellular pathogen
        rag = future.graph.RAG(cellPathogenLabels[i])

        for label in props:
            # Scan through each label, and obtain the adjacent labels
            adjacency = list(rag.neighbors(label.label))
            if (len(adjacency) > 1):
                adjacency.remove(0)
                currentLabelArea = label.area
                # Now, scan through the list of adjacent labels, and compare their areas.
                isCell = True
                for adjacentLabel in adjacency:
                    # If an adjacent label has a greater area, then this is a pathogen
                    if (props[adjacentLabel - 1].area > currentLabelArea):
                        isCell = False
                if (isCell):
                    # If this is a cell, then record the number of pathogens within it as
                    # well
                    extract_cell_info(cellInfo, cellImages[i][1], label, len(adjacency))
                else:
                    extract_pathogen_info(pathogenInfo, pathogenImages[i][1], label)
            elif (len(adjacency) == 1):
                if (adjacency[0] == 0):
                    extract_cell_info(cellInfo, cellImages[i][1], label, 0)
                else:
                    extract_pathogen_info(pathogenInfo, pathogenImages[i][1], label)
    return {'pathogenInfo': pathogenInfo, 'cellInfo': cellInfo}

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

# The function is simply a function to check if there exists a path to a user-defined directory.
# If there is not one, then it will print an error.
# arguments:
#   - path (string): The path to the directory
# returns:
#   - None. Simply prints/outputs an error.
def file_exception_checking(path):
    if not (os.path.exists(path)):
        print('The directory path does not exist.')

# Extracts information about a specific pathogen.
# Arguments:
#   - pathogenInfo: The dictionary that holds all pathogen information (i.e. like a 
#                   'global' data)
#   - ogImage: Original image that the pathogen comes from
#   - regionInfo: Information about the pathogen that will be added to pathogenInfo
def extract_pathogen_info(pathogenInfo, ogImage, regionInfo):
    pathogenInfo['bounding_box'].append(regionInfo.bbox)
    pathogenInfo['area'].append(regionInfo.area)
    pathogenInfo['image'].append(ogImage)

# Extracts information about a specific cell.
# Arguments:
#   - cellInfo: The dictionary that holds all cell information (i.e. like a 'global' data)
#   - ogImage: Original image that the cell comes from
#   - regionInfo: Information about the cell that will be added to cellInfo
#   - pathogenNum: The number of pathogens in the cell
def extract_cell_info(cellInfo, ogImage, regionInfo, pathogenNum):
    cellInfo['bounding_box'].append(regionInfo.bbox)
    cellInfo['area'].append(regionInfo.area)
    cellInfo['image'].append(ogImage)
    cellInfo['pathogen_number'].append(pathogenNum)
