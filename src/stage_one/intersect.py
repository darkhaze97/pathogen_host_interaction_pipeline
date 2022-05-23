from collections import defaultdict
import math
import os
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import numpy as np
import imagej
import csv
ij = imagej.init()

from decision_tree import predict
from helper import filter_one_hundred_mean_intensity

# The function below takes in tuples of pathogen images and cell images (These tuples are generated
# by label_images_otsu). Overall, it filters out extracellular pathogens, and calculates
# the number of cells that are infected and not infected.
# arguments:
#   - pathogenImages: A tuple of the labelled image, and the black and white image of the pathogen
#   - cellImages: A tuple of the labelled image, and the black and white image of the cell
# returns:
#   - A dictionary which has the keys 'pathogenInfo' and 'cellInfo'. These map to
#     values about pathogens and cells. Note that cellInfo['area'][2] and
#     cellInfo['vacuole_number'][2] are referring to the same cell.
def get_intersection_information(pathogenImages, cellImages, nucleiImages, savePath):
    # Relabel all labels to 0 (background) or 1 in both the cell and pathogen images.
    labelledCell = [(cell[0] > 0).astype(int) for cell in cellImages]
    labelledPathogen = [(path[0] > 0).astype(int) for path in pathogenImages]
    
    intracellularPathogens = obtain_intracellular_pathogens(labelledPathogen, labelledCell, pathogenImages)

    pathogenInfo = defaultdict(list)
    cellInfo = defaultdict(list)

    # Use regionprops_table on the cell labels by supplying the fully intracellular pathogens as an
    # intensity image. Then apply regionprops_table to find the intensity_mean, where 
    # intensity_mean > 0 indicates an infected cell, and intensity_mean == 0 indicates a
    # non-infected cell. In addition, match nuclei with each infected cell, to find the
    # dist_nuclear_centroid for the pathogens.
    # Scan through every intracellular pathogen image, and apply regionprops_table to each
    # corresponding image.
    for i in range(len(intracellularPathogens)):
        cellProps = measure.regionprops(cellImages[i][0], intensity_image=intracellularPathogens[i])
        # Scan through each prop generated, and filter through the intensity_mean.
        for cell in cellProps:
            # Obtain the bound of the cell. This will be used later to extract pixels from an
            # image.
            bound = cell.bbox

            # If intensity_mean == 0, this means that no pathogen is inside this cell.
            if (cell['intensity_mean'] == 0):
                extract_cell_info(cellInfo, i, cell, 0)
            else:
                # Obtain the bounding box using label.bbox around the fully intracellular pathogens
                # and the nuclei images to attempt to find the nucleus matching the corresponding
                # cell.
                nucleiBoundLabels = nucleiImages[i][0][bound[0]:bound[2], bound[1]: bound[3]]
                # Find the nucleus with 100% intensity_mean when using the cell label as an intensity
                # image.
                nucleiProps = measure.regionprops(nucleiBoundLabels, intensity_image=cell.image)
                # The size of nucleiProps should always be of at size 1, as we have accounted
                # for any multinuclear cell label in image.py.
                nuclearCentroid = nucleiProps[0]['centroid']
                
                # Below is to find how many pathogens are in the cell label.
                intracellularBoundLabels = intracellularPathogens[i][bound[0]:bound[2], bound[1]:bound[3]]
                pathogenProps = measure.regionprops(intracellularBoundLabels, intensity_image=cell.image)
                # Below is a filter to remove all pathogens that have a less than 100%
                # intensity_mean. (<100% indicates the pathogen is not within the cell.
                # This can occur if the pathogen is in another cell, but is in the same bounding
                # box.)
                pathogenProps = list(filter(filter_one_hundred_mean_intensity, pathogenProps))
                # Create the cell info, by using the length of pathogenProps as the number
                # of infected cells within this cell.
                extract_cell_info(cellInfo, i, cell, len(pathogenProps))
                # Next, extract information about each pathogen.
                for pathogen in pathogenProps:
                    # Obtain fluorescence data first. This will be used for the
                    # decision tree later.
                    measure_fluorescence(pathogenInfo, pathogenImages[i], bound, savePath)
                    extract_pathogen_info(pathogenInfo, i, pathogen, nuclearCentroid)

    if (os.path.exists(savePath + '.csv')):
        os.remove(savePath + '.csv')
    # Run the decision tree to determine the number of pathogens in each parasitophorous
    # vacuole. We only run the decision tree if there are valid pathogens.
    prepare_decision_tree(pathogenInfo)
    
    return {'pathogenInfo': pathogenInfo, 'cellInfo': cellInfo}

# This function obtains the pathogens that are completely intracellular in a cell. 
# It joins pathogenImages with cellImages, and performs other processing steps to 
# eventually obtain the image of fully intracellular pathogens.
# Arguments:
#   - labelledPathogen (list<numpy array>): A list of the labelled images of the pathogens.
#   - labelledCell (list<numpy array>): A list of the labelled images of the cells.
#   - pathogenImages (list<(labelled images (list), images (list), imagePath (string)))>: A list
#                                                 containing the original labelled pathogen images.
# Returns:
#   - intracellularPathogens (list<numpy array>): A list of the labelled images of the fully
#                                                 intracellular pathogens.
def obtain_intracellular_pathogens(labelledPathogen, labelledCell, pathogenImages):
    # Join the labels for each image in labelledPathogen and labelledCell. Then,
    # obtain the intracellular pathogens only. The intracellular pathogens will be
    # labelled 3, since it overlaps with the cell labels.
    joinedLabels = [segmentation.join_segmentations(p, c) for p, c in zip(labelledPathogen, labelledCell)]
    intracellularPathogens = [(j == 3).astype(int) for j in joinedLabels]

    # Now, we need to find the pathogens that are completely intracellular, and not
    # slightly inside a cell whilst being slightly outside of the cell at the same time.
    # For every single pathogenImage, use the corresponding intracellularPathogen image
    # as an intensity image. Then, use the intensity to determine the pathogens
    # that are completely intracellular.
    pathPropAllImg = [measure.regionprops_table(pathIm[0], intensity_image=intraPath,
                                          properties=['label', 'intensity_mean'])\
                      for pathIm, intraPath in zip(pathogenImages, intracellularPathogens)]
    # Scan through elements in pathPropAllImg, and for each element, obtain the labels that have
    # an intensity_mean of 1.0. In this step, we have now obtained the labels of the pathogens
    # that are completely intracellular.
    fullyIntracellularLabels = [[pathLab for pathLab, pathInt in zip(pathPropImg['label'], pathPropImg['intensity_mean']) \
                                   if (pathInt == 1.0)] for pathPropImg in pathPropAllImg]
    
    # Below is to filter out the fully intracellular pathogens from the original labelled
    # image that we have.
    fullyIntracellularAllImg = [(np.in1d(pathIm[0], fullyIntracellularLabel)).reshape(
                                len(pathIm[0]),
                                len(pathIm[0][0])
                                )
                                for pathIm, fullyIntracellularLabel in zip(pathogenImages, fullyIntracellularLabels)]
    # Below simply changes fullyIntracellularAllImg to 0 (background) or 1 (a region)
    intracellularPathogens = [(fullyIntracellularImg == True).astype(int) \
                              for fullyIntracellularImg in fullyIntracellularAllImg]
    return intracellularPathogens

# This function measures the fluorescence of a pathogen. We run this information through
# a decision tree, to help predict the number of pathogens in a vacuole.
# Arguments:
#   - pathogenInfo (dict): The pathogenInfo dictionary. We use it here to update it.
#   - image (list<(labelled images (list), images (list), imagePath (string))>): This is used to 
#                                      obtain the path to the image, for loading in imageJ.
#   - bound (tuple): Tuple containing the bounding box of the pathogen.
#   - savePath (string): The save file name for the csv that will be created from this measurement.
#                        This file will be deleted after it is read.
def measure_fluorescence(pathogenInfo, image, bound, savePath):
    # Measure the fluorescence readings, and prepare to add to the dictionary
    # that stores information about the entity.
    # We form rectangles around the original image, and then check the
    # fluorescence readings.
    xChange = abs(bound[3] - bound[1])
    yChange = abs(bound[2] - bound[0])
    macro = f"""
        open('{image[2]}');
        makeRectangle({bound[1]}, {bound[0]}, {xChange}, {yChange})
        run("Set Measurements...", "mean min redirect=None decimal=3");
        run("Measure");
        saveAs("Results", "{savePath}.csv");
    """
    ij.py.run_macro(macro)
    with open(savePath + '.csv') as f:
        csvread = csv.reader(f)
        # Obtain the header from the csv.
        header = []
        header = next(csvread)
        # Obtain the next row from the csv.
        row = next(csvread)
        for i in range(len(header)):
            if (header[i] == ' '):
                continue
            if (header[i] not in pathogenInfo):
            #Create a new value for header[j].
                pathogenInfo[header[i]] = []
            pathogenInfo[header[i]].append(row[i]) 

# Extracts information about a specific pathogen.
# Arguments:
#   - pathogenInfo: The dictionary that holds all pathogen information (i.e. like a 
#                   'global' data)
#   - ogImage: Original image that the pathogen comes from
#   - regionInfo: Information about the pathogen that will be added to pathogenInfo
#   - nuclearCentroid: The centroid for the nucleus that the pathogen is in.
def extract_pathogen_info(pathogenInfo, imageNum, regionInfo, nuclearCentroid):
    pathogenInfo['bounding_box'].append(regionInfo.bbox)
    pathogenInfo['area'].append(regionInfo.area)
    pathogenInfo['perimeter'].append(regionInfo.perimeter)
    pathogenInfo['image'].append(imageNum)
    pathogenInfo['diameter'].append(regionInfo.equivalent_diameter_area)
    pathogenInfo['circularity'].append(4 * math.pi * regionInfo.area/(regionInfo.perimeter ** 2))
    dist = math.sqrt((nuclearCentroid[0] - regionInfo.centroid[0]) ** 2 +\
                     (nuclearCentroid[1] - regionInfo.centroid[1]) ** 2)
    pathogenInfo['dist_nuclear_centroid'].append(dist)

# Extracts information about a specific cell.
# Arguments:
#   - cellInfo: The dictionary that holds all cell information (i.e. like a 'global' data)
#   - ogImage: Original image that the cell comes from
#   - regionInfo: Information about the cell that will be added to cellInfo
#   - pathogenNum: The number of pathogens in the cell
def extract_cell_info(cellInfo, imageNum, regionInfo, pathogenNum):
    cellInfo['bounding_box'].append(regionInfo.bbox)
    cellInfo['area'].append(regionInfo.area)
    cellInfo['image'].append(imageNum)
    cellInfo['perimeter'].append(regionInfo.perimeter)
    cellInfo['vacuole_number'].append(pathogenNum)
    cellInfo['diameter'].append(regionInfo.equivalent_diameter_area)
    cellInfo['circularity'].append(4 * math.pi * regionInfo.area/(regionInfo.perimeter ** 2))

# The function below simply prepares information to run through the decision tree.
# It will then obtain the predictions, and place it into the pathogenInfo dictionary.
# Arguments:
#   pathogenInfo: The dictionary that contains information about the pathogens
def prepare_decision_tree(pathogenInfo):
    arr = np.array([pathogenInfo['area']], dtype=np.uint32)
    arr = np.append(arr, [pathogenInfo['circularity']], axis=0)
    arr = np.append(arr, [pathogenInfo['perimeter']], axis=0)
    arr = np.append(arr, [pathogenInfo['diameter']], axis=0)
    arr = np.append(arr, [pathogenInfo['Max']], axis=0)
    arr = np.append(arr, [pathogenInfo['Mean']], axis=0)
    arr = np.transpose(arr)
    pred = predict(arr)
    # Convert predictions from floats into ints
    pred = list(map(int, pred))
    pathogenInfo['pathogens_in_vacuole'] = pred
