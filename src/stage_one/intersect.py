import math
import os
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import numpy as np
import imagej
import csv
ij = imagej.init()

from .decision_tree import predict

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
        # We use regionprops_table instead of regionprops, to reduce the number of properties
        # to analyse and make a small performance boost.
        props = measure.regionprops_table(pathogenImages[i][0], intensity_image=intracellularPathogens[i],
                                          properties=['label', 'intensity_mean'])
        fullyIntracellular = []
        for j in range(0, len(props['label'])):
            if (props['intensity_mean'][j] == 1.0):
                fullyIntracellular.append(props['label'][j])
        # Now, filter out the fully intracellular labels from pathogenImages[i][0], and
        # then convert each image to having only two labels; the background (0), and
        # pathogen labels (1)
        truthTable = (np.in1d(pathogenImages[i][0], fullyIntracellular)).reshape(
                        len(pathogenImages[i][0]),
                        len(pathogenImages[i][0][0])
                     )
        intracellularPathogens[i] = (truthTable == True).astype(int)

    pathogenInfo = {'bounding_box': [], 'area': [], 'image': [], 'perimeter': [],
                    'diameter': [], 'circularity': [], 'dist_nuclear_centroid': []}
    cellInfo = {'bounding_box': [], 'area': [], 'image': [], 'vacuole_number': [], 'perimeter': [],
                'diameter': [], 'circularity': []}

    # Use regionprops_table on the cell labels by supplying the fully intracellular pathogens as an
    # intensity image. Then apply regionprops_table to find the intensity_mean, where 
    # intensity_mean > 0 indicates an infected cell, and intensity_mean == 0 indicates a
    # non-infected cell. In addition, match nuclei with each infected cell, to find the
    # dist_nuclear_centroid for the pathogens.
    # Scan through every intracellular pathogen image, and apply regionprops_table to each
    # corresponding image.
    for i in range(0, len(intracellularPathogens)):
        cellProps = measure.regionprops(cellImages[i][0], intensity_image=intracellularPathogens[i])
        # Scan through each prop generated, and filter through the intensity_mean.
        for cell in cellProps:
            # Apply a regionprops to each cell separately again, and use the original Cell image
            # as an intensity image to find out if the current label is a cell label.
            bound = cell.bbox
            # Convert cell['image'] into an integer array.
            cellImg = (cell['image'] == True).astype(int)
            assertCell = measure.regionprops(cellImg,
                                             intensity_image=cellImages[i][3][bound[0]:bound[2], bound[1]:bound[3]])
            # If statement below only considers 'true' cell labels. E.g. when we combined
            # the nuclei and the cell labels, if there were any nuclei labels by themselves
            # they will not be considered.
            if (assertCell[0]['intensity_mean'] == 0):
                continue
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
                nucleiProps = list(filter(filter_one_hundred_mean_intensity, nucleiProps))
                # If the nucleiProps length is not one, it does not matter, always choose the first
                # element in nucleiProps. The size of nucleiProps should always be of at least size
                # 1.
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
    # vacuole.
    prepare_decision_tree(pathogenInfo)
    
    return {'pathogenInfo': pathogenInfo, 'cellInfo': cellInfo}

# The function below is to simply scan the immediate neighbours 
# around a pixel (all 8 pixels around). If a pixel is from a different label,
# then determine if it needs to be added to the neighbours list, and then repeat.
# It is worst case O((p + 1)log(p + 1)). The reason why I implement binary search and
# sort with this is because adding onto the neighbours list occurs rarely (only at the
# first encounter of a different label), and the algorithm will mainly be performing
# a binary search for these labels instead. Therefore, on average, this search algorithm
# will be O(logp), rather than O(p) (when scanning through a list linearly).
# Arguments:
#   - labelImage: The labelledImage that we are analysing
#   - row: The current row we have scanned to
#   - column: The current column we have scanned to
#   - label: The current label we are considering neighbours for.
#   - neighbours: The list that contains the neighbours for the current label.
def scan_neighbours(labelImage, row, column, label, neighbours):
    # We will create variables that indicate whether or not we can scan one pixel
    # to the left, one pixel to the right, one pixel above or one pixel below.
    # This is to account for if the index is out of range (e.g. if column is 0,
    # we should not be able to scan into column -1)
    leftValid = True
    rightValid = True
    topValid = True
    botValid = True
    if (column == 0):
        leftValid = False
    if (column == len(labelImage) - 1):
        rightValid = False
    if (row == 0):
        topValid = False
    if (row == len(labelImage) - 1):
        botValid = False
    # If leftValid is True, then it means we can scan to the left pixels.
    # Check one pixel left of labelImage[row][column]
    if (leftValid):
        left = labelImage[row][column - 1]
        valid_neighbour_pixel_check(neighbours, label, left)
    # Check one pixel top left.
    if (leftValid and topValid):
        topLeft = labelImage[row - 1][column - 1]
        valid_neighbour_pixel_check(neighbours, label, topLeft)
    # Check one pixel bottom left.
    if (leftValid and botValid):
        botLeft = labelImage[row + 1][column - 1]
        valid_neighbour_pixel_check(neighbours, label, botLeft)
    # Check one pixel top.
    if (topValid):
        top = labelImage[row - 1][column]
        valid_neighbour_pixel_check(neighbours, label, top)
    # Check one pixel bottom.
    if (botValid):
        bot = labelImage[row + 1][column]
        valid_neighbour_pixel_check(neighbours, label, bot)
    # Check one pixel right.
    if (rightValid):
        right = labelImage[row][column + 1]
        valid_neighbour_pixel_check(neighbours, label, right)
    # Check one pixel top right.
    if (rightValid and topValid):
        topRight = labelImage[row - 1][column + 1]
        valid_neighbour_pixel_check(neighbours, label, topRight)
    # Check one pixel bot right
    if (rightValid and botValid):
        botRight = labelImage[row + 1][column + 1]
        valid_neighbour_pixel_check(neighbours, label, botRight)
        
# Below is to check if the neighbourLabel should be placed into the neighbours list.
# The time complexity of this is O((p+1)log(p+1)) in the worst case (needing to append and sort)
# and the best case is O(logp), which is simply a binary search only
# Arguments:
#   - neighbours: The list of neighbours for label
#   - label: The label we are currently checking neighbours for.
#   - neighbourLabel: A potential neighbourLabel to add to the neighbours list.
def valid_neighbour_pixel_check(neighbours, label, neighbourLabel):
    # Check that the neighbourLabel is not the same as the current label we are looking at,
    # and that the neighbourLabel is not the background (0).
    if (not neighbourLabel == label and not neighbourLabel == 0):
        # If the pixel immediately to the left is not the current label
        # we are looking at, and is not the background, check if this label
        # has already been added to neighbours in O(logp) time (binary search). 
        # If it has, then skip. If not, add to the neighbours list and sort in O(plogp) time.
        if (not exists(neighbours, neighbourLabel, 0, len(neighbours) - 1)):
            neighbours.append(neighbourLabel)
            neighbours.sort()

# Below is to perform a binary search for a value val in arr. This is therefore performed
# in O(logn) time.
# Arguments:
#   - arr: The array
#   - val: The value to check for.
#   - first: The first index to check for
#   - last: The last index to check for.
# Returns:
#   - True if the value exists in first - last. False if the value does not exist in first - last
def exists(arr, val, first, last):
    # Base cases
    if (len(arr) == 0):
        return False
    if (first == last):
        if (arr[first] == val):
            return True
        else:
            return False
    # Else, continue to base case.
    mid = int((first + last)/2)
    # Return true or false, based on the base cases.
    return (exists(arr, val, first, mid) or exists(arr, val, mid + 1, last))

# ===============TODO======================
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
        for i in range(0, len(header)):
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

def filter_one_hundred_mean_intensity(data):
    if (data['intensity_mean'] == 1):
        return True
    return False