import math
import os
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import numpy as np
import imagej
import csv
ij = imagej.init()

from decision_tree import predict

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
def get_intersection_information(pathogenImages, cellImages, savePath):
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

    # Now, rejoin the intracellularPathogens with the corresponding cell labels, so that
    # we can begin counting the number of infected cells and uninfected cells.
    cellPathogenLabels = []
    for i in range(0, len(intracellularPathogens)):
        joinedLabels = segmentation.join_segmentations(intracellularPathogens[i], labelledCell[i])
        cellPathogenLabel = measure.label(joinedLabels)
        cellPathogenLabels.append(cellPathogenLabel)

    pathogenInfo = {'bounding_box': [], 'area': [], 'image': [], 'perimeter': [],
                    'diameter': [], 'circularity': []}
    cellInfo = {'bounding_box': [], 'area': [], 'image': [], 'vacuole_number': [], 'perimeter': [],
                'diameter': [], 'circularity': []}

    # Analyse the properties of each label in each image using regionprops
    for i in range(0, len(cellPathogenLabels)):
        # We will obtain the region properties, and then scan through these
        props = measure.regionprops(cellPathogenLabels[i], intensity_image=pathogenImages[i][1])
        # props will be sorted in ascending order by label number.
        
        # For each label, find all the neighbours that are not the background.
        # This will occur in O(8*q^2*log(p)) time in the general case of
        # checking. It will be O(8*q^2*(p+1)log(p+1)) if we are inserting into the
        # neighbours list (defined later). How we obtained the log(p) and (p+1)log(p+1) will
        # be described in scan_neighbours.
        # Note: q refers to the size of the bounding box, which will always be less than
        # 2048, since regions that are adjacent to the edges have been removed. This is the
        # reason why I used bounding boxes, to limit q. p is referring
        # to the size of neighbours, and is defined by: 0 <= p < infinity. However, p is
        # usually never grows too big. 
        # Overall, if we have n labels to scan through, then our performance will be
        # O(8*n*q^2*(p+1)log(p+1)). Therefore, this algorithm is limited more by the number of 
        # labels that we have in an image, rather than how we use our data structures. 
        # E.g. if we used the RAG data structure defined in skimage, it would take 
        # approximately 20 seconds to form the data structures for a small set of images.
        # In our version, it takes about 12 seconds for the same set, which is a 40% increase
        # in performance. Another rationale behind using this algorithm as compared to the
        # RAG library in skimage is that we are only interested in looking at the
        # neighbours. Therefore, there would be unneccessary overhead by using RAG.
        for label in props:
            # First, obtain the bounding box. This will be used many times later.
            bound = label.bbox
            
            plt.imshow(label.image)
            plt.show()

            # Skip the background, as we do not really need to analyse this.
            if (label.label == 0):
                continue
            # Scan through each label, obtain the bounding box, and then scan through
            # the bounding box in the original image (cellPathogenLabels[i]) to find the
            # neighbours.
            # Scan through bound[0] - bound[2] to then scan through bound[1] and bound[3]
            # to find the label defined by label.label. Then, once we reach the label, we
            # simply check the neighbours.
            neighbours = []
            for row in range(bound[0], bound[2]):
                for column in range(bound[1], bound[3]):
                    if (not cellPathogenLabels[i][row][column] == label.label):
                        continue
                    # Else, we simply check the neighbours.
                    scan_neighbours(cellPathogenLabels[i], row, column, label.label, neighbours)
            
            # If there are neighbours, then we either have a pathogen or an infected cell. 
            # This is because the pathogen is touching the cell, and the cell can be touching 
            # at least one pathogen.
            #       If current label's area is bigger than all the adjacent labels, then the current label is an
            #       infected cell.
            #       Else, we have a pathogen.
            # Else, if there are no neighbours, then the current label is a non-infected cell
            if (len(neighbours) >= 1): 
                currentLabelArea = label.area
                # Now, scan through the list of adjacent labels, and compare their areas.
                isCell = True
                for adjacentLabel in neighbours:
                    # If an adjacent label has a greater area, then this is a pathogen
                    if (props[adjacentLabel - 1].area > currentLabelArea):
                        isCell = False
                if (isCell):
                    # If this is a cell, then record the number of pathogens within it as
                    # well
                    extract_cell_info(cellInfo, i, label, len(neighbours))
                else:
                    # Obtain fluorescence data first. This will be used for the
                    # decision tree later.
                    measure_fluorescence(pathogenInfo, pathogenImages[i], bound, savePath)
                    extract_pathogen_info(pathogenInfo, i, label)
            elif (len(neighbours) == 0):
                extract_cell_info(cellInfo, i, label, 0)
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
def extract_pathogen_info(pathogenInfo, imageNum, regionInfo):
    pathogenInfo['bounding_box'].append(regionInfo.bbox)
    pathogenInfo['area'].append(regionInfo.area)
    pathogenInfo['perimeter'].append(regionInfo.perimeter)
    pathogenInfo['image'].append(imageNum)
    pathogenInfo['diameter'].append(regionInfo.equivalent_diameter_area)
    pathogenInfo['circularity'].append(4 * math.pi * regionInfo.area/(regionInfo.perimeter ** 2))

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
