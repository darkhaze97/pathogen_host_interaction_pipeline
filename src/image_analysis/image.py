import os
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, segmentation, morphology, img_as_float
from skimage.segmentation import clear_border
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter
import pandas as pd
import cProfile
plt.style.use('fivethirtyeight')


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
def image_analysis(nucleiPath, pathogenPath, cellPath, threshold):
    
    # Scan through the nuclei, and correct...
    nucleiImages = label_nuclei_images(nucleiPath, threshold)
    pathogenImages = label_pathogen_images(pathogenPath, threshold)
    cellImages = label_cell_images(cellPath, nucleiImages, threshold)
    
    intersection_info = get_intersection_information(pathogenImages, cellImages)
    # Generate information about whether the cells are infected, and the number of
    # pathogens in the cell. In addition, generate information on the pathogens.
    # cProfile.runctx('get_intersection_information(pathogenImages, cellImages)', {'get_intersection_information': get_intersection_information}, {'pathogenImages': pathogenImages, 'cellImages': cellImages}, filename='report.txt' )
    
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
        **intersection_info
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
#   - list<(labelled images (list), images (list))>: A list with tuples of labelled
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
        
        # If nucleiImages are provided, combine the labelImg with the nucleiLabel.
        # Note that nucleiImage[0] corresponds to the nucleiLabel that was accessed first.
        if (not nucleiImages == None):
            # plt.imshow(alteredImg)
            # plt.show()
            # img  = img_as_float(alteredImg)
            # img = gaussian_filter(alteredImg, 1)
            # seed = np.copy(img)
            # seed[1:-1, 1:-1] = img.min()
            
            # mask = img
            # dilated = reconstruction(seed, mask, method='dilation')
            
            # # Apply watershedding to the cell images.
            # # plt.imshow(img - dilated)
            # # plt.show()
            # kernel = np.ones((3,3), np.uint8)
            
            # sure_bg = cv2.dilate(alteredImg, kernel, iterations=10)
            
            # distTransform = cv2.distanceTransform(alteredImg, cv2.DIST_L2, 5)
            
            # ret2, sure_fg = cv2.threshold(distTransform, 0.28*distTransform.max(), 255, 0)
            
            # sure_fg = np.uint8(sure_fg)
            # unknown = cv2.subtract(sure_bg, sure_fg)
            
            # ret3, markers = cv2.connectedComponents(sure_fg)
            
            # markers = markers + 1
            # # 1 is now the background
            
            # markers[unknown==255] = 0
            
            # alteredImg = cv2.watershed(image, markers)
            
            # alteredImg = (alteredImg > 1).astype(int)
            
            alteredImg = combine_cell_nuclei_label(alteredImg, nucleiImages[i][0])
            i = i + 1

        # Convert the alteredImg into a boolean array. Then we are able to remove 
        # small objects.
        labelImg = alteredImg > 0
        
        # Remove small objects in the altered image. The minimal size will be based on if the
        # image was a pathogen, cell or nucleus. A larger minimal value is needed for
        # the cell, since there can be a lot of noise in these photos compared to the nuclei
        # or pathogens.
        labelImg = morphology.remove_small_objects(labelImg, 2000) if (not nucleiImages == None) \
                        else morphology.remove_small_objects(labelImg, 100)

        # Then, remove holes that are within labels.
        labelImg = morphology.remove_small_holes(labelImg, 1000)
        
        # Place unique labels on segmented areas
        labelImg = measure.label(labelImg, connectivity=greyImage.ndim) 

        # plt.imshow(labelImg)
        # plt.show()
        images.append((labelImg, greyImage))
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

    pathogenInfo = {'bounding_box': [], 'area': [], 'image': [], 'perimeter': []}
    cellInfo = {'bounding_box': [], 'area': [], 'image': [], 'pathogen_number': [], 'perimeter': []}

    # Analyse the properties of each label in each image using regionprops
    for i in range(0, len(cellPathogenLabels)):
        # We will obtain the region properties, and then scan through these
        props = measure.regionprops(cellPathogenLabels[i])
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
            # Skip the background, as we do not really need to analyse this.
            if (label.label == 0):
                continue
            # Scan through each label, obtain the bounding box, and then scan through
            # the bounding box in the original image (cellPathogenLabels[i]) to find the
            # neighbours.
            bound = label.bbox
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
                    extract_pathogen_info(pathogenInfo, i, label)
            elif (len(neighbours) == 0):
                extract_cell_info(cellInfo, i, label, 0)
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
def extract_pathogen_info(pathogenInfo, imageNum, regionInfo):
    pathogenInfo['bounding_box'].append(regionInfo.bbox)
    pathogenInfo['area'].append(regionInfo.area)
    pathogenInfo['perimeter'].append(regionInfo.perimeter)
    pathogenInfo['image'].append(imageNum)

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
    cellInfo['pathogen_number'].append(pathogenNum)

if __name__ == '__main__':
    image_analysis("../Images/Nuclei", "../Images/Pathogen", "../Images/Cell", 10000)