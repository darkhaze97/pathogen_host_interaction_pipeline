import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

from .helper import filter_zero_mean_intensity

from .voronoi import voronoi_seg, voronoi_seg_alt

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
        # If there are only 2 nuclei, we can easily perform a voronoi segmentation by
        # forming a perpendicular line at the midpoint of the line that joins the two
        # centroids.
        # If there are more than 2 nuclei, use the scipy.spatial.Voronoi method to
        # perform a voronoi segmentation.
        if (len(centroidList) == 2):
            cellImg = voronoi_seg_alt(centroidList, cellImg)
            cellImg = correct_voronoi(cellImg, nucleiBox)
            # plt.imshow(cellImg)
            # plt.show()
        elif (len(centroidList) > 2):
            cellImg = voronoi_seg(centroidList, cellImg)
            cellImg = correct_voronoi(cellImg, nucleiBox)
            # plt.imshow(cellImg)
            # plt.show()
        # Add the label to the new labelled image.
        boundBox = newLabelImg[bound[0]:bound[2], bound[1]:bound[3]]
        # If a specific pixel in the newLabelImg is of label 0, we can add the cell label
        # bounded by bound[0]:bound[2] and bound[1]:bound[3] freely to it. However,
        # if there is a pixel in this box that is not originally 0 and is not the label
        # of the current cell being analysed, this means that this 
        # region can be potentially lost. Therefore, the corresponding pixels from the 
        # region have been added back in.
        newLabelImg[bound[0]:bound[2], bound[1]:bound[3]] = np.where(boundBox == 0, cellImg, boundBox)

# This function is called after a voronoi segmentation. This is to fix the segmentation,
# wherein a voronoi region may contain labels that are not adjacent to each other. This
# indicates that the region is actually connected to another voronoi region.
# Therefore, this function facilitates this reconnection. A while loop will call
# to another function, which will progressively correct the segmented image after
# each pass, by connecting a non-connected voronoi segment to different voronoi region.
# Arguments:
#   - segmentedImg (numpy array): The voronoi segmented image.
#   - nucleiBox (numpy array): The image of the nuclei for the image.
# Returns:
#   - segmentedImg (numpy array): The corrected voronoi segmented image.
def correct_voronoi(segmentedImg, nucleiBox):
    while True:
        vorRegions = measure.regionprops(segmentedImg)
        segmentedImg, corrected = connect_region(vorRegions, segmentedImg, nucleiBox)
        if (not corrected):
            break
    return segmentedImg
        
# This function is called by correct_voronoi. After this function reconnects a non-connected
# voronoi segment to a different voronoi region, it will return with the updated segmentedImg.
# This function returns, instead of continuing in the loop, as the segmentedImg must be
# relabelled after each correction occurs.
# Arguments:
#   - vorRegions (list): list of attributes for each voronoi region
#   - segmentedImg (numpy array): The voronoi segmented image.
#   - nucleiBox (numpy array): The image of the nuclei for the image
# Returns:
#   - A tuple of segmentedImg and a bool: The bool represents whether the image has been
#     corrected in this step. If it is corrected, then we call connect_region again. If it
#     has not, then we do not need to call connect_region again, as this indicates that
#     the image is already correct.
def connect_region(vorRegions, segmentedImg, nucleiBox):
    for r in vorRegions:
        # First, obtain the regions that are not meant to be in the voronoi region
        # defined by r.
        separateRegs = obtain_separate_regions(segmentedImg, nucleiBox, r)
        
        # Make a list of the regions other than the current one being analysed.
        otherRegions = [vorRegion for vorRegion in vorRegions if not vorRegion.label == r.label]
        
        # Then, attempt to connect the regions (defined by cellProps) to otherRegions.
        # Reminder: sepReg is referring to the regions that are separated from the main region's
        # nucleus. OtherReg is a cell in another voronoi region that sepReg might be connected to
        # instead.
        for sepReg in separateRegs:
            sepRegImg = np.zeros(segmentedImg.shape)
            # First, obtain the image of the separated region.
            sepRegImg[sepReg.bbox[0]:sepReg.bbox[2], sepReg.bbox[1]:sepReg.bbox[3]] = sepReg.image
            sepRegImg = np.where(sepRegImg == 1, r.label, sepRegImg)
            for otherReg in otherRegions:
                # Create a new image of the otherRegion.
                otherRegImg = np.zeros(segmentedImg.shape)
                otherRegImg[otherReg.bbox[0]:otherReg.bbox[2], otherReg.bbox[1]:otherReg.bbox[3]] =otherReg.image
                # Set the labels of otherRegImg to be otherReg.label
                otherRegImg = np.where(otherRegImg == 1, otherReg.label, otherRegImg)
                # Label the otherRegImg. This will be used to compare whether the regions are connected.
                # We need the number of regions from this labelled image.
                otherRegLabel, otherRegNum = measure.label(otherRegImg, return_num=True)
                # Now, connect the separated region and the other region.
                # The separated region will have the same label as the other region,
                # to allow measure.label to correctly label adjacent regions, in the case
                # that the separated region is connected to the other region. If this is not
                # done, then label will assume that adjacent regions with different
                # labels will be two separate regions, rather than one single region.
                connectImage = np.where(sepRegImg > 0, otherReg.label, otherRegImg)
                connectLabel, numReg = measure.label(connectImage, return_num=True)
                # Check if the resulting number of regions is the same as before.
                if (numReg == otherRegNum):
                    # If the number of labelled regions is the same as before the sepReg
                    # was added on, then sepReg is connected to the new region.
                    return (np.where(sepRegImg > 0, otherReg.label, segmentedImg), True)
    # If we have not changed anything, return True in the second index of the tuple.
    # This will stop the overarching while loop that is calling connect_region.
    return (segmentedImg, False)

# This function obtains the separated regions in a voronoi region. A separated region
# is a region that is not adjacent to the nucleus of the voronoi region.
# I.e. it is not part of the main cell in this voronoi region, and instead, part of
# another cell in another voronoi region.
# Arguments:
#   - segmentedImg (numpy array): The voronoi segmented image.
#   - nucleiBox (numpy array): The image of the nuclei.
#   - ogRegion (Dictionary): A list of information for the region that we are currently
#                            scanning in.
# Returns:
#   - A list of the regions that are not touching the nuclei, separate to the 
#     main label in the voronoi region.
def obtain_separate_regions(segmentedImg, nucleiBox, ogRegion):
    # Below obtains the nucleus for this voronoi region. It will then be used
    # as an intensity image to determine the regions that are not connected to this
    # nucleus, to find the correct region that they are connected to.
    intensityImage = (segmentedImg == ogRegion['label']).astype(int)
    nucleiProps = measure.regionprops(nucleiBox,
                              intensity_image=intensityImage
                             )
    # Obtain the nucleus with the highest intensity.
    n = max(nucleiProps, key=lambda n:n.intensity_mean)
    
    # rLabel is the labelling of the region defined by ogRegion.
    # It will differentially label regions that are not connected.
    # The shape of rLabel is segmentedImg.shape, so that we can record the exact
    # positions of each label, and connect with other labels correctly.
    rLabel = np.zeros(segmentedImg.shape)
    rLabel[ogRegion.bbox[0]:ogRegion.bbox[2], ogRegion.bbox[1]:ogRegion.bbox[3]] = ogRegion.image
    rLabel = measure.label(rLabel)
    
    # intensityImage will contain an image of the nucleus for the region defined
    # by ogRegion. This will be used to extract out the regions that are not connected to this
    # nucleus.
    # Once again, we use nucleiBox.shape, so that we can use this as an intensity image
    # with rLabel, noting that nucleiBox and segmentedImg have the same shape.
    intensityImage = np.zeros(nucleiBox.shape)
    intensityImage[n.bbox[0]:n.bbox[2], n.bbox[1]:n.bbox[3]] = n.image

    # The section below simply filters out the labelled region that contains the
    # nucleus. The regions that are not connected to the cell (and therefore,
    # not connected to the nucleus) remain. These regions are connected to a label
    # in another voronoi region.
    separateRegs = measure.regionprops(rLabel, intensity_image=intensityImage)
    return list(filter(filter_zero_mean_intensity, separateRegs))