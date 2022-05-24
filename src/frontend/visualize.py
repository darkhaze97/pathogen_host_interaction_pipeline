
import sys
import PySimpleGUI as sg
from dash import Dash, html, dcc, Output, Input, ctx
import dash_bootstrap_components as dbc
from PySimpleGUI import Text, Image, Window, Column, Button, WIN_CLOSED
import cv2
import pickle
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation, morphology, color, img_as_ubyte

import ComponentLibrary

# The variable below is to help print out statistics associated with each entity.
# Each stat, e.g. area, will be connected to a lambda function, which will print out
# the stat. The lambda function takes in an entity dictionary that contains all the
# information about the entity, and also an index
# that will be used to print out information related to the entity specified by the
# index. This will be used for both pathogenInfo and cellInfo.
# Whenever a new stat/column is added, you only need to add to this statPrinter. This makes it
# convenient.
statPrinter = {
    'area': lambda info, i: '\nArea: ' + str(round(info['area'][i], 2)) +  u'\u03bcm\u00b2',
    'perimeter': lambda info, i: '\nPerimeter: ' + str(round(info['perimeter'][i], 2)) + u'\u03bcm',
    'circularity': lambda info, i: '\nCircularity: ' + str(round(info['circularity'][i], 3)),
    'vacuole_number': lambda info, i: '\nNumber of pathogen/vacuole(s): ' + str(info['vacuole_number'][i]),
    'diameter': lambda info, i: '\nDiameter: ' + str(round(info['diameter'][i], 2)) + u'\u03bcm',
    'Mean': lambda info, i: '\nMean fluorescence: ' + str(info['Mean'][i]),
    'Max': lambda info, i: '\nMax fluorescence: ' + str(info['Max'][i]),
    'Min': lambda info, i: '\nMin fluorescence: ' + str(info['Min'][i]),
    'pathogens_in_vacuole': lambda info, i: '\nPathogens in the vacuole: ' + str(info['pathogens_in_vacuole'][i]),
    'dist_nuclear_centroid': lambda info, i: '\nDistance from nucleus: ' + str(round(info['dist_nuclear_centroid'][i], 2)) + u'\u03bcm',
    # Place the unused stats below, and follow the same structure as used for bounding_box
    # and image
    'bounding_box': lambda info, i: '',
    'image': lambda info, i: ''
}

def visualize(info):
    # print(info)
    
    # Obtain the names of each cell image, and use a hash table to map a cell image
    # name to an image.
    imgNameMap = {}
    for label, bwImg, path in info['cellImages']:
        imgNameMap[(path.split('/'))[-1]] = (label, img_as_ubyte(bwImg))
    
    show_all_labels('A01f00p00d00.tif', imgNameMap)
    
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    
    app.layout = html.Div(
        children=[
            ComponentLibrary.sidebar(imgNameMap),
            ComponentLibrary.mainview(imgNameMap)

        ],
        style={
            'width': '100%',
            'height': '100vh'
        }
    )

    # This is to update the centre image.
    @app.callback(
        Output("main-img", "src"),
        Input("img-select", "value"),
        Input("main-img-filter", "value"))
    def updateCellImg(file_name, filter):
        # NEXT: If I reselect the already selected filter, then I should deselect.
        print(file_name)
        if (filter == 'show-all'):
            return show_all_labels(file_name, imgNameMap)
        elif (filter == 'show-selected'):
            # Only show label of current selection.
            print('Lol')
        return Image.fromarray(img_as_ubyte(imgNameMap[file_name][1]))
    
    app.run_server(debug=True)

def show_all_labels(imgName, imgNameMap):
    # For each label, draw a contour line.
    labelImg = imgNameMap[imgName][0]
    bwImg = np.copy(imgNameMap[imgName][1])
    overlay = color.label2rgb(labelImg, bwImg)
    return Image.fromarray(img_as_ubyte(overlay))
    # plt.imshow(result_image)
    # plt.show()
    # # Obtain the regionprops of each label.
    # regionInfo = measure.regionprops(labelImg)
    # for label in regionInfo:
    #     # Obtain the contours for label. This needs to be done separately for each
    #     # label, so that the final contours on the image are correctly separated
    #     # (for adjacent labels)
    #     labelMap = np.zeros_like(labelImg, dtype='uint8')
    #     labelMap = (labelImg == label.label).astype(int)
    #     contours, _ = cv2.findContours(labelMap.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     cv2.drawContours(bwCpy, [contours[0]], -1, (0, 0, 255), 1)
    # return bwCpy

def form_pathogen_info(info, index):
    retStr = 'Pathogen number: ' + str(index)
    for key in info.keys():
        retStr = retStr + statPrinter[key](info, index)
    return retStr
    
def form_cell_info(info, index):
    retStr = 'Cell number: ' + str(index)
    for key in info.keys():
        retStr = retStr + statPrinter[key](info, index)
    return retStr

def form_whole_image_from_array(imageArray):
    imageArray = imageArray[10:100, 20:200]
    success, encoded_image = cv2.imencode('.png', imageArray)
    return encoded_image.tobytes()

def reform_cropped_image_from_array(imageArray, bbox):
    cropImage = imageArray[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    cropImage = cv2.resize(cropImage, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    success, encoded_image = cv2.imencode('.png', cropImage)
    return encoded_image.tobytes()

if __name__ == '__main__':
    # Check that we have the correct number of arguments
    if (not len(sys.argv) == 2):
        print('Correct use: python visualize.py (filename)')
        exit(0)
    filePath = sys.argv[1]
    with open(filePath, 'rb') as f:
        info = pickle.load(f)
        visualize(info)
    