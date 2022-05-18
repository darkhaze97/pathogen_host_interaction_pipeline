import math

import PySimpleGUI as sg
import cv2
import pickle
import csv

from frontend import visualize

# from image import image_analysis

from stage_one import image_analysis

def main():
    nucleiPath = ''
    pathogenPath = ''
    cellPath = ''
    
    fileList = [
        [
            sg.Text('Select the nuclei image folder.'),
            sg.In(size=(25, 1), enable_events=True, key='#nucleiimagepath'),
            sg.FolderBrowse(),
        ],
        [
            sg.Text('Select the pathogen image folder.'),
            sg.In(size=(25, 1), enable_events=True, key='#pathogenimagepath'),
            sg.FolderBrowse(),
        ],
        [
            sg.Text('Select the cell image folder.'),
            sg.In(size=(25, 1), enable_events=True, key='#cellimagepath'),
            sg.FolderBrowse(),
        ],
        [
            sg.Button('Confirm')
        ]
    ]

    layout = [
        [
            sg.Column(fileList)
        ]
    ]
    window = sg.Window(title='Pipeline', layout=layout)
    while True:
        event, values = window.read()
        if (event == sg.WIN_CLOSED):
            exit(0)
        elif (event == 'Confirm'):
            nucleiPath = values['#nucleiimagepath']
            pathogenPath = values['#pathogenimagepath']
            cellPath = values['#cellimagepath']
            break
    
    window.close()
        
    # Obtain the magnification from the user.
    magnification = 0
    micronpp = 0
    customThreshold = None
    paramList = [
        [
            sg.Text('Magnification of the image:'),
            sg.In(size=(25, 1), enable_events=True, key='#magnification')
        ],
        [
            sg.Text('Micron per pixel (Default 6.5):'),
            sg.In(size=(25, 1), enable_events=True, key='#micronpp', default_text='6.5')
        ],
        [
            sg.Text('Optional custom threshold value:'),
            sg.In(size=(25, 1), enable_events=True, key='#customthreshold')
        ],
        [
            sg.Button('Confirm')
        ]
    ]
    
    layout = [
        [
            sg.Column(paramList)
        ]
    ]
    
    window = sg.Window(title='Pipeline', layout=layout)
    while True:
        event, values = window.read()
        if (event == sg.WIN_CLOSED):
            exit(0)
        elif (event == 'Confirm'):
            magnification = float(values['#magnification']) if not values['#magnification'] == '0' \
                            else 1.0
            micronpp = float(values['#micronpp'])
            if (not values['#customthreshold'] == '' and values['#customthreshold'].isnumeric()):
                customThreshold = int(values['#customthreshold'])
            break
    
    window.close()
    
    savePath = ''
    
    saveComponent = [
        [
            sg.Text('Choose a file name.'),
            sg.In(size=(25,1), enable_events=True, key='#filename')
        ],
        [
            sg.Text('Select a path to save the data.'),
            sg.In(size=(25,1), enable_events=True, key='#savepath'),
            sg.FolderBrowse()
        ],
        [
            sg.Button('Confirm')
        ]
    ]
    
    layout = [
        [
            sg.Column(saveComponent)
        ]
    ]
    window = sg.Window(title='Pipeline', layout=layout)
    
    while True:
        event, values = window.read()
        if (event == sg.WIN_CLOSED):
            break
        elif (event == 'Confirm'):
            savePath = values['#savepath'] + '/' + values['#filename']
            break
    
    window.close()
        
    info = image_analysis(nucleiPath, pathogenPath, cellPath, customThreshold, savePath)
    micronpp = micronpp/magnification
    # Change the area and perimeter values using the micron per pixel for the given magnification
    change_measurements(info, micronpp)
    
    savePath = values['#savepath'] + '/' + values['#filename']
    # First, create the pickle file. This is to use with visualize.py, so that
    # the user can see the images again in the GUI.
    print(info)
    with open(savePath + '.pickle', 'wb') as f:
        pickle.dump(info, f)

    # Next, create the csv for the data.
    with open(savePath + '.csv', 'w') as f:
        write_csv(f, info)
            
    visualize(info)

# The function below is called by the main function. It will simply call
# change_measurements_entity on pathogenInfo and cellInfo separately. The rationale
# for this function was to decouple it from the main function, so that the main function
# does not need to know that it needs to call change_measurements on 'pathogenInfo'
# or 'cellInfo'. Any extra entities that I want to print can simply be added to
# change_measurements instead of the main function. Whenever a new column/stat is to be added,
# simply call addColumn.
# Arguments:
#   - info: Dictionary containing all information generated from image_analysis in image.py
#   - micronpp: Microns per pixel.
def change_measurements(info, micronpp):
    # Change values in readout1, e.g. the mean size of pathogens.
    info['readout1']['mean_pathogen_size'] = info['readout1']['mean_pathogen_size'] * micronpp * micronpp
    # We do not access 'area' or 'perimeter' directly, for extensibility.
    # We use is_measurement, and also change it if we include any extra measurements,
    # and do not need to change anything in this function.
    change_measurements_entity(info, micronpp, 'pathogenInfo')
    change_measurements_entity(info, micronpp, 'cellInfo')
    # Add any new columns that should be examined. Read the documentation for 
    # addColumn for further details.
    # addColumn('circularity', info, 'pathogenInfo',
    #           lambda dict, i: (4 * math.pi * dict['area'][i]/(dict['perimeter'][i] ** 2)))
    # addColumn('circularity', info, 'cellInfo',
    #           lambda dict, i: (4 * math.pi * dict['area'][i]/(dict['perimeter'][i] ** 2)))

# The function below changes the measurements in an entity's information.
# It does this by changing the pixels in measurements (like area or perimeter) into
# microns.
# Arguments:
#   - info: Dictionary containing all information generated from image_analysis in image.py
#   - micronpp: Microns per pixel.
#   - entityInfo: The entity in which measurements will be changed.
def change_measurements_entity(info, micronpp, entityInfo):
    if (entityInfo not in info):
        print('This entity is not in the info dictionary. Please contact creator.')
        return
    for key in info[entityInfo]:
        if (key == 'perimeter' or key == 'diameter' or key == 'dist_nuclear_centroid'):
            for i in range(0, len(info[entityInfo][key])):
                info[entityInfo][key][i] = info[entityInfo][key][i] * micronpp
        elif (key == 'area'):
            for i in range(0, len(info[entityInfo][key])):
                info[entityInfo][key][i] = info[entityInfo][key][i] * micronpp * micronpp

# The function below is to simply add an extra value to look at, apart from area and
# perimeter.
# Arguments:
#   - colName: The new column name
#   - info: The dictionary containing all information generated from image_analysis in image.py
#   - entityInfo: The entity in which the new column will be added
#   - fnc: A function that will be applied to each entity. The index and info[entityInfo]
#          will be passed in, where the index is used to obtain the current entity that is
#          being examined. This function must have a return value.
def addColumn(colName, info, entityInfo, fnc):
    # For every entity in info[entityInfo], apply fnc
    # randKey simply obtains a random key that is already present in info[entityInfo],
    # so that we do not need to hard code a column name in, like area or perimeter
    randKey = list(info[entityInfo].keys())[0]
    entityNum = len(info[entityInfo][randKey])
    info[entityInfo][colName] = []
    for i in range(0, entityNum):
        info[entityInfo][colName].append(fnc(info[entityInfo], i))

# Below simply checks if the key is of a measurement.
# Arguments:
#   - key: The key to be examined.
def is_measurement(key):
    return True if (key == 'area' or key == 'perimeter') else False

# Below checks if the key is valid for the csv. 
# The valid keys exclude bounding_box and image.
# Arguments:
#   - key: The key to be examined
def valid_csv_header(key):
    return False if (key == 'bounding_box' or key == 'image') else True

# The function below is called by the main function. It will simply call
# write_csv_entity on pathogenInfo and cellInfo separately. The rationale
# for this function was to decouple it from the main function, so that the main function
# does not need to know that it needs to call write_csv on 'pathogenInfo'
# or 'cellInfo'. Any extra entities that I want to print can simply be added to
# write_csv instead of the main function.
# Arguments:
#   - info: Dictionary containing all information generated from image_analysis in image.py
#   - f: The csv file open for writing.
def write_csv(f, info):
    write_csv_entity(f, info, 'pathogenInfo', 'Pathogen Information')
    write_csv_entity(f, info, 'cellInfo', 'Cell Information')

# Below is to write to a csv file with a subheading, and the entity to target.
# Arguments:
#   - f: The csv file open for writing
#   - info: Dictionary containing all information generated from image_analysis in image.py
#   - entityInfo: The entity to target for writing (e.g. pathogenInfo or cellInfo)
#   - subheading: The subheading for the entity to target. This will be placed onto a
#                 separate row.
def write_csv_entity(f, info, entityInfo, subheading):
    writer = csv.writer(f)
    #Create the pathogen information.
    writer.writerow([subheading])
    # Add the headers for the pathogen information.
    pathogenHeaders = []
    for key in info[entityInfo].keys():
        if (valid_csv_header(key)):
            pathogenHeaders.append(key)
    # Then, write these headers to the csv
    writer.writerow(pathogenHeaders)
    rows = []
    for i in range(0, len(info[entityInfo]['area'])):
        row = []
        for key in info[entityInfo].keys():
            if (valid_csv_header(key)):
                row.append(info[entityInfo][key][i])
        rows.append(row)
    for row in rows:
        writer.writerow(row)

# =============== ADD ANY NEW COLUMN METHODS HERE ================
# These methods must return a value.

# ================================================================
if __name__ == '__main__':
    main()
