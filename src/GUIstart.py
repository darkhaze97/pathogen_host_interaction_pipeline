import PySimpleGUI as sg
import cv2
import pickle
import csv

from visualize import visualize

from image import image_analysis
# class SelectPath(ed.Component):
#     def __init__(self):
#         super().__init__()
#     def render(self):
#         form_data = ed.StateManager({
#             'Select the nuclei image folder.': pathlib.Path(os.getcwd()),
#             'Select the pathogen image folder.': pathlib.Path(os.getcwd()),
#             'cellPath': pathlib.Path(os.getcwd())
#         })
#         return View(layout='row')(
#             Label('Select the nuclei image folder.'),
#             Form(form_data, config=CheckBox),
#             Button(title='Confirm', onclick=)
#         )

def main():
    # ed.App(SelectPath()).start()
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
    
    magnifList = [
        [
            sg.Text('Magnification of the image:'),
            sg.In(size=(25, 1), enable_events=True, key='#magnification')
        ],
        [
            sg.Text('Micron per pixel (Default 6.5):'),
            sg.In(size=(25, 1), enable_events=True, key='#micronpp', default_text='6.5')
        ],
        [
            sg.Button('Confirm')
        ]
    ]
    
    layout = [
        [
            sg.Column(magnifList)
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
            break
    
    window.close()
        
    info = image_analysis(nucleiPath, pathogenPath, cellPath)
    micronpp = micronpp/magnification
    # Change the area and perimeter values using the micron per pixel for the given magnification
    change_measurements(info, micronpp)
    
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
            # First, create the pickle file. This is to use with visualize.py, so that
            # the user can see the images again in the GUI.
            with open(savePath + '.pickle', 'wb') as f:
                pickle.dump(info, f)


            # Change: Maybe do not need a csv --> User can already visualise
            # Next, create the csv for the data.
            # Get a copy of the info dictionary, an
            # with open(savePath + '.csv', 'w') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['Pathogen information'])
            #     # Add the headers for the pathogen information.
            #     pathogenHeaders = []
            #     for key in info['pathogenInfo'].keys():
            #         if (valid_csv_header(key)):
            #             pathogenHeaders.append(key)
            #     # Then, write these headers to the csv
            #     writer.writerow(pathogenHeaders)
            #     rows = []
            #     for i in range(0, len(info['pathogenInfo']['area'])):
            #         row = []
            #         for key in info['pathogenInfo'].keys():
            #             if (valid_csv_header(key)):
            #                 row.append(info['pathogenInfo'][key][i])
            #         rows.append(row)
            #     for row in rows:
            #         writer.writerow(row)
            break
    
    window.close()
            
    visualize(info)

# The function below is called by the main function. It will simply call
# change_measurements_entity on pathogenInfo and cellInfo separately. The rationale
# for this function was to decouple it from the main function, so that the main function
# does not need to know that it needs to call change_measurements on 'pathogenInfo'
# or 'cellInfo'. Any extra entities that I want to print can simply be added to
# change_measurements instead of the main function.
# Arguments:
#   - info: Dictionary containing all information generated from image_analysis in image.py
#   - micronpp: Microns per pixel.
def change_measurements(info, micronpp):
    # We do not access 'area' or 'perimeter' directly, for extensibility.
    # We use is_measurement, and also change it if we include any extra measurements,
    # and do not need to change anything in this function.
    change_measurements_entity(info, micronpp, 'pathogenInfo')
    change_measurements_entity(info, micronpp, 'cellInfo')

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
        if (is_measurement(key)):
            for i in range(0, len(info[entityInfo][key])):
                info[entityInfo][key][i] = info[entityInfo][key][i] * micronpp

# Below simply checks if the key is of a measurement.
# Arguments:
#   - key: The key to be examined.
def is_measurement(key):
    return True if (key == 'area' or key == 'perimeter') else False
        

# def valid_csv_header(key):
#     return False if (key == 'bounding_box' or key == 'image') else True

if __name__ == '__main__':
    main()
