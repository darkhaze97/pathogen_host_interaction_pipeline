import PySimpleGUI as sg
from PIL import Image
import cv2

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
            break
        elif (event == 'Confirm'):
            nucleiPath = values['#nucleiimagepath']
            pathogenPath = values['#pathogenimagepath']
            cellPath = values['#cellimagepath']
            break
    
    window.close()
        
    info = image_analysis(nucleiPath, pathogenPath, cellPath)
    
    pathogenColumn = [
        [
            sg.Text('Pathogen information')
        ]
    ]
    
    for i in range(0, len(info['pathogenInfo']['area'])):
        pathogenColumn.append([sg.Text(form_pathogen_info(info['pathogenInfo'],
                                                          i)),
                               sg.Image(reform_image_from_array(info['pathogenInfo']['image'][i],
                                                                info['pathogenInfo']['bounding_box'][i]),
                                        size = (200, 200)
                               )])
    
    layout = [
        [
            sg.Column(pathogenColumn)
        ]
    ]
    
    window = sg.Window(title='Pipeline', layout=layout)
    while True:
        event, values = window.read()
        if (event == sg.WIN_CLOSED):
            break
    
    window.close()
    
    cellColumn = [
        [
            sg.Text('Cell information')
        ]
    ]
    
    # for i in range(0, len(info['cellInfo']['area'])):
    #     cellColumn.append([sg.Image(reform_image_from_array(info['cellInfo']['image'][i]),
    #                                 size = (200, 200))
    #     ])
    cellColumn.append([sg.Image(reform_image_from_array(info['cellInfo']['image'][0],
                                                        info['cellInfo']['bounding_box'][0]))])
    layout2 = [
        [
            sg.Column(cellColumn)
        ]
    ]
    
    window = sg.Window(title='Pipeline', layout=layout2)
    while True:
        event, values = window.read()
        if (event == sg.WIN_CLOSED):
            break
    

def form_pathogen_info(info, index):
    return ('Pathogen number: ' + str(index) + '\nArea: ' + str(info['area'][index]))

def reform_image_from_array(imageArray, bbox):
    cropImage = imageArray[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    success, encoded_image = cv2.imencode('.png', cropImage)
    return encoded_image.tobytes()

if __name__ == '__main__':
    main()
