import PySimpleGUI as sg
import cv2
import pickle

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
            break
            exit(0)
        elif (event == 'Confirm'):
            nucleiPath = values['#nucleiimagepath']
            pathogenPath = values['#pathogenimagepath']
            cellPath = values['#cellimagepath']
            break
    
    window.close()
        
    info = image_analysis(nucleiPath, pathogenPath, cellPath)
    
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
            exit(0)
        elif (event == 'Confirm'):
            savePath = values['#savepath'] + '/' + values['#filename']
            with open(savePath, 'wb') as f:
                pickle.dump(info, f)
            break
    
    window.close()
            
    visualize(info)
    
if __name__ == '__main__':
    main()
