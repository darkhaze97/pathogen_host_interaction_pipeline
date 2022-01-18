import PySimpleGUI as sg
from image import image_analysis

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
                                                          i))])
    
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
    

def form_pathogen_info(info, index):
    return ('Pathogen number: ' + str(index) + '\nArea: ' + str(info['area'][index]))


if __name__ == '__main__':
    main()
