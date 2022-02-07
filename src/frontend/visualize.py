
import sys
import PySimpleGUI as sg
from PySimpleGUI import Text, Image, Window, Column, Button, WIN_CLOSED
import cv2
import pickle
import tkinter as tk
# from PIL import Image, ImageTk
import matplotlib.pyplot as plt

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
    # Place the unused stats below, and follow the same structure as used for bounding_box
    # and image
    'bounding_box': lambda info, i: '',
    'image': lambda info, i: ''
}

def visualize(info):
    # root = tk.Tk()
    # canv = tk.Canvas(root, width=1048, height=1048, bg='white')
    # canv.grid(row=2, column=3)
    # img = Image.open("A01f00p00d00.png")
    # img = img.resize((400, 400))
    # img = ImageTk.PhotoImage(img)  # PIL solution
    # canv.create_image(0, 0, anchor=tk.NW, image=img)
    # root.mainloop()
    
#     global window
#     window = tk.Tk()
    
#     window.columnconfigure([0, 1, 2], minsize=150)

#     global viewingColumn
#     viewingColumn = tk.Frame(master=window)
#     viewingColumn.grid(row=0, column=2)
#     label2 = tk.Label(master=viewingColumn)
#     label2.pack()
    
#     global selectSpecificColumn
#     selectSpecificColumn = tk.Frame(master=window)
#     selectSpecificColumn.grid(row=0, column=1)

#     selectColumn = tk.Frame(master=window)
#     selectColumn.grid(row=0, column=0)
#     selectColumn.rowconfigure([0, 1, 2], minsize=50)
#     viewAllButton = tk.Button(master=selectColumn,
#                               text='View whole images',
#                               command = lambda:view_all(info, selectSpecificColumn))
#     selectColumnLabel = tk.Label(master=selectColumn, text='Select types of images to view.')
#     selectColumnLabel.grid(row=0, column=0)
    
#     viewAllButton.grid(row=1, column=0)

#     viewPathogenButton = tk.Button(master=selectColumn,
#                                    text='View pathogen labels')
#     viewPathogenButton.grid(row=2, column=0)

#     viewCellButton = tk.Button(master=selectColumn, text='View cell labels')
#     viewCellButton.grid(row=3, column=0)

    
#     window.mainloop()
    
    # selectColumn = [
    #     [
    #         Button('View whole images', key='#wholeimages')
    #     ],
    #     [
    #         Button('View pathogen labels'),
    #     ],
    #     [
    #         Button('View cell labels')
    #     ]
    # ]
    
    # selectSpecific = []
    # viewingColumn = []
    
    # baseLayout = [
    #     [
    #         Column(selectColumn),
    #         Column(selectSpecific, key='#selectspecificimage'),
    #         Column(viewingColumn, key='#viewingcolumn')
    #     ]
    # ]
    
    # window = Window(title='Pipeline', layout=baseLayout)
    
    # while True:
    #     event, values = window.read()
    #     if (event == WIN_CLOSED):
    #         break
    #     elif (event == '#wholeimages'):
    #         selectSpecificCol = window['#selectspecificimage']
    #         buttonList = []
    #         for i in range(0, len(info['cellImages'])):
    #             buttonList.append([Button('Field of view #' + str(i))])
    #         print('HARR')
    #         selectSpecificCol.update(buttonList)
    #         window.close()
    
    pathogenColumn = [
        [
            Text('Pathogen information')
        ]
    ]
    
    for i in range(0, len(info['pathogenInfo']['area'])):
        pathogenColumn.append([Text(form_pathogen_info(info['pathogenInfo'],i)),
                               Image(reform_cropped_image_from_array(info['pathogenImages'][info['pathogenInfo']['image'][i]][1],
                                                                     info['pathogenInfo']['bounding_box'][i]),
                                     size = (200, 200)
                               )])
    
    layout = [
        [
            Column(pathogenColumn)
        ]
    ]
    
    window = Window(title='Pipeline', layout=layout, size=(450, 600))
    while True:
        event, values = window.read()
        if (event == WIN_CLOSED):
            break
    
    window.close()
    
    cellColumn = [
        [
            Text('Cell information')
        ]
    ]
    
    for i in range(0, len(info['cellInfo']['area'])):
        cellColumn.append([Text(form_cell_info(info['cellInfo'], i)),
                           Image(reform_cropped_image_from_array(info['cellImages'][info['cellInfo']['image'][i]][1],
                                                                 info['cellInfo']['bounding_box'][i]),
                           size = (200, 200))
        ])
    # cellColumn.append([sg.Image(reform_image_from_array(info['cellInfo']['image'][0],
    #                                                     info['cellInfo']['bounding_box'][0]))])
    layout2 = [
        [
            Column(cellColumn, scrollable=True)
        ]
    ]
    
    window = Window(title='Pipeline', layout=layout2, size=(450, 600))
    while True:
        event, values = window.read()
        if (event == WIN_CLOSED):
            break
    

# def view_all(info, selectColumn):
#     global selectColumnInfo, selectSpecificColumn
#     clear_select_column()
#     selectColumnInfo = tk.Frame(master=selectSpecificColumn)
#     selectColumnInfo.columnconfigure([0, 1, 2], minsize=50)
#     for i in range(0, len(info['cellImages'])):
#         label = tk.Button(master=selectColumnInfo,
#                           text='Field of view #' + str(i),
#                           command=lambda:show_all_image(info['nucleiImages'][i],
#                                                         info['pathogenImages'][i],
#                                                         info['cellImages'][i]))
#         label.pack()
#     selectColumnInfo.pack()
    
# def show_all_image(nucleiImage, pathogenImage, cellImage):
#     global viewingColumnInfo, viewingColumn
#     clear_viewing_column()
#     viewingColumnInfo = tk.Frame(master=viewingColumn)
#     viewingColumnInfo.columnconfigure([0, 1, 2], minsize=50)
#     nuclei = tk.Canvas(viewingColumnInfo)
#     nuclei.grid(row=0, column=0)
#     img = ImageTk.PhotoImage(image=Image.fromarray(nucleiImage[1]))
    
#     # plt.imshow(nucleiImage[0])
#     # plt.show()
#     # img = ImageTk.PhotoImage(Image.open())
#     nuclei.create_image(20, 20, anchor=tk.NW, image=img)
#     # label = tk.Label(master=viewingColumnInfo, text='Hi I gu')
#     # label.grid(row=0, column=0)
#     # label = tk.Label(master=viewingColumnInfo, image=img)
#     # label.grid(row=0, column=0)
#     viewingColumnInfo.pack()
    
# def clear_select_column():
#     global selectColumnInfo
#     # Try block is for when the program first starts, and selectColumnInfo is not yet
#     # defined.
#     try:
#         selectColumnInfo.pack_forget()
#     except:
#         print('Continuing')

# def clear_viewing_column():
#     global viewingColumnInfo
#     try:
#         viewingColumnInfo.pack_forget()
#     except:
#         print('Continuing')

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
    