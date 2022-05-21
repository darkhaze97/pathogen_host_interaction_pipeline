from dash import html
from PIL import Image
import numpy as np

from .imagebox import imagebox

def mainview(info):
    print(info['cellImages'][0][0])
    # data = np.reshape((info['cellImages'][0][0]), (1024, 720))
    data = Image.fromarray(info['cellImages'][0][0].astype(np.uint8))
    data.save('lol.png')
    return html.Div(
        children=[imagebox(data)],
        style={
            'backgroundColor': 'blue',
            'top': 0,
            'margin-left': '18%',
            'width': '82%',
            'height': '100%',
            'position': 'static'
        }
    )
# Checkboxes
# Able to select: Show all segmentation labels
# Able to select: Show segmentation label of selected cell

# Then next to it, show the enlarged image of the cell, highlight
# the row of the data table that represents the data for this cell.
#

# At the bottom, show data table