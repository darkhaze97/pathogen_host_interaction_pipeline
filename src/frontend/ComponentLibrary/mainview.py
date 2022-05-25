from dash import html, dcc
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np

from .imagebox import imagebox
from .filterimage import filter_seg
from .table import obtain_table

def mainview(imgNameMap, cellInfo):
    data = Image.fromarray(imgNameMap[(list(imgNameMap.keys()))[0]][1])
    return html.Div(
        [
            imagebox(data),
            filter_seg(),
            obtain_table(cellInfo)
        ],
        style={
            'margin-left': '18%',
            'width': '82%',
            'height': '100%',
            'textAlign': 'center'
        },
    )
    
    
    
# Checkboxes
# Able to select: Show all segmentation labels
# Able to select: Show segmentation label of selected cell

# Then next to it, show the enlarged image of the cell, highlight
# the row of the data table that represents the data for this cell.
#

# At the bottom, show data table

# If I want to show the selected region, I may have to store the label number
# for each cell in the cellInfo.