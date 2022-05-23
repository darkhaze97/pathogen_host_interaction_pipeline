from dash import html, dcc
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np

from .imagebox import imagebox

def mainview(imgNameMap):
    data = imgNameMap[(list(imgNameMap.keys()))[0]][1]
    return html.Div(
        [
            imagebox(data),
            filter_seg()
        ],
        style={
            'margin-left': '18%',
            'width': '82%',
            'height': '100%'
        }
    )
    
def filter_seg():
    return html.Div(
        [
            dbc.RadioItems(
                id="main-img-filter",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options = 
                    [
                        {'label': 'Show all labels', 'value': 'show-all'},
                        {'label': 'Show selected labels', 'value': 'show-selected'}
                    ],
                value = '',
           )
        ],
        className="radio-group"
    )
    
    
# Checkboxes
# Able to select: Show all segmentation labels
# Able to select: Show segmentation label of selected cell

# Then next to it, show the enlarged image of the cell, highlight
# the row of the data table that represents the data for this cell.
#

# At the bottom, show data table