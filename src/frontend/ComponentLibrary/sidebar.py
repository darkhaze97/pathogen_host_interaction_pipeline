from dash import html, Output, Input
import dash_bootstrap_components as dbc

def sidebar(imgNameMap):
    # Extract the names of each image.
    imgNames = [name for name in imgNameMap.keys()]
    
    return html.Div(
        [
            dbc.RadioItems(
                id="img-select",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[{'label': name, 'value': name} for name in imgNames],
                value=imgNames[0]
            )
        ],
        style={
            'backgroundColor': 'lightblue',
            'top': 0,
            'left': 0,
            'width': '18%',
            'height': '100%',
            'position': 'fixed',
            'padding': '2px 1px'
        },
    )