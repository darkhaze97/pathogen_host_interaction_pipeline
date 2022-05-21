from dash import html
import dash_bootstrap_components as dbc

def sidebar(info):
    # Extract the names of each image.
    imgNames = [(cellInfo[2].split('/'))[-1] for cellInfo in info['cellImages']]
    
    # @app.callback() --> This is to update the centre image.
    
    return html.Div(
        dbc.Nav(
            [dbc.NavLink(name, href=f"/{name}", active="exact", id=name)
            for name in imgNames],
            vertical=True,
            pills=True
        ),
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