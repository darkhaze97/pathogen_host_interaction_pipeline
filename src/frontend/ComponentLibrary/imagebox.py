from dash import html
import plotly.express as px

def imagebox(data):
    
    return html.Img(
        src=data,
        style={
            'width': '100%',
            'height': '50%',
            'backgroundColor': 'red',
            'backgroundSize': 'contain',
            'aspectRatio': '1280/720'
        }
    )