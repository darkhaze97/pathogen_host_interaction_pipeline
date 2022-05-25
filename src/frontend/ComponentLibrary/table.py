from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

def obtain_table(cellInfo):
    
    return html.Div(
        [
            html.H5(
                "Cell Information",
                style={
                    'height': '10%'
                }
            ),
            generate_cell_table(cellInfo)
        ],
        style={
            'height': '27%',
            'position': 'relative'
        }
    ) 

def generate_cell_table(cellInfo):
    blackList = ['bounding_box', 'image']
    colNames = [colName for colName in cellInfo if (colName not in blackList)]
    colNames = [colName for colName in cellInfo if (colName not in blackList)]
    
    tableHeader = [
        html.Thead(html.Tr(
        [html.Th(colName) for colName in colNames]))
    ]
    
    tableBody = [
        html.Tbody(generate_rows(colNames, cellInfo))
    ]
    return  html.Div(
                dbc.Table(
                    tableHeader + tableBody,
                    bordered=True,
                    style={
                        'position': 'relative'
                    }
                ),
                style={
                    'overflow-y': 'scroll',
                    'height': '90%'
                }
            )

def generate_rows(colNames, cellInfo):
    # Later, I will add an id to the columns.
    return [html.Tr([html.Td(cellInfo[col][i]) for col in colNames]) 
            for i in range(len(cellInfo['area']))]