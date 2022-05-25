from dash import html, dcc
import dash_bootstrap_components as dbc

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
                        {'label': 'No labels', 'value': 'no-labels'},
                        {'label': 'Show all labels', 'value': 'show-all'},
                        {'label': 'Show selected labels', 'value': 'show-selected'}
                    ],
                value = 'no-labels',
           )
        ],
        className="radio-group"
    )