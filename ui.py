import os
import pathlib
import numpy as np
import datetime as dt
import dash
import plotly.express as px
from dash import dcc
from dash import html

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
from ui_datafinder import datasets, infokeeper, availablemodes

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

dataset = datasets()
available = availablemodes()

info = infokeeper()
info.set_model("Basic_nn")
info.set_baseline(0)
info.set_period(0)

all_sets = available.get_keys()
periods = available.return_periods(all_sets[0])
models = available.return_models(all_sets[0])

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Energy Consumption Predictor"

# server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Energy Consumption Forecaster", className="app__header__title"),
                        html.P(
                            "This app predicts the future consumption of energy for individual households using a variety of ML models.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
            ],
            className="app__header",
        ),

        html.Div(
            [
                dash.dcc.Dropdown(id="dataset",
                        options= all_sets,
                        multi=False,
                        value=all_sets[0],
                        style={'width': "100%"},
                        placeholder='Select an available dataset'
                ),
                dash.dcc.Dropdown(id="model",
                        options=models,
                        multi=False,
                        value=models[0],
                        style={'width': "100%"},
                        placeholder='Select Forecasting Period'
                ),
                dash.dcc.Dropdown(id="baseline",
                        options=[
                            {"label": "On", "value": 1},
                            {"label": "Off", "value": 0}],
                        multi=False,
                        value=0,
                        style={'width': "100%"},
                        placeholder='Enable Baseline'
                ),
                dash.dcc.Dropdown(id="period",
                        options=periods,
                        multi=False,
                        value=periods[0],
                        style={'width': "100%"},
                        placeholder='Select Forecasting Period'
                )
            ],
            style={'display': 'inline-block'},
            className="app__buttons",
        ),

        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("Energy Forecast (kW)", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="energy-forecast",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                        dcc.Interval(
                            id="call-update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    ],
                    className="two-thirds column energy__forecast__container",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Error Distribution",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Slider(
                                            id="bin-slider",
                                            min=1,
                                            max=60,
                                            step=1,
                                            value=20,
                                            updatemode="drag",
                                            marks={
                                                20: {"label": "20"},
                                                40: {"label": "40"},
                                                60: {"label": "60"},
                                            },
                                        )
                                    ],
                                    className="slider",
                                ),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="bin-auto",
                                            options=[
                                                {"label": "Auto", "value": "Auto"}
                                            ],
                                            value=["Auto"],
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                        ),
                                        html.P(
                                            "# of Bins: Auto",
                                            id="bin-size",
                                            className="auto__p",
                                        ),
                                    ],
                                    className="auto__container",
                                ),
                                dcc.Graph(
                                    id="error-distribution",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # wind direction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Performance Characteristics", className="graph__title"
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="performance-characteristics",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(
    Output("energy-forecast", "figure"), 
    [Input('dataset', 'value'),
     Input('model', 'value'),
     Input('period', 'value'),
     Input('baseline', 'value')]
)
def gen_energy_forecast(set_name, model, baseline, period):
    """
    Generate the wind speed graph.
    :params interval: update the graph based on an interval
    """

    df = dataset.get_performance_data(set_name, model, baseline, period)

    # Have not implemented the baseline model into this yet
    trace = dict(
        type="scatter",
        y=df[model],
        x=df['Date'],
        line={"color": "#42C4F7"},
        hoverinfo="skip",
        mode="lines",
    )

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=700,
        xaxis={
            "range": [min(df['Date']), max(df["Date"])],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "nticks": 10,
            "title": "Date",
        },
        yaxis={
            "range": [
                min(df[model]),
                max(df[model]),
            ],
            "showgrid": True,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
            "nticks": max(6, round(df[model].iloc[-1] / 10)),
        },
    )

    return dict(data=[trace], layout=layout)


@app.callback(
    Output("performance-characteristics", "figure"), 
    [Input('dataset', 'value'),
     Input('model', 'value'),
     Input('period', 'value'),
     Input('baseline', 'value')]
)
def gen_metrics(set_name, model, baseline, period):
    """
    Generate the wind direction graph.
    :params interval: update the graph based on an interval
    """

    df = dataset.get_metrics_data(set_name, model, baseline, period)

    data = [
        dict(
            type="scatterpolar",
            r=df["Value"],
            theta=df["Metric"],
            color=df["Model"],
            mode="lines",
            fill="toself",
            line={"color": "rgba(32, 32, 32, .6)", "width": 1},
        )
    ]

    layout = dict(
        height=350,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        autosize=False,
        polar={
            "bgcolor": app_color["graph_line"],
        },
        showlegend=False,
    )

    return dict(data=data, layout=layout)


@app.callback(
    Output("error-distribution", "figure"),
    [Input('dataset', 'value'),
     Input('model', 'value'),
     Input('period', 'value'),
     Input('baseline', 'value')],
    [
        State("energy-forecast", "figure"),
        State("bin-slider", "value"),
        State("bin-auto", "value"),
    ],
)
def gen_error_histogram(set_name, model, period, baseline, energy_forecast_figure, slider_value, auto_state):
    """
    Genererate wind histogram graph.
    :params interval: upadte the graph based on an interval
    :params energy_forecast_figure: current wind speed graph
    :params slider_value: current slider value
    :params auto_state: current auto state
    """

    df = dataset.get_performance_data(set_name, model, baseline, period)
    error_val = []

    try:
        error_val = df["Actual"] - df[model]
        if "Auto" in auto_state:
            bin_val = np.histogram(
                error_val,
                bins=range(int(round(min(error_val))), int(round(max(error_val)))),
            )
        else:
            bin_val = np.histogram(error_val, bins=slider_value)
    except Exception as error:
        raise PreventUpdate

    avg_val = float(sum(error_val)) / len(error_val)
    median_val = np.median(error_val)

    pdf_fitted = rayleigh.pdf(
        bin_val[1], loc=(avg_val) * 0.55, scale=(bin_val[1][-1] - bin_val[1][0]) / 3
    )

    y_val = (pdf_fitted * max(bin_val[0]) * 20,)
    y_val_max = max(y_val[0])
    bin_val_max = max(bin_val[0])

    trace = dict(
        type="bar",
        x=bin_val[1],
        y=bin_val[0],
        marker={"color": app_color["graph_line"]},
        showlegend=False,
        hoverinfo="x+y",
    )

    traces_scatter = [
        {"line_dash": "dash", "line_color": "#2E5266", "name": "Average"},
        {"line_dash": "dot", "line_color": "#BD9391", "name": "Median"},
    ]

    scatter_data = [
        dict(
            type="scatter",
            x=[bin_val[int(len(bin_val) / 2)]],
            y=[0],
            mode="lines",
            line={"dash": traces["line_dash"], "color": traces["line_color"]},
            marker={"opacity": 0},
            visible=True,
            name=traces["name"],
        )
        for traces in traces_scatter
    ]

    trace3 = dict(
        type="scatter",
        mode="lines",
        line={"color": "#42C4F7"},
        y=y_val[0],
        x=bin_val[1][: len(bin_val[1])],
        name="Rayleigh Fit",
    )
    layout = dict(
        height=350,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        xaxis={
            "title": "Error",
            "showgrid": False,
            "showline": False,
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "title": "Number of Samples",
            "fixedrange": True,
        },
        autosize=True,
        bargap=0.01,
        bargroupgap=0,
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "xanchor": "center",
            "y": 1,
            "x": 0.5,
        },
        shapes=[
            {
                "xref": "x",
                "yref": "y",
                "y1": int(max(bin_val_max, y_val_max)) + 0.5,
                "y0": 0,
                "x0": avg_val,
                "x1": avg_val,
                "type": "line",
                "line": {"dash": "dash", "color": "#2E5266", "width": 5},
            },
            {
                "xref": "x",
                "yref": "y",
                "y1": int(max(bin_val_max, y_val_max)) + 0.5,
                "y0": 0,
                "x0": median_val,
                "x1": median_val,
                "type": "line",
                "line": {"dash": "dot", "color": "#BD9391", "width": 5},
            },
        ],
    )
    return dict(data=[trace, scatter_data[0], scatter_data[1], trace3], layout=layout)


@app.callback(
    Output("bin-auto", "value"),
    [Input("bin-slider", "value")],
    [State("energy-forecast", "figure")],
)
def deselect_auto(slider_value, energy_forecast_figure):
    """ Toggle the auto checkbox. """

    # prevent update if graph has no data
    if "data" not in energy_forecast_figure:
        raise PreventUpdate
    if not len(energy_forecast_figure["data"]):
        raise PreventUpdate

    if energy_forecast_figure is not None and len(energy_forecast_figure["data"][0]["y"]) > 5:
        return [""]
    return ["Auto"]


@app.callback(
    Output('model', 'options'),
    Output('model', 'value'),
    Input('dataset', 'value')
)
def update_model_options(dataset):
    
    options = []
    models = available.return_models(dataset)
    for model in models[:1]:
        option = {'labels': model, 'value': model}
        options.append(option)
    
    value = models[0]

    return options, value


@app.callback(
    Output('period', 'options'),
    Output('period', 'value'),
    Input('dataset', 'value')
)
def update_period_options(dataset):
    
    options = []
    periods = available.return_periods(dataset)
    for period in periods[:1]:
        option = {'labels': str(period), 'value': period}
        options.append(option)
    
    value = periods[0]

    return options, value


@app.callback(
    Output('baseline', 'options'),
    Output('baseline', 'value'),
    Input('dataset', 'value')
)
def update_baseline_options(dataset):

    options = []
    values = available.return_dataset(dataset)
    models = values.get('Models')
    if 'Baseline' in models:
        options = [{'label': 'On', 'value': 1},
                   {'label': 'Off', 'value': 0}]
    else:
        options = [{'label': 'Off', 'value': 0}]
    
    value = 0

    return options, value


@app.callback(
    Output("bin-size", "children"),
    [Input("bin-auto", "value")],
    [State("bin-slider", "value")],
)
def show_num_bins(autoValue, slider_value):
    """ Display the number of bins. """

    if "Auto" in autoValue:
        return "# of Bins: Auto"
    return "# of Bins: " + str(int(slider_value))


if __name__ == "__main__":
    app.run_server(debug=True)