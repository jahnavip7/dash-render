# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZUm9k7Z2uQL_n4aUBgUrnmBMI97BP5MJ
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime

# Load all model CSVs (Assuming all CSVs are in the same folder)
models_dir = "model_data\"

# Check if directory exists
if not os.path.exists(models_dir):
    raise FileNotFoundError(f"Directory not found: {models_dir}")

model_files = [f for f in os.listdir(models_dir) if f.endswith(".csv")]

if not model_files:
    raise ValueError(
        f"No CSV files found in {models_dir}. Please add your model CSV files to this directory.")

# Dictionary to store all model data
models_data = {}

# Process each model file with interpolation
for file in model_files:
    model_name = file.replace(".csv", "")  # Extract model name
    df = pd.read_csv(os.path.join(models_dir, file))
    df["dates"] = pd.to_datetime(df["dates"], format='%m/%d/%y')

    # Sort by date
    df = df.sort_values("dates")

    # Create continuous date range
    full_date_range = pd.date_range(
        start=df["dates"].min(), end=df["dates"].max(), freq="D")
    df_full = pd.DataFrame({"dates": full_date_range})

    # Merge with full date range and interpolate
    df_interpolated = pd.merge(df_full, df, on="dates", how="left")
    if "groundtruth" in df.columns:
        df_interpolated["groundtruth"] = df_interpolated["groundtruth"].interpolate(
            method="linear")
    if "predictions" in df.columns:
        df_interpolated["predictions"] = df_interpolated["predictions"].interpolate(
            method="linear")
    elif "predicted values" in df.columns:
        df_interpolated["predicted values"] = df_interpolated["predicted values"].interpolate(
            method="linear")

    models_data[model_name] = df_interpolated

# Get min/max dates and date list for slider
all_dates = pd.concat([df["dates"] for df in models_data.values()])
min_date = all_dates.min()
max_date = all_dates.max()
date_list = list(pd.date_range(start=min_date, end=max_date, freq="D"))

# Initialize Dash App
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Time-Series Forecast Dashboard",
            style={'textAlign': 'center', 'marginBottom': 30}),

    # Main content container
    html.Div([
        # Sidebar for Model Selection
        html.Div([
            html.H3("Select Models:", style={'marginBottom': 20}),
            dcc.Checklist(
                id="model-selection",
                options=[{"label": model, "value": model}
                         for model in models_data.keys()],
                value=[],  # Default empty
                inline=False,
                style={'fontSize': '16px', 'lineHeight': '2'}
            )
        ], style={
            "width": "20%",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "10px",
            "marginRight": "20px"
        }),

        # Graph Area
        html.Div([
            dcc.Graph(
                id="time-series-graph",
                style={'height': '600px'}
            ),
            html.Div([
                html.Label("Adjust Date Range", style={'marginTop': '20px'}),
                dcc.RangeSlider(
                    id="date-range-slider",
                    min=0,
                    max=len(date_list) - 1,
                    step=1,
                    value=[0, len(date_list) - 1],
                    marks={
                        0: {'label': min_date.strftime('%Y-%m-%d')},
                        len(date_list) - 1: {'label': max_date.strftime('%Y-%m-%d')}
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginTop': '20px', 'padding': '20px'})
        ], style={"width": "75%"})
    ], style={"display": "flex", "margin": "20px"})
])


@app.callback(
    Output("time-series-graph", "figure"),
    [Input("model-selection", "value"),
     Input("date-range-slider", "value")]
)
def update_graph(selected_models, slider_range):
    """Update graph dynamically based on selected models and date range."""
    try:
        # Create Figure with white background
        fig = go.Figure()

        # Convert slider indices to dates
        start_date = date_list[slider_range[0]]
        end_date = date_list[slider_range[1]]

        # Plot ground truth only once
        ground_truth_plotted = False

        # Find y-axis range across all selected data
        all_values = []

        # Add each selected model's data to the graph
        for model in selected_models:
            df = models_data[model]
            df_filtered = df[(df["dates"] >= start_date)
                             & (df["dates"] <= end_date)]

            # Add ground truth only once
            if not ground_truth_plotted and "groundtruth" in df_filtered.columns:
                all_values.extend(df_filtered["groundtruth"].dropna().tolist())
                fig.add_trace(go.Scatter(
                    x=df_filtered["dates"],
                    y=df_filtered["groundtruth"],
                    mode="lines",
                    name="Actual Values",
                    line=dict(color='black', width=2, shape='spline')
                ))
                ground_truth_plotted = True

            # Add predicted values - handle both possible column names
            pred_col = None
            if "predictions" in df_filtered.columns:
                pred_col = "predictions"
            elif "predicted values" in df_filtered.columns:
                pred_col = "predicted values"

            if pred_col:
                # Clean up model name for display
                display_name = model.replace(
                    "results-csv_", "").replace("result-csv_", "")
                all_values.extend(df_filtered[pred_col].dropna().tolist())
                fig.add_trace(go.Scatter(
                    x=df_filtered["dates"],
                    y=df_filtered[pred_col],
                    mode="lines",
                    name=f"{display_name}",  # Just show the model name
                    line=dict(dash='dash', width=2, shape='spline')
                ))

        # Calculate y-axis range with fixed increments
        if all_values:
            y_min = 0
            y_max = ((max(all_values) // 2000) + 1) * 2000

            # Update Layout with Fixed Y-Axis
            fig.update_layout(
                title={
                    'text': f"Smooth Time-Series Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    font=dict(size=12)
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                showlegend=True,
                plot_bgcolor='white',
                yaxis=dict(
                    range=[y_min, y_max],
                    tickmode="linear",
                    dtick=2000  # Y-axis increments by 2000
                )
            )

            # Add grid lines for better readability
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        return fig

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        fig = go.Figure()
        fig.update_layout(
            title={
                'text': f"Error loading data: {str(e)}",
                'x': 0.5,
                'xanchor': 'center'
            }
        )
        return fig


# Run App
if __name__ == "__main__":
    app.run(debug=True)
