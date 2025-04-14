import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime

# Load all model CSVs (Assuming all CSVs are in the same folder)
models_dir = "/Users/jahnavipb/Downloads/VISA DOCS-PARENTS/forecast_dashboard/model_data"

# Check if directory exists
if not os.path.exists(models_dir):
    raise FileNotFoundError(f"Directory not found: {models_dir}")

model_files = [f for f in os.listdir(models_dir) if f.endswith(".csv")]

if not model_files:
    raise ValueError(
        f"No CSV files found in {models_dir}. Please add your model CSV files to this directory.")

# Dictionary to store all model data
models_data = {}

# Load and sort model files
for file in model_files:
    model_name = file.replace(".csv", "")  # Extract model name
    df = pd.read_csv(os.path.join(models_dir, file))
    df["dates"] = pd.to_datetime(df["dates"], format='%m/%d/%y')
    df = df.sort_values("dates")  # Keep original weekly frequency
    models_data[model_name] = df

# Get min/max dates and date list for slider
all_dates = pd.concat([df["dates"] for df in models_data.values()])
min_date = all_dates.min()
max_date = all_dates.max()
date_list = list(pd.date_range(start=min_date, end=max_date, freq="W"))

# Initialize Dash App
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Time-Series Forecast Dashboard",
            style={'textAlign': 'center', 'marginBottom': 30}),

    html.Div([
        html.Div([
            html.H3("Select Models:", style={'marginBottom': 20}),
            dcc.Checklist(
                id="model-selection",
                options=[{
                "label": model.replace("2results_v14_", "").replace("results-csv_", "").replace("result-csv_", ""),
                "value": model
                    } for model in models_data.keys()],
                value=[],
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

        html.Div([
            dcc.Graph(id="time-series-graph", style={'height': '600px'}),
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
    try:
        fig = go.Figure()
        start_date = date_list[slider_range[0]]
        end_date = date_list[slider_range[1]]
        ground_truth_plotted = False
        all_values = []

        for model in selected_models:
            df = models_data[model]
            df_filtered = df[(df["dates"] >= start_date)
                             & (df["dates"] <= end_date)]

            if not ground_truth_plotted and "ground_truth" in df_filtered.columns:
                all_values.extend(df_filtered["ground_truth"].dropna().tolist())
                fig.add_trace(go.Scatter(
                    x=df_filtered["dates"],
                    y=df_filtered["ground_truth"],
                    mode="lines",
                    name="Actual Values",
                    line=dict(color='black', width=2, shape='spline')
                ))
                ground_truth_plotted = True

            pred_col = "predictions" if "predictions" in df_filtered.columns else (
                "predicted values" if "predicted values" in df_filtered.columns else None)

            if pred_col:
                display_name = model.replace("2results_v14_", "").replace("results-csv_", "").replace("result-csv_", "")

                all_values.extend(df_filtered[pred_col].dropna().tolist())
                fig.add_trace(go.Scatter(
                    x=df_filtered["dates"],
                    y=df_filtered[pred_col],
                    mode="lines",
                    name=f"{display_name}",
                    line=dict(dash='dash', width=2, shape='spline')
                ))

        if all_values:
            y_min = 0
            y_max = ((max(all_values) // 2000) + 1) * 2000
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
                    dtick=2000
                )
            )
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
print("Loaded models:", model_files)
print("Available models for dashboard:", models_data.keys())


# Run App
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True)
