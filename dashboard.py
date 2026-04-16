import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv('Data/merged_output.csv')

# =============================================================================
# DATA PREP
# =============================================================================

# Top 10 Machines with Most Errors
top10_machines = df.dropna(subset=['errors_machineID'])
top10_machines = top10_machines.groupby('errors_machineID').size().reset_index(name='count')
top10_machines = top10_machines.nlargest(10, 'count')
top10_machines['errors_machineID'] = top10_machines['errors_machineID'].astype(int).astype(str)

# Error Frequency
error_freq = df['errors_errorID'].value_counts().reset_index()
error_freq.columns = ['Error Type', 'Count']

# Failure Types
failure_freq = df['failures_failure'].value_counts().reset_index()
failure_freq.columns = ['Failure Type', 'Count']

# Machine Models Distribution
model_dist = df.drop_duplicates(subset=['machines_machineID'])[['machines_machineID', 'machines_model', 'machines_age']]
model_counts = model_dist['machines_model'].value_counts().reset_index()
model_counts.columns = ['Model', 'Count']

# Age vs Cumulative Failures
age_failures = df.dropna(subset=['machines_age', 'failures_features_cumulative_failures'])
age_failures = age_failures.drop_duplicates(subset=['machines_machineID'])[
    ['machines_machineID', 'machines_age', 'failures_features_cumulative_failures', 'machines_model']
]

# Days Since Last Failure (Top 10 at risk)
at_risk = df.dropna(subset=['failures_features_days_since_last_failure'])
at_risk = at_risk.groupby('errors_machineID')['failures_features_days_since_last_failure'].mean().reset_index()
at_risk.columns = ['Machine ID', 'Avg Days Since Last Failure']
at_risk['Machine ID'] = at_risk['Machine ID'].astype(int).astype(str)
at_risk = at_risk.nsmallest(10, 'Avg Days Since Last Failure')

# Telemetry Averages per Machine
telemetry_avg = df.groupby('telemetry_machineID')[
    ['telemetry_volt', 'telemetry_rotate', 'telemetry_pressure', 'telemetry_vibration']
].mean().reset_index()
telemetry_avg.columns = ['Machine ID', 'Avg Volt', 'Avg Rotate', 'Avg Pressure', 'Avg Vibration']

# Errors by Hour
df['errors_hour'] = pd.to_datetime(df['errors_datetime'], errors='coerce').dt.hour
errors_by_hour = df.dropna(subset=['errors_hour'])
errors_by_hour = errors_by_hour.groupby('errors_hour').size().reset_index(name='Count')
errors_by_hour.columns = ['Hour', 'Count']

# Errors by Month
df['errors_month'] = pd.to_datetime(df['errors_datetime'], errors='coerce').dt.month
errors_by_month = df.dropna(subset=['errors_month'])
errors_by_month = errors_by_month.groupby('errors_month').size().reset_index(name='Count')
errors_by_month.columns = ['Month', 'Count']
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
errors_by_month['Month'] = errors_by_month['Month'].map(month_names)

# Errors by Day of Week
errors_by_day = df.dropna(subset=['failures_features_dayofweek'])
errors_by_day = errors_by_day.groupby('failures_features_dayofweek').size().reset_index(name='Count')
errors_by_day.columns = ['Day', 'Count']
day_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
errors_by_day['Day'] = errors_by_day['Day'].map(day_names)

# Maintenance per Component
maint_comp = df['maint_comp'].value_counts().reset_index()
maint_comp.columns = ['Component', 'Count']

# =============================================================================
# COLORS
# =============================================================================
COLORS = {
    "background": "#F4F6F9",
    "card":       "#FFFFFF",
    "primary":    "#2C7BE5",
    "secondary":  "#6E84A3",
    "success":    "#00D97E",
    "danger":     "#E63757",
    "warning":    "#F6C343",
    "text":       "#12263F",
    "subtext":    "#95AAC9",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "24px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
}

# =============================================================================
# APP
# =============================================================================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": COLORS["background"], "padding": "24px", "fontFamily": "Arial"},
    children=[

        # ── TITLE ──
        html.H1("🏭 Smart Factory Monitoring Dashboard",
                style={"color": COLORS["text"], "textAlign": "center", "marginBottom": "24px"}),

        # ── SECTION: OVERVIEW ──
        html.H2("📊 Overview", style={"color": COLORS["text"]}),

        # KPI Cards
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
            children=[
                html.Div([
                    html.H4("Total Machines", style={"color": COLORS["subtext"]}),
                    html.H2(str(int(df['errors_machineID'].nunique())), style={"color": COLORS["text"]})
                ], style={**CARD_STYLE, "flex": "1", "textAlign": "center"}),

                html.Div([
                    html.H4("Total Errors", style={"color": COLORS["subtext"]}),
                    html.H2(str(df['errors_errorID'].notna().sum()), style={"color": COLORS["danger"]})
                ], style={**CARD_STYLE, "flex": "1", "textAlign": "center"}),

                html.Div([
                    html.H4("Total Failures", style={"color": COLORS["subtext"]}),
                    html.H2(str(df['failures_failure'].notna().sum()), style={"color": COLORS["warning"]})
                ], style={**CARD_STYLE, "flex": "1", "textAlign": "center"}),

                html.Div([
                    html.H4("Avg Cumulative Failures", style={"color": COLORS["subtext"]}),
                    html.H2(str(round(df['failures_features_cumulative_failures'].mean(), 1)),
                            style={"color": COLORS["primary"]})
                ], style={**CARD_STYLE, "flex": "1", "textAlign": "center"}),
            ]
        ),

        # ── SECTION: ERRORS ANALYSIS ──
        html.H2("⚠️ Errors Analysis", style={"color": COLORS["text"]}),

        # Error Frequency + Top 10 Machines
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
            children=[
                html.Div([
                    dcc.Graph(figure=px.bar(
                        error_freq, x='Error Type', y='Count',
                        title='Error Frequency by Type',
                        color_discrete_sequence=[COLORS["primary"]]
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),

                html.Div([
                    dcc.Graph(figure=px.bar(
                        top10_machines, x='errors_machineID', y='count',
                        title='Top 10 Machines with Most Errors',
                        color_discrete_sequence=[COLORS["danger"]],
                        labels={'errors_machineID': 'Machine ID', 'count': 'Error Count'},
                        category_orders={'errors_machineID': top10_machines['errors_machineID'].tolist()}
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),
            ]
        ),

        # Errors by Month + Hour
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
            children=[
                html.Div([
                    dcc.Graph(figure=px.bar(
                        errors_by_month, x='Month', y='Count',
                        title='Errors by Month',
                        color_discrete_sequence=[COLORS["primary"]],
                        category_orders={'Month': ['Jan','Feb','Mar','Apr','May','Jun',
                                                   'Jul','Aug','Sep','Oct','Nov','Dec']}
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),

                html.Div([
                    dcc.Graph(figure=px.bar(
                        errors_by_hour, x='Hour', y='Count',
                        title='Errors by Hour of Day',
                        color_discrete_sequence=[COLORS["secondary"]]
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),
            ]
        ),

        # Errors by Day of Week
        html.Div(style=CARD_STYLE, children=[
            dcc.Graph(figure=px.bar(
                errors_by_day, x='Day', y='Count',
                title='Errors by Day of Week',
                color_discrete_sequence=[COLORS["warning"]],
                category_orders={'Day': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}
            ))
        ]),

        # ── SECTION: FAILURES ANALYSIS ──
        html.H2("🔴 Failures Analysis", style={"color": COLORS["text"]}),

        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
            children=[
                html.Div([
                    dcc.Graph(figure=px.pie(
                        failure_freq, names='Failure Type', values='Count',
                        title='Failure Types Distribution',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),

                html.Div([
                    dcc.Graph(figure=px.bar(
                        maint_comp, x='Component', y='Count',
                        title='Maintenance per Component',
                        color_discrete_sequence=[COLORS["success"]]
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),
            ]
        ),

        # ── SECTION: MACHINE STATUS ──
        html.H2("⚙️ Machine Status", style={"color": COLORS["text"]}),

        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
            children=[
                html.Div([
                    dcc.Graph(figure=px.pie(
                        model_counts, names='Model', values='Count',
                        title='Machine Models Distribution',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),

                html.Div([
                    dcc.Graph(figure=px.scatter(
                        age_failures, x='machines_age', y='failures_features_cumulative_failures',
                        color='machines_model',
                        title='Machine Age vs Cumulative Failures',
                        labels={'machines_age': 'Machine Age (years)',
                                'failures_features_cumulative_failures': 'Cumulative Failures'},
                        color_discrete_sequence=px.colors.qualitative.Bold
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),
            ]
        ),

        # ── SECTION: PREDICTIVE MAINTENANCE ──
        html.H2("🔮 Predictive Maintenance", style={"color": COLORS["text"]}),

        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
            children=[
                html.Div([
                    dcc.Graph(figure=px.bar(
                        at_risk, x='Machine ID', y='Avg Days Since Last Failure',
                        title='Top 10 Machines At Risk (Least Days Since Last Failure)',
                        color_discrete_sequence=[COLORS["danger"]],
                        category_orders={'Machine ID': at_risk['Machine ID'].tolist()}
                    ))
                ], style={**CARD_STYLE, "flex": "1"}),
            ]
        ),

        # ── SECTION: TELEMETRY ──
        html.H2("📡 Telemetry", style={"color": COLORS["text"]}),

        # Telemetry Dropdown
        html.Div(style=CARD_STYLE, children=[
            html.Label("Select Telemetry Metric:", style={"color": COLORS["text"]}),
            dcc.Dropdown(
                id='telemetry-dropdown',
                options=[
                    {'label': 'Voltage',   'value': 'Avg Volt'},
                    {'label': 'Rotation',  'value': 'Avg Rotate'},
                    {'label': 'Pressure',  'value': 'Avg Pressure'},
                    {'label': 'Vibration', 'value': 'Avg Vibration'},
                ],
                value='Avg Volt',
                clearable=False,
                style={"width": "300px", "marginBottom": "16px"}
            ),
            dcc.Graph(id='telemetry-chart')
        ]),

        # Footer
        html.P("Smart Factory Monitoring Platform — DEPI Final Project",
               style={"color": COLORS["subtext"], "textAlign": "center", "marginTop": "24px"})
    ]
)

# =============================================================================
# CALLBACK: TELEMETRY CHART
# =============================================================================
@app.callback(
    Output('telemetry-chart', 'figure'),
    Input('telemetry-dropdown', 'value')
)
def update_telemetry(metric):
    top_telemetry = telemetry_avg.nlargest(20, metric)
    top_telemetry['Machine ID'] = top_telemetry['Machine ID'].astype(int).astype(str)
    fig = px.bar(
        top_telemetry,
        x='Machine ID', y=metric,
        title=f'Top 20 Machines — {metric}',
        color_discrete_sequence=[COLORS["primary"]],
        category_orders={'Machine ID': top_telemetry['Machine ID'].tolist()}
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)