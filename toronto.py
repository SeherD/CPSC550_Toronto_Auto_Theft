import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import csv
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
from sklearn.cluster import KMeans

data = pd.read_csv('export.csv', header=None)
data.columns = ['Longitude', 'Latitude', 'Neighbourhood', 'Timestamp', 'Day', 'Hour']
df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

coordinates = data[['Longitude', 'Latitude']]


# Spark data analysis dataframes
neighbourhood_frequencies = pd.read_csv('spark_data/neighbourhood_freq.csv')
weekday_frequencies = pd.read_csv('spark_data/weekday_freq.csv')

neighbourhood_fig = px.bar(neighbourhood_frequencies, x='NEIGHBOURHOOD_158', y='count', title="Theft frequency per neighbourhood")
neighbourhood_fig.update_layout(xaxis_title="Neighbourhood", yaxis_title="Theft count")

weekday_fig = px.bar(weekday_frequencies, x='OCC_DOW', y='count', title="Theft frequency per weekday")
weekday_fig.update_layout(xaxis_title="Weekday", yaxis_title="Theft count")

def get_prophet_plotlys():
    with open('spark_data/model.json') as f:
        prophet_model = model_from_json(f.read())
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)
    return plot_plotly(prophet_model, forecast), plot_components_plotly(prophet_model, forecast)

prophet_trend, prophet_components = get_prophet_plotlys()

app = dash.Dash(__name__)
# Create a Dash web application
app.layout = html.Div([
    html.H1("Toronto Theft Data Analytics Dashboard"),
  
    html.H1("Theft Frequencies"),
    html.Div([
        dcc.Graph(figure=neighbourhood_fig),
        dcc.Graph(figure=weekday_fig)
    ]),
    html.H1("Theft Predictions"),
    html.Div([
        dcc.Graph(figure=prophet_trend),
        dcc.Graph(figure=prophet_components)
    ])
])




# Run the web application
if __name__ == '__main__':
    app.run_server(debug=True)
