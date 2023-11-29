import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
from folium.plugins import MarkerCluster
from folium import Circle, Marker
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Original code for loading and creating DataFrame
data = pd.read_csv('export.csv', header=None)
data.columns = ['Longitude', 'Latitude', 'Neighbourhood', 'Timestamp', 'Day', 'Hour']

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract the relevant columns for clustering (latitude and longitude)
coordinates = data[['Longitude', 'Latitude']]

# Perform k-means clustering
kmeans = KMeans(n_clusters=50)  # You can change the number of clusters
clusters = kmeans.fit_predict(coordinates)

# Add cluster labels to your dataframe
data['Cluster'] = clusters

app = dash.Dash(__name__)
# Create a Dash web application
app.layout = html.Div([
    html.H1("Toronto Cluster Map Dashboard"),

    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=data['Timestamp'].min(),
        end_date=data['Timestamp'].max(),
        display_format='YYYY-MM-DD',
        style={'margin': '10px'}
    ),

    html.Div(id='map-container')
])

# Callback to update the map based on the selected date range
@app.callback(
    Output('map-container', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_map(start_date, end_date):
    filtered_data = data[(data['Timestamp'] >= start_date) & (data['Timestamp'] <= end_date)]

    # Create a Folium map centered at Toronto
    m = folium.Map(location=[43.70, -79.42], zoom_start=11)

    # Loop through clusters and add a Circle and Marker for each cluster
    for cluster_id in filtered_data['Cluster'].unique():
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_id]
        
        # Calculate the centroid of the cluster
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()

        # Calculate the radius of the circle based on the cluster size
        radius = np.sqrt(len(cluster_data)) * 50  # Adjust the scaling factor as needed

        # Draw a circle around the centroid to represent the size of the cluster
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)

        # Add a Marker showing the number of points in the cluster
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m)

    # Convert the Folium map to HTML
    map_html = m._repr_html_()

    return html.Iframe(srcDoc=map_html, width='100%', height='600px')

# Run the web application
if __name__ == '__main__':
    app.run_server(debug=True)
