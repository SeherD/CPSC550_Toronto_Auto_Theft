import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
from folium import Circle, Marker
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def choose_optimal_num_clusters(coordinates):
    return optimal_num_clusters

# Original code for loading and creating DataFrame
data = pd.read_csv('export.csv', header=None)
data.columns = ['Longitude', 'Latitude', 'Neighbourhood', 'Timestamp', 'Day', 'Hour']

#List of neighbhorhoods in Toronto
neighborhood_options = sorted(
    [{'label': neighborhood, 'value': neighborhood} for neighborhood in data['Neighbourhood'].unique()],
    key=lambda x: x['label']
)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract the relevant columns for clustering (latitude and longitude)
coordinates = data[['Longitude', 'Latitude']]

# Uncomment the next line and comment out the elbow method code
optimal_num_clusters = 10
# optimal_num_clusters = choose_optimal_num_clusters(coordinates)

# Perform k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)  # Set the same random seed
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

    html.Div(id='map-container'),
    
    html.Div([
        dcc.Dropdown(
            id='neighborhood-dropdown',
            options=neighborhood_options,
            value=None,  # Default value
            placeholder="Select a neighborhood",
            style={'margin': '10px', 'width': '48%'}
        ),
        
        dcc.Dropdown(
            id='neighborhood-dropdown-2',
            options=neighborhood_options,
            value=None,  # Default value
            placeholder="Select another neighborhood",
            style={'margin': '10px', 'width': '48%'}
        ),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
    html.Div([
        html.Div(id='neighborhood-map-container', style={'display': 'inline-block', 'width': '50%'}),
        html.Div(id='neighborhood-map-container-2', style={'display': 'inline-block', 'width': '50%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),


])

# Callback to update the map based on the selected date range and top N clusters
@app.callback(
    Output('map-container', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_map(start_date, end_date):
    filtered_data = data[(data['Timestamp'] >= start_date) & (data['Timestamp'] <= end_date)]

    # Create a Folium map centered at Toronto
    m = folium.Map(location=[43.70, -79.42], zoom_start=10)

    # Loop through clusters and add a Circle and Marker for each cluster
    for cluster_id in filtered_data['Cluster'].unique():
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_id]

        # Calculate the centroid of the cluster
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()

        # Calculate the radius of the circle based on the cluster size
        radius = np.sqrt(len(cluster_data)) * 35  # Adjust the scaling factor as needed

        # Draw a circle around the centroid to represent the size of the cluster
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)

        # Add a Marker showing the number of points in the cluster
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m)

    # Convert the Folium map to HTML
    map_html = m._repr_html_()

    return html.Iframe(srcDoc=map_html, width='100%', height='300px')

@app.callback(
    Output('neighborhood-map-container', 'children'),
    [Input('neighborhood-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_neighborhood_map(selected_neighborhood, start_date, end_date):
    if not selected_neighborhood:
        return "Please select a neighborhood."

    neighborhood_data = data[(data['Neighbourhood'] == selected_neighborhood) & 
                             (data['Timestamp'] >= start_date) & 
                             (data['Timestamp'] <= end_date)]

    # Perform k-means clustering for the selected neighborhood data
    neighborhood_coordinates = neighborhood_data[['Longitude', 'Latitude']]
    kmeans_neighborhood = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    neighborhood_clusters = kmeans_neighborhood.fit_predict(neighborhood_coordinates)
    neighborhood_data['Cluster'] = neighborhood_clusters

    # Create a new Folium map for the selected neighborhood
    m_neighborhood = folium.Map(location=[43.70, -79.42], zoom_start=10)  # Adjust zoom as needed

    # Loop through clusters and add a Circle and Marker for each cluster
    for cluster_id in range(optimal_num_clusters):
        cluster_data = neighborhood_data[neighborhood_data['Cluster'] == cluster_id]

        if cluster_data.empty:
            continue

        # Calculate the centroid of the cluster
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()

        # Calculate the radius of the circle based on the cluster size
        radius = np.sqrt(len(cluster_data)) * 25  # Adjust the scaling factor as needed

        # Draw a circle around the centroid to represent the size of the cluster
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m_neighborhood)

        # Add a Marker showing the number of points in the cluster
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m_neighborhood)

    # Convert the Folium map to HTML
    map_html_neighborhood = m_neighborhood._repr_html_()

    return html.Iframe(srcDoc=map_html_neighborhood, width='100%', height='300px')

@app.callback(
    Output('neighborhood-map-container-2', 'children'),
    [Input('neighborhood-dropdown-2', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_neighborhood_map_2(selected_neighborhood, start_date, end_date):
    if not selected_neighborhood:
        return "Please select a second neighborhood for comparison."

    neighborhood_data = data[(data['Neighbourhood'] == selected_neighborhood) & 
                             (data['Timestamp'] >= start_date) & 
                             (data['Timestamp'] <= end_date)]

    # Perform k-means clustering for the selected neighborhood data
    neighborhood_coordinates = neighborhood_data[['Longitude', 'Latitude']]
    kmeans_neighborhood = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    neighborhood_clusters = kmeans_neighborhood.fit_predict(neighborhood_coordinates)
    neighborhood_data['Cluster'] = neighborhood_clusters

    # Create a new Folium map for the selected neighborhood
    m_neighborhood = folium.Map(location=[43.70, -79.42], zoom_start=10)  # Adjust zoom as needed

    # Loop through clusters and add a Circle and Marker for each cluster
    for cluster_id in range(optimal_num_clusters):
        cluster_data = neighborhood_data[neighborhood_data['Cluster'] == cluster_id]

        if cluster_data.empty:
            continue

        # Calculate the centroid of the cluster
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()

        # Calculate the radius of the circle based on the cluster size
        radius = np.sqrt(len(cluster_data)) * 25  # Adjust the scaling factor as needed

        # Draw a circle around the centroid to represent the size of the cluster
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m_neighborhood)

        # Add a Marker showing the number of points in the cluster
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m_neighborhood)

    # Convert the Folium map to HTML
    map_html_neighborhood = m_neighborhood._repr_html_()

    return html.Iframe(srcDoc=map_html_neighborhood, width='100%', height='300px')
# Run the web application
if __name__ == '__main__':
    app.run_server(debug=False)