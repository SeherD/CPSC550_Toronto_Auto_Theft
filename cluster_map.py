import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
from folium import Circle, Marker
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Loading data from a CSV file into a pandas DataFrame
data = pd.read_csv('export.csv', header=None)
data.columns = ['Longitude', 'Latitude', 'Neighbourhood', 'Timestamp', 'Day', 'Hour']

# Generating a list of neighborhood options for the dropdown menu
neighborhood_options = sorted(
    [{'label': neighborhood, 'value': neighborhood} for neighborhood in data['Neighbourhood'].unique()],
    key=lambda x: x['label']
)

# Converting the 'Timestamp' column to datetime objects for easy handling
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extracting coordinates for clustering
coordinates = data[['Longitude', 'Latitude']]

# Function to determine the optimal number of clusters using the elbow method
def choose_optimal_num_clusters(coordinates):
    inertia_values = []
    cluster_range = range(2, 51)  # Range of potential cluster sizes to explore
    for num_clusters in cluster_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(coordinates)
        inertia_values.append(kmeans.inertia_)  # Inertia values are indicative of the fit quality

    # Plotting the elbow graph to visually find the optimal cluster number
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertia_values, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.grid(True)
    plt.show()
    optimal_num_clusters = int(input("Enter the optimal number of clusters based on the elbow method: "))
    return optimal_num_clusters

# This below line is commented out, but would be used to call the choose_optimal_num_clusters function
# so the user can choose the optimal number of k-mean clusters using the elbow method.
# however, since we have already performed the choosing of the number of clusters, it can
# be defined manually (see report for more details)
#optimal_num_clusters = choose_optimal_num_clusters(coordinates)

# Number of clusters chosen with the elbow method
optimal_num_clusters = 10

# Applying KMeans clustering to our data
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
clusters = kmeans.fit_predict(coordinates)

# Adding cluster information to the original DataFrame
data['Cluster'] = clusters

# Initializing the Dash application
app = dash.Dash(__name__)

# Defining the layout of the Dash application
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
            value=None,
            placeholder="Select a neighborhood",
            style={'margin': '10px', 'width': '48%'}
        ),
        dcc.Dropdown(
            id='neighborhood-dropdown-2',
            options=neighborhood_options,
            value=None,
            placeholder="Select another neighborhood",
            style={'margin': '10px', 'width': '48%'}
        ),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        html.Div(id='neighborhood-map-container', style={'display': 'inline-block', 'width': '50%'}),
        html.Div(id='neighborhood-map-container-2', style={'display': 'inline-block', 'width': '50%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
])

    # Callback function to update the map based on user-selected date range
@app.callback(
    Output('map-container', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_map(start_date, end_date):
    # Filter the data to include only events within the selected date range
    filtered_data = data[(data['Timestamp'] >= start_date) & (data['Timestamp'] <= end_date)]
    m = folium.Map(location=[43.70, -79.42], zoom_start=10)  # Initializing a Folium map centered around Toronto

    # Iterate through each cluster and add it to the map with Circle and Marker
    for cluster_id in filtered_data['Cluster'].unique():
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_id]
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()
        radius = np.sqrt(len(cluster_data)) * 35  # Dynamically adjust the radius based on cluster size

        # Visual representation of clusters using circles and markers
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m)

    # Embedding the Folium map into an HTML iframe for rendering in Dash
    map_html = m._repr_html_()
    return html.Iframe(srcDoc=map_html, width='100%', height='300px')

# Callback for updating the map of a specific neighborhood based on user selection
@app.callback(
    Output('neighborhood-map-container', 'children'),
    [Input('neighborhood-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_neighborhood_map(selected_neighborhood, start_date, end_date):
    # Handling the case where no neighborhood is selected
    if not selected_neighborhood:
        return "Please select a neighborhood."

    # Filter the data to include only events in the selected neighborhood and date range
    neighborhood_data = data[(data['Neighbourhood'] == selected_neighborhood) & 
                             (data['Timestamp'] >= start_date) & 
                             (data['Timestamp'] <= end_date)]

    # Applying KMeans clustering specific to the selected neighborhood
    neighborhood_coordinates = neighborhood_data[['Longitude', 'Latitude']]
    kmeans_neighborhood = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    neighborhood_clusters = kmeans_neighborhood.fit_predict(neighborhood_coordinates)
    neighborhood_data['Cluster'] = neighborhood_clusters

    # Generate and configure a Folium map for the selected neighborhood
    m_neighborhood = folium.Map(location=[43.70, -79.42], zoom_start=10)

    # Adding each cluster to the neighborhood map with visual markers
    for cluster_id in range(optimal_num_clusters):
        cluster_data = neighborhood_data[neighborhood_data['Cluster'] == cluster_id]
        if cluster_data.empty:
            continue  # Skip iteration if there's no data for a cluster
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()
        radius = np.sqrt(len(cluster_data)) * 25  # Set the circle radius based on the number of points in the cluster

        # Visualize each cluster using a Circle and Marker on the map
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m_neighborhood)
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m_neighborhood)

    # Convert the Folium map of the neighborhood to HTML for rendering
    map_html_neighborhood = m_neighborhood._repr_html_()
    return html.Iframe(srcDoc=map_html_neighborhood, width='100%', height='300px')

# Callback function for the second neighborhood dropdown selection
@app.callback(
    Output('neighborhood-map-container-2', 'children'),
    [Input('neighborhood-dropdown-2', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_neighborhood_map_2(selected_neighborhood, start_date, end_date):
    # Provide feedback if the second neighborhood is not selected
    if not selected_neighborhood:
        return "Please select a second neighborhood for comparison."

    # Filter data for the selected second neighborhood and date range
    neighborhood_data = data[(data['Neighbourhood'] == selected_neighborhood) & 
                             (data['Timestamp'] >= start_date) & 
                             (data['Timestamp'] <= end_date)]
    neighborhood_coordinates = neighborhood_data[['Longitude', 'Latitude']]

    # Applying KMeans clustering for the second neighborhood
    kmeans_neighborhood = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    neighborhood_clusters = kmeans_neighborhood.fit_predict(neighborhood_coordinates)
    neighborhood_data['Cluster'] = neighborhood_clusters

    # Creating a Folium map for the second neighborhood
    m_neighborhood = folium.Map(location=[43.70, -79.42], zoom_start=10)

    # Loop through the clusters in the second neighborhood and add them to the map
    for cluster_id in range(optimal_num_clusters):
        cluster_data = neighborhood_data[neighborhood_data['Cluster'] == cluster_id]
        if cluster_data.empty:
            continue  # Skip empty clusters
        centroid_longitude, centroid_latitude = cluster_data[['Longitude', 'Latitude']].mean()
        radius = np.sqrt(len(cluster_data)) * 25  # Adjust radius according to cluster size

        # Add visual markers for each cluster in the second neighborhood
        Circle(
            location=[centroid_latitude, centroid_longitude],
            radius=radius,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m_neighborhood)
        Marker(
            location=[centroid_latitude, centroid_longitude],
            popup=f"Cluster: {cluster_id}<br>Points: {len(cluster_data)}",
            icon=None
        ).add_to(m_neighborhood)

    # Embed the second neighborhood map into HTML for display
    map_html_neighborhood = m_neighborhood._repr_html_()
    return html.Iframe(srcDoc=map_html_neighborhood, width='100%', height='300px')

# Start the Dash application server
if __name__ == '__main__':
    app.run_server(debug=False)