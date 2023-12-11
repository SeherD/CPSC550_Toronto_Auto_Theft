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
    # Elbow method to find optimal number of clusters
    inertia_values = []
    cluster_range = range(2, 51)  # You can adjust the range of cluster numbers
    for num_clusters in cluster_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # Set a fixed random seed (e.g., 42)
        kmeans.fit(coordinates)
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertia_values, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.grid(True)
    plt.show()

    # Choose the number of clusters with the "elbow" point
    optimal_num_clusters = int(input("Enter the optimal number of clusters based on the elbow method: "))
    return optimal_num_clusters

# Original code for loading and creating DataFrame
data = pd.read_csv('export.csv', header=None)
data.columns = ['Longitude', 'Latitude', 'Neighbourhood', 'Timestamp', 'Day', 'Hour']

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

    dcc.Dropdown(
        id='cluster-dropdown',
        options=[{'label': f'Cluster {i}', 'value': i} for i in range(optimal_num_clusters)],
        multi=True,
        value=list(range(5)),  # Show top 5 clusters by default
        style={'margin': '10px'}
    ),

    html.Div(id='map-container')
])

# Callback to update the map based on the selected date range and top N clusters
@app.callback(
    Output('map-container', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('cluster-dropdown', 'value')]
)
def update_map(start_date, end_date, selected_clusters):
    filtered_data = data[(data['Timestamp'] >= start_date) & (data['Timestamp'] <= end_date)]

    # Create a Folium map centered at Toronto
    m = folium.Map(location=[43.70, -79.42], zoom_start=11)

    # Loop through selected clusters and add a Circle and Marker for each cluster
    for cluster_id in selected_clusters:
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

    return html.Iframe(srcDoc=map_html, width='100%', height='600px')

# Run the web application
if __name__ == '__main__':
    app.run_server(debug=False)
