import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import csv
from sklearn.cluster import KMeans

data = pd.read_csv('export.csv', header=None)
data.columns = ['Longitude', 'Latitude', 'Neighbourhood', 'Timestamp', 'Day', 'Hour']
df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

coordinates = data[['Longitude', 'Latitude']]

# Perform k-means clustering
kmeans = KMeans(n_clusters=50)  # You can change the number of clusters
clusters = kmeans.fit_predict(coordinates)

# Add cluster labels to your dataframe
data['Cluster'] = clusters
# Assuming 'clusters' is the array of cluster labels you obtained from k-means

# Count the points in each cluster
cluster_counts = pd.Series(clusters).value_counts()

app = dash.Dash(__name__)
# Create a Dash web application
app.layout = html.Div([
    html.H1("Toronto Cluster Map Dashboard"),

    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df['Timestamp'].min(),
        end_date=df['Timestamp'].max(),
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
    filtered_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

    # Create a Folium map centered at Toronto
    m = folium.Map(location=[43.70, -79.42], zoom_start=11)

    # Use MarkerCluster for better performance with a large number of markers
    marker_cluster = MarkerCluster().add_to(m)

    # Plot clusters on the map
    for index, row in filtered_df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Cluster: {row['Cluster']}<br>Timestamp: {row['Timestamp']}",
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    # Convert the Folium map to HTML
    map_html = m._repr_html_()

    return html.Iframe(srcDoc=map_html, width='100%', height='600px')

# Run the web application
if __name__ == '__main__':
    app.run_server(debug=True)
