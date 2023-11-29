import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('export.csv', header=None)  # Adjust the path to your data file
data.columns = ['Longitude', 'Latitude', 'Location', 'Date', 'Day', 'SomeValue']

# Extract the relevant columns for clustering (latitude and longitude)
coordinates = data[['Longitude', 'Latitude']]

# Perform k-means clustering
kmeans = KMeans(n_clusters=50)  # You can change the number of clusters
clusters = kmeans.fit_predict(coordinates)

# Add cluster labels to your dataframe
data['Cluster'] = clusters
# Assuming 'clusters' is the array of cluster labels you obtained from k-means

# Count the points in each cluster
cluster_counts = pd.Series(clusters).value_counts()

# Calculate total data points
total_points = len(data)

# Calculate relative probability for each cluster
cluster_probabilities = cluster_counts / total_points

# Normalize the probabilities to sum to 1 (optional)
cluster_probabilities /= cluster_probabilities.sum()


centroids = kmeans.cluster_centers_

# Scale the probabilities for visualization
scaling_factor = 100  # Example scaling factor, adjust based on your needs
scaled_probabilities = cluster_probabilities * scaling_factor

# Create a mapping from cluster label to scaled probability
cluster_to_probability = pd.Series(scaled_probabilities, index=cluster_counts.index)
# Extract the longitude and latitude of the centroids
centroid_longitudes = centroids[:, 0]
centroid_latitudes = centroids[:, 1]

# Extract the probabilities (scaled for plotting) for each centroid
centroid_sizes = [cluster_to_probability[i] * scaling_factor for i in range(len(centroids))]

# Plot the centroids with sizes proportional to the scaled probabilities
scatter = plt.scatter(centroid_longitudes, centroid_latitudes, s=centroid_sizes, c=range(len(centroids)), cmap='viridis')

# Now, cluster_probabilities will give you the relative chance of car theft for each cluster
# Basic plot (for testing)plt.figure(figsize=(10, 6))

# Adding a legend
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()

print(cluster_probabilities)
# Further steps would include exporting this clustered data for visualization on Google Maps