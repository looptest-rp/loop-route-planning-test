#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import folium
from k_means_constrained import KMeansConstrained

def cluster_data(file, n_clusters, max_weight):
    # Read the CSV file into a Pandas dataframe
    data = pd.read_csv(file)
    
    # Apply the KMeansConstrained algorithm to the dataframe
    km_cons = KMeansConstrained(n_clusters=n_clusters,
                                init='k-means++',
                                size_max=max_weight,
                                random_state=42,
                                n_jobs=-1)
    y_predicted = km_cons.fit_predict(data[['longitude', 'latitude']])
    
    # Add the cluster labels to the dataframe
    data['cluster'] = y_predicted + 1
    
    # Create a list of colors for the clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    colors *= n_clusters // len(colors) + 1
    
    # Create a folium map object centered on the mean of the delivery locations
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)
    
    # Add cluster markers and lines to the map
    for i in range(1, n_clusters+1):
        cluster_data = data[data['cluster'] == i]
        color = colors[i-1]
        cluster_points = []
        for _, row in cluster_data.iterrows():
            # Add marker for each point
            folium.Marker(location=[row['latitude'], row['longitude']],
                          icon=folium.Icon(color=color),
                          popup=f"Cluster {i}").add_to(m)
            # Add point to list of cluster points
            cluster_points.append([row['latitude'], row['longitude']])
        # Add polyline connecting the cluster points
        folium.PolyLine(locations=cluster_points,
                        color=color).add_to(m)
    
    # Return the dataframe with the cluster labels and the folium map object
    return data, m

