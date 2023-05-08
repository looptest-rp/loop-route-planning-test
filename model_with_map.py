#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import folium
from k_means_constrained import KMeansConstrained

def cluster_data(file, n_clusters, max_weight):
    # Read the CSV file into a Pandas dataframe
    data = pd.read_csv(file)
    
    # check if weight constraint column exists in the csv file 
    # for clients using weight as a constraint then this field needs to be present in the csv file
    if 'Weight' in data.columns:
        data = data.loc[data.index.repeat(data['Weight'])].reset_index(drop=True)
        data['Weight'] = 1
    else:
        data['Weight'] = 1
    
    # Error handling message if data cannot be clustered based on provided clusters and constraints
    if len(data) > n_clusters * max_weight:
            raise ValueError(f"""
                Clustering of the data is impossible with the defined combination of number of clusters ({n_clusters}) 
                and max stops/weight constraints ({max_weight}).
                Either increase the number of clusters or the max stops/weight constraints.
                The total weight or number of orders in your data ({len(data)}) should be less than or equal to the 
                multiplication of number of clusters and max stops/weight ({n_clusters*max_weight}).
                """)


    # Apply the KMeansConstrained algorithm to the dataframe
    km_cons = KMeansConstrained(n_clusters=n_clusters,
                                init='k-means++',
                                size_max=max_weight,
                                random_state=42,
                                max_iter = 300,
                                n_jobs=-1)
    y_predicted = km_cons.fit_predict(data[['longitude', 'latitude']])
    
    # Add the cluster labels to the dataframe
    data['cluster'] = y_predicted
    
    # Group data back to its original state
    data = data.groupby([col for col in data.columns if col != 'Weight']).agg({'Weight': 'sum'}).reset_index()
    
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

