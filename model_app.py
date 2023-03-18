#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from model_with_map import cluster_data

# Add a title to the app
st.title("Route Planning App")

# Create a file uploader component in Streamlit
file = st.file_uploader("Upload a CSV file with orders latitude and longitude fields")

# Add a sidebar to the app
st.sidebar.title("Input Parameters")

# Accept user input for n_clusters and max_weight
n_clusters = st.sidebar.text_input("Number of clusters", value="50")
max_weight = st.sidebar.text_input("Maximum weight/orders per cluster", value="28")

# If a file is uploaded, cluster the data and display the results
if file is not None:
    # Convert the input values to integers
    n_clusters = int(n_clusters)
    max_weight = int(max_weight)

    # Call the cluster_data function with the user input arguments
    data, m = cluster_data(file, n_clusters, max_weight)
    
    # Display the data with the cluster labels
    st.write("Clustered Data:")
    st.write(data)
    
    # Display the folium map
    st.write("Cluster Map:")
    st_folium(m, width=1000)
else:
    st.write("Please upload a file to begin route planning.")

