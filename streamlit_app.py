import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

# Title of the Streamlit app
st.title('Keyword Clustering')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write(data)
    
    # Vectorize the keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['keywords'])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    # Add the cluster labels to the data
    data['cluster'] = kmeans.labels_
    
    # Display the dataframe with cluster labels
    st.write(data)
    
    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=data['cluster'])
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Keyword Clustering')
    
    # Show the plot in Streamlit
    st.pyplot(plt)

# Instruction for the user
st.text("Upload a CSV file containing keywords to see the clustering results.")
