import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load your dataset
data = pd.read_csv('E-commerce Customer Behavior - Sheet1.csv')

# Data Preprocessing
# Convert categorical variables to numerical values
data = pd.get_dummies(data, columns=['Gender', 'City', 'Membership Type', 'Satisfaction Level'], drop_first=True)

# Check for missing values and handle them if necessary
if data.isnull().sum().any():
    st.warning("There are missing values in the dataset. Please handle them before proceeding.")
    data.dropna(inplace=True)  # Simple approach to drop rows with missing values

# Streamlit App
st.title("Customer Segmentation Dashboard")

# Sidebar for user input
st.sidebar.header("User Input")

# Display the raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)

# Display the number of clusters
clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)

# KMeans Clustering
features = ['Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
kmeans = KMeans(n_clusters=clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Display Cluster Analysis
st.subheader("Cluster Analysis")
cluster_analysis = data.groupby('Cluster').mean().reset_index()
st.write(cluster_analysis)

# Visualization
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(x=data['Total Spend'], y=data['Items Purchased'], hue=data['Cluster'], palette='viridis', ax=ax)
plt.title("Customer Segmentation")
plt.xlabel("Total Spend")
plt.ylabel("Items Purchased")
st.pyplot(fig)

# Download button for segmented data
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(data)

st.download_button(
    label="Download Segmented Data",
    data=csv,
    file_name='segmented_customers.csv',
    mime='text/csv',
)
