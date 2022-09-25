import streamlit as st
import numpy as np
import seaborn as sns
import plotly.express as px



# Load iris
df = sns.load_dataset('iris')

# Make 3D Scatter plot
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')

# Updated text
st.write("""
# IRIS Dataset
One of Data Science's most fundamental datasets
""")

# Plot!
st.plotly_chart(fig, use_container_width=True)