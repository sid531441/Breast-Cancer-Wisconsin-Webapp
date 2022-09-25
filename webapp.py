import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
# Wisconsin Breast Cancer Dataset
- Siddharth Ashok Unnithan
- Sangeetha Suresh
- Yunus Shareef

---
#### **From the IDA and EDA processes, we have filtered out which columns/ \
features are useful for detecting cancer.**
---

##### Vary the x-axis with the given features to see how they \
are related to diagnosis.
"""

df = pd.read_csv('data.csv')

use = ['id', 
       'diagnosis', 
       'radius_mean', 
       'compactness_mean', 
       'concavity_mean', 
       'concave points_mean', 
       'radius_worst', 
       'texture_worst', 
       'compactness_worst', 
       'concavity_worst', 
       'concave points_worst', 
       'symmetry_worst']

df = df[use]

x = st.selectbox(
    'X axis = ',
     df.drop(columns=["id", "diagnosis"]).columns,
     key = 'x')

fig1 = sns.catplot(data=df, x=x, y='diagnosis', kind='box')
fig1.fig.set_size_inches(15,5)
st.pyplot(fig1)

"""
---
#### Select the y-axis to form a scatter plot with the chosen feature \
and tick the checkboxes for the required distributions.
"""

y = st.selectbox(
    'Y axis = ',
     df.drop(columns=["id", "diagnosis"]).columns,
     key = 'y')

right_plot_true = st.checkbox('Right plot', key='right_plot')
up_plot_true = st.checkbox('Top plot', key='top_plot')

fig2 = sns.jointplot(data=df, x=x, y=y, kind='reg', lowess=True)
fig2.fig.set_size_inches(15,7)

if not right_plot_true:
    fig2.ax_marg_y.remove()
if not up_plot_true:
    fig2.ax_marg_x.remove()

st.pyplot(fig2)