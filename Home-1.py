from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('โปรเจคทำนายชนิดของดอกไอริสด้วย KNN')
st.subheader('เว็บแอปพลิเคชันสำหรับทำนายสายพันธุ์ดอกไอริส')

st.sidebar.image("./img/wit.jpg")

st.markdown("---")

st.write("## สายพันธุ์ดอกไอริส (Iris Species)")
col1, col2, col3 = st.columns(3)

with col1:
   st.subheader("Versicolor")
   st.image("./img/iris1.jpg", caption="Iris Versicolor")

with col2:
   st.subheader("Virginica")
   st.image("./img/iris2.jpg", caption="Iris Virginica")

with col3:
   st.subheader("Setosa")
   st.image("./img/iris3.jpg", caption="Iris Setosa")