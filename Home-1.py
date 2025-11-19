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



# Use caching to load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Expander for data statistics
with st.expander("ดูสถิติและกราฟข้อมูลดอกไม้ (View Data Statistics and Chart)"):
    dt = load_data("./data/iris.csv")
    
    st.write("#### ข้อมูล 10 แถวแรก (First 10 rows)")
    st.write(dt.head(10))

    # Prepare data for the chart with meaningful labels
    feature_sums = dt[['petal.length', 'petal.width', 'sepal.length', 'sepal.width']].sum()
    sum_df = pd.DataFrame(feature_sums, columns=["ผลรวม (Total Sum)"])
    
    st.write("#### กราฟแท่งแสดงผลรวมของแต่ละคุณลักษณะ (Bar Chart of Feature Sums)")
    st.bar_chart(sum_df)