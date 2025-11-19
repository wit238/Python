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


html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")