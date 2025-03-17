import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

# Load Data (Membaca dataset dari file CSV)
day_df = pd.read_csv("dashboard/main_data_day.csv")
hour_df = pd.read_csv("dashboard/main_data_hour.csv")

# Data Preprocessing (membersihkan dan menyiapkan data)
weather_map = {1: 'Cerah', 2: 'Mendung', 3: 'Hujan Ringan', 4: 'Hujan Lebat'}
day_df['dteday'] = pd.to_datetime(day_df['dteday']) # Mengubah kolom tanggal menjadi tipe datetime
day_df['weathersit'] = day_df['weathersit'].map(weather_map) # Mengubah angka kondisi cuaca menjadi label
# Menentukan jenis hari (Weekend atau Weekday)
day_df['day_type'] = day_df['weekday'].apply(lambda x: 'Weekend' if x in [0, 6] else 'Weekday')

hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")
st.title('Bike Sharing Dashboard')

# Sidebar untuk filter data berdasarkan rentang tanggal dan kondisi cuaca
date_range = st.sidebar.date_input("Pilih Rentang Tanggal", [day_df["dteday"].min(), day_df["dteday"].max()],
                                   min_value=day_df["dteday"].min(), max_value=day_df["dteday"].max())
selected_weather = st.sidebar.multiselect("Pilih Kondisi Cuaca", options=day_df["weathersit"].unique(), 
                                          default=day_df["weathersit"].unique())
selected_day_type = st.sidebar.multiselect("Pilih Jenis Hari", options=day_df["day_type"].unique(), 
                                           default=day_df["day_type"].unique())

# Validasi input rentang tanggal
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Silakan pilih rentang tanggal yang valid.")
    st.stop()

# Filter data berdasarkan input pengguna
filtered_df = day_df[(day_df["dteday"] >= pd.to_datetime(start_date)) & 
                      (day_df["dteday"] <= pd.to_datetime(end_date)) & 
                      (day_df["weathersit"].isin(selected_weather)) & 
                      (day_df["day_type"].isin(selected_day_type))]

# Menampilkan metrik utama
total_rentals = filtered_df['cnt'].sum()
avg_rentals = filtered_df['cnt'].mean()
highest_day = filtered_df.loc[filtered_df['cnt'].idxmax(), 'dteday'].strftime('%Y-%m-%d') if not filtered_df.empty else "Data tidak tersedia"
col1, col2, col3 = st.columns(3)
col1.metric("Total Sewa", f"{total_rentals:,}")
col2.metric("Rata-rata Sewa Harian", f"{avg_rentals:,.2f}")
col3.metric("Hari Tertinggi Sewa", highest_day)

# Tabs untuk Visualisasi
tab1, tab2, tab3 = st.tabs(["Pengaruh Cuaca", "Hari Kerja vs Akhir Pekan", "Analisis Lanjutan"])

# Visualisasi pengaruh cuaca terhadap penyewaan sepeda
with tab1:
    st.subheader("Pengaruh Cuaca terhadap Jumlah Penyewaan Sepeda")
    weather_order = filtered_df.groupby('weathersit')['cnt'].sum().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8,5))
    color_palette = sns.color_palette("Blues", len(weather_order))
    sns.barplot(x='weathersit', y='cnt', data=filtered_df, estimator=sum, order=weather_order, ax=ax, palette=color_palette)
    ax.set_title("Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
    ax.set_xlabel("")  # Menghapus label sumbu X
    ax.set_ylabel("")  # Menghapus label sumbu Y
    st.pyplot(fig)

# Visualisasi perbandingan penyewaan pada Weekday dan Weekend
with tab2:
    st.subheader("Perbedaan Jumlah Penyewaan Sepeda antara Hari Kerja dan Akhir Pekan")
    fig, ax = plt.subplots(figsize=(8,5))
    custom_palette = {"Weekend": "#71b5a0", "Weekday": "#e89574"}
    sns.boxplot(x='day_type', y='cnt', data=filtered_df, ax=ax, palette=custom_palette)
    ax.set_title("Perbandingan Penyewaan Sepeda pada Hari Kerja dan Akhir Pekan")
    ax.set_xlabel("")  
    ax.set_ylabel("") 
    st.pyplot(fig)

# Analisis Lanjutan dengan RFM, Clustering, dan Geospatial Analysis (Heatmap)
with tab3:
    st.subheader("Analisis Lanjutan")
    
    # RFM Analysis
    st.markdown("### RFM Analysis")
    rfm_df = day_df.groupby('dteday').agg(
        Recency=('dteday', lambda x: (day_df['dteday'].max() - x.max()).days),
        Frequency=('cnt', 'count'),
        Monetary=('cnt', 'sum')
    ).reset_index()
    rfm_scaled = (rfm_df[['Recency', 'Frequency', 'Monetary']] - rfm_df[['Recency', 'Frequency', 'Monetary']].min()) / \
                 (rfm_df[['Recency', 'Frequency', 'Monetary']].max() - rfm_df[['Recency', 'Frequency', 'Monetary']].min())
    rfm_scaled = rfm_scaled.fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=rfm_df['Cluster'], y=rfm_df['Monetary'], palette='viridis', ax=ax)
    ax.set_title("RFM Clustering - Monetary per Cluster")
    st.pyplot(fig)
    
    # Clustering Penyewaan Berdasarkan Jam
    st.markdown("### Clustering Penyewaan Berdasarkan Jam")
    hour_df['hour'] = hour_df['hr']
    clustering_data = hour_df.groupby('hour').agg(total_rides=('cnt', 'sum')).reset_index()
    kmeans_hour = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustering_data['Cluster'] = kmeans_hour.fit_predict(clustering_data[['total_rides']])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=clustering_data['hour'], y=clustering_data['total_rides'], hue=clustering_data['Cluster'], palette='coolwarm', ax=ax)
    ax.set_title("Clustering Pola Penyewaan Sepeda per Jam")
    st.pyplot(fig)

    # Geospatial Analysis (Heatmap)
    st.markdown("### Geospatial Analysis (Heatmap)")
    if 'lat' in hour_df.columns and 'long' in hour_df.columns:
        map_bike = folium.Map(location=[hour_df['lat'].mean(), hour_df['long'].mean()], zoom_start=12)
        heat_data = list(zip(hour_df['lat'], hour_df['long'], hour_df['cnt']))
        HeatMap(heat_data).add_to(map_bike)
        st.components.v1.html(map_bike._repr_html_(), height=500)
    else:
        st.warning("Data geospasial tidak ditemukan dalam dataset.")