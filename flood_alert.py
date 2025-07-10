#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


# In[5]:


# API Integration
import requests
from datetime import datetime
import json

# Database
import mysql.connector
from mysql.connector import Error


# In[7]:


# Translation
from googletrans import Translator

# SMS (Twilio simulation)
import random

# Streamlit for UI
import streamlit as st
import folium
from streamlit_folium import folium_static

# Text-to-Speech
import pyttsx3


# In[9]:


# Set page config
st.set_page_config(
    page_title="🌊 Smart Flood Alert System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# In[11]:


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .city-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# In[13]:


class FloodAlertSystem:
    def __init__(self):
        self.api_key = "0afccaf637522ffe03857823c2e39611"
        self.cities = {
            'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'lang': 'Tamil'},
            'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'lang': 'Kannada'},
            'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'lang': 'Hindi'},
            'Delhi': {'lat': 28.7041, 'lon': 77.1025, 'lang': 'Hindi'},
            'Kochi': {'lat': 9.9312, 'lon': 76.2673, 'lang': 'Malayalam'},
            'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'lang': 'Hindi'}
        }
        self.translator = Translator()
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        
        # Mock phone numbers for testing
        self.phone_numbers = {
            'Chennai': ['+91' + str(random.randint(7000000000, 9999999999)) for _ in range(5)],
            'Bangalore': ['+91' + str(random.randint(7000000000, 9999999999)) for _ in range(5)],
            'Kolkata': ['+91' + str(random.randint(7000000000, 9999999999)) for _ in range(5)],
            'Delhi': ['+91' + str(random.randint(7000000000, 9999999999)) for _ in range(5)],
            'Kochi': ['+91' + str(random.randint(7000000000, 9999999999)) for _ in range(5)],
            'Hyderabad': ['+91' + str(random.randint(7000000000, 9999999999)) for _ in range(5)]
        }
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize MySQL database and create tables"""
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='0910'
            )
            cursor = connection.cursor()
            
            # Create database if not exists
            cursor.execute("CREATE DATABASE IF NOT EXISTS flood_alerts")
            cursor.execute("USE flood_alerts")
            
            # Create predictions_log table
            create_table_query = """
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                city_name VARCHAR(100),
                temperature FLOAT,
                humidity FLOAT,
                rainfall FLOAT,
                pressure FLOAT,
                wind_speed FLOAT,
                prediction VARCHAR(10),
                sms_sent VARCHAR(10),
                confidence FLOAT
            )
            """
            cursor.execute(create_table_query)
            
            connection.commit()
            cursor.close()
            connection.close()
            
            st.success("✅ Database initialized successfully!")
            
        except Error as e:
            st.error(f"❌ Database error: {e}")
    
    def generate_mock_dataset(self):
        """Generate mock flood dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Latitude': np.random.uniform(8, 35, n_samples),
            'Longitude': np.random.uniform(68, 97, n_samples),
            'Rainfall': np.random.exponential(50, n_samples),
            'Temperature': np.random.normal(28, 8, n_samples),
            'Humidity': np.random.uniform(40, 95, n_samples),
            'River_Discharge': np.random.exponential(1000, n_samples),
            'Water_Level': np.random.uniform(0, 15, n_samples),
            'Elevation': np.random.uniform(0, 500, n_samples),
            'Land_Cover': np.random.choice(['Urban', 'Rural', 'Forest', 'Water'], n_samples),
            'Soil_Type': np.random.choice(['Clay', 'Sandy', 'Loamy', 'Rocky'], n_samples),
            'Population_Density': np.random.uniform(100, 10000, n_samples),
            'Infrastructure': np.random.choice(['Good', 'Average', 'Poor'], n_samples),
            'Historical_Floods': np.random.randint(0, 5, n_samples)
        }
        
        # Create flood occurrence based on logical conditions
        flood_conditions = (
            (data['Rainfall'] > 100) & 
            (data['Humidity'] > 80) & 
            (data['Water_Level'] > 10) |
            (data['Rainfall'] > 200) |
            (data['Historical_Floods'] > 2)
        )
        
        data['Flood_Occurred'] = flood_conditions.astype(int)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Handle categorical variables
        categorical_cols = ['Land_Cover', 'Soil_Type', 'Infrastructure']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col != 'Flood_Occurred']
        
        return df
    
    def train_model(self):
        """Train Random Forest model"""
        # Generate mock dataset
        df = self.generate_mock_dataset()
        df = self.preprocess_data(df)
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['Flood_Occurred']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        joblib.dump(self.model, 'flood_model.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
        
        return df, accuracy
        # Add classification report and confusion matrix
        report = classification_report(y_test, y_pred, output_dict=False)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Display in Streamlit
        st.text("📊 Classification Report:")
        st.text(report)
        
        st.subheader("🔄 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    
    def get_weather_data(self, city):
        """Fetch real-time weather data from OpenWeatherMap"""
        try:
            city_info = self.cities[city]
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={city_info['lat']}&lon={city_info['lon']}&appid={self.api_key}&units=metric"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'rainfall': data.get('rain', {}).get('1h', 0),
                    'description': data['weather'][0]['description']
                }
            else:
                # Return mock data if API fails
                return {
                    'temperature': random.uniform(20, 40),
                    'humidity': random.uniform(60, 90),
                    'pressure': random.uniform(1000, 1020),
                    'wind_speed': random.uniform(5, 15),
                    'rainfall': random.uniform(0, 50),
                    'description': 'Partly cloudy'
                }
        except Exception as e:
            st.error(f"Weather API error: {e}")
            return None
    
    def predict_flood(self, weather_data, city):
        """Predict flood probability"""
        if self.model is None:
            st.error("Model not trained yet!")
            return None
        
        try:
            # Create feature vector
            city_info = self.cities[city]
            features = [
                city_info['lat'],  # Latitude
                city_info['lon'],  # Longitude
                weather_data['rainfall'],  # Rainfall
                weather_data['temperature'],  # Temperature
                weather_data['humidity'],  # Humidity
                random.uniform(500, 2000),  # River_Discharge (mock)
                random.uniform(2, 8),  # Water_Level (mock)
                random.uniform(10, 100),  # Elevation (mock)
                random.randint(0, 3),  # Land_Cover (encoded)
                random.randint(0, 3),  # Soil_Type (encoded)
                random.uniform(1000, 8000),  # Population_Density (mock)
                random.randint(0, 2),  # Infrastructure (encoded)
                random.randint(0, 3)   # Historical_Floods (mock)
            ]
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0][1]
            
            return {
                'prediction': 'Yes' if prediction == 1 else 'No',
                'probability': probability,
                'confidence': max(probability, 1-probability)
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def translate_message(self, message, target_lang):
        """Translate alert message to local language"""
        try:
            lang_codes = {
                'Tamil': 'ta',
                'Hindi': 'hi',
                'Malayalam': 'ml',
                'Kannada': 'kn',
                'Bengali': 'bn'
            }
            
            if target_lang in lang_codes:
                translated = self.translator.translate(
                    message, dest=lang_codes[target_lang]
                )
                return translated.text
            else:
                return message
        except Exception as e:
            st.error(f"Translation error: {e}")
            return message
    
    def send_sms_alert(self, city, prediction_result):
        """Simulate SMS alert sending"""
        if prediction_result['prediction'] == 'Yes':
            message = f"⚠️ Flood Alert: Heavy risk of flooding in {city}. Please take safety precautions."
            
            # Translate message
            local_lang = self.cities[city]['lang']
            translated_message = self.translate_message(message, local_lang)
            
            # Simulate SMS sending
            phone_numbers = self.phone_numbers[city]
            
            return {
                'sent': True,
                'message': message,
                'translated_message': translated_message,
                'recipients': len(phone_numbers),
                'phone_numbers': phone_numbers
            }
        else:
            return {
                'sent': False,
                'message': "No flood risk detected. No SMS sent.",
                'translated_message': "",
                'recipients': 0,
                'phone_numbers': []
            }
    
    def log_prediction(self, city, weather_data, prediction_result, sms_result):
        """Log prediction to database"""
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='0910',
                database='flood_alerts'
            )
            cursor = connection.cursor()
            
            insert_query = """
            INSERT INTO predictions_log 
            (timestamp, city_name, temperature, humidity, rainfall, pressure, wind_speed, prediction, sms_sent, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                datetime.now(),
                city,
                weather_data['temperature'],
                weather_data['humidity'],
                weather_data['rainfall'],
                weather_data['pressure'],
                weather_data['wind_speed'],
                prediction_result['prediction'],
                'Yes' if sms_result['sent'] else 'No',
                prediction_result['confidence']
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            cursor.close()
            connection.close()
            
            st.success("✅ Prediction logged to database!")
            
        except Error as e:
            st.error(f"Database logging error: {e}")
    
    def create_visualizations(self, weather_data, city, prediction_result):
        """Create weather and flood risk visualizations"""
        
        # Weather Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature', 'Humidity', 'Rainfall', 'Wind Speed'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Temperature gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=weather_data['temperature'],
            title={'text': "Temperature (°C)"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 35], 'color': "yellow"},
                    {'range': [35, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ), row=1, col=1)
        
        # Humidity gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=weather_data['humidity'],
            title={'text': "Humidity (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ), row=1, col=2)
        
        # Rainfall gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=weather_data['rainfall'],
            title={'text': "Rainfall (mm)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ), row=2, col=1)
        
        # Wind Speed gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=weather_data['wind_speed'],
            title={'text': "Wind Speed (m/s)"},
            gauge={
                'axis': {'range': [None, 30]},
                'bar': {'color': "darkorange"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 30], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 25
                }
            }
        ), row=2, col=2)
        
        fig.update_layout(
            title=f"🌤️ Weather Dashboard - {city}",
            font=dict(size=12),
            height=600
        )
        
        return fig
    
    def create_flood_risk_map(self, city):
        """Create flood risk map using Folium"""
        city_info = self.cities[city]
        
        # Create map centered on city
        m = folium.Map(
            location=[city_info['lat'], city_info['lon']],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add city marker
        folium.Marker(
            [city_info['lat'], city_info['lon']],
            popup=f"{city} - Monitoring Station",
            tooltip=f"Click for {city} details",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add flood risk zones (mock data)
        risk_zones = [
            {'lat': city_info['lat'] + 0.01, 'lon': city_info['lon'] + 0.01, 'risk': 'High'},
            {'lat': city_info['lat'] - 0.01, 'lon': city_info['lon'] - 0.01, 'risk': 'Medium'},
            {'lat': city_info['lat'] + 0.005, 'lon': city_info['lon'] - 0.005, 'risk': 'Low'},
        ]
        
        colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        
        for zone in risk_zones:
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=20,
                popup=f"Risk Level: {zone['risk']}",
                color=colors[zone['risk']],
                fill=True,
                fillColor=colors[zone['risk']],
                fillOpacity=0.6
            ).add_to(m)
        
        return m


# In[15]:


# Initialize the system
@st.cache_resource
def get_flood_system():
    return FloodAlertSystem()

def main():
    st.markdown('<h1 class="main-header">🌊 Smart Flood Alert System</h1>', unsafe_allow_html=True)
    
    # Initialize system
    flood_system = get_flood_system()
    
    # Sidebar
    st.sidebar.header("🏙️ City Selection")
    selected_city = st.sidebar.selectbox(
        "Choose a city:",
        list(flood_system.cities.keys())
    )
    
    st.sidebar.markdown("---")
    
    # Train model button
    if st.sidebar.button("🤖 Train ML Model"):
        with st.spinner("Training model..."):
            df, accuracy = flood_system.train_model()
            st.sidebar.success(f"Model trained! Accuracy: {accuracy:.2f}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📍 {selected_city} Flood Alert Dashboard")
        
        # Get weather data button
        if st.button("🌤️ Get Current Weather & Flood Prediction"):
            with st.spinner("Fetching weather data..."):
                weather_data = flood_system.get_weather_data(selected_city)
                
                if weather_data:
                    # Display weather info
                    st.subheader("Current Weather Conditions")
                    
                    weather_col1, weather_col2, weather_col3 = st.columns(3)
                    
                    with weather_col1:
                        st.metric("🌡️ Temperature", f"{weather_data['temperature']:.1f}°C")
                        st.metric("💧 Humidity", f"{weather_data['humidity']:.1f}%")
                    
                    with weather_col2:
                        st.metric("🌧️ Rainfall", f"{weather_data['rainfall']:.1f}mm")
                        st.metric("💨 Wind Speed", f"{weather_data['wind_speed']:.1f}m/s")
                    
                    with weather_col3:
                        st.metric("🔽 Pressure", f"{weather_data['pressure']:.1f}hPa")
                        st.write(f"**Conditions:** {weather_data['description'].title()}")
                    
                    # Make flood prediction
                    if flood_system.model is not None:
                        prediction_result = flood_system.predict_flood(weather_data, selected_city)
                        
                        if prediction_result:
                            # Display prediction
                            st.subheader("🔮 Flood Prediction Results")
                            
                            if prediction_result['prediction'] == 'Yes':
                                st.markdown(f'''
                                <div class="alert-box alert-danger">
                                    <h3>⚠️ FLOOD ALERT ISSUED ⚠️</h3>
                                    <p><strong>Prediction:</strong> {prediction_result['prediction']}</p>
                                    <p><strong>Probability:</strong> {prediction_result['probability']:.2f}</p>
                                    <p><strong>Confidence:</strong> {prediction_result['confidence']:.2f}</p>
                                </div>
                                ''', unsafe_allow_html=True)
                            else:
                                st.markdown(f'''
                                <div class="alert-box alert-success">
                                    <h3>✅ NO FLOOD RISK DETECTED</h3>
                                    <p><strong>Prediction:</strong> {prediction_result['prediction']}</p>
                                    <p><strong>Probability:</strong> {prediction_result['probability']:.2f}</p>
                                    <p><strong>Confidence:</strong> {prediction_result['confidence']:.2f}</p>
                                </div>
                                ''', unsafe_allow_html=True)
                            
                            # SMS Alert
                            sms_result = flood_system.send_sms_alert(selected_city, prediction_result)
                            
                            st.subheader("📱 SMS Alert System")
                            if sms_result['sent']:
                                st.success(f"✅ SMS Alert sent to {sms_result['recipients']} recipients!")
                                
                                # Show translated message
                                local_lang = flood_system.cities[selected_city]['lang']
                                st.write(f"**English:** {sms_result['message']}")
                                st.write(f"**{local_lang}:** {sms_result['translated_message']}")
                                
                                # Show phone numbers
                                with st.expander("📞 View recipient phone numbers"):
                                    for i, phone in enumerate(sms_result['phone_numbers'], 1):
                                        st.write(f"{i}. {phone}")
                            else:
                                st.info("ℹ️ No SMS alert sent - No flood risk detected")
                            
                            # Log to database
                            flood_system.log_prediction(selected_city, weather_data, prediction_result, sms_result)
                            
                            # Create visualizations
                            st.subheader("📊 Weather Visualization")
                            fig = flood_system.create_visualizations(weather_data, selected_city, prediction_result)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Please train the ML model first using the sidebar button!")
    
    with col2:
        st.header("🗺️ Flood Risk Map")
        
        if st.button("🌍 Generate Map"):
            with st.spinner("Creating flood risk map..."):
                flood_map = flood_system.create_flood_risk_map(selected_city)
                folium_static(flood_map, width=400, height=500)
        
        st.markdown("---")
        
        # Additional features
        st.header("🔧 Additional Features")
        
        if st.button("📊 View Historical Data"):
            st.info("Historical flood data analysis would be displayed here")
        
        if st.button("🎯 Model Performance"):
            st.info("Model accuracy metrics and performance charts would be shown here")
        
        if st.button("📈 Trend Analysis"):
            st.info("Long-term flood trends and patterns would be analyzed here")


# In[17]:


if __name__ == "__main__":
    main()


# In[ ]:




