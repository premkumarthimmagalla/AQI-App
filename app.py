import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Beijing Air Quality Analysis",
    page_icon="🌍",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Model"])

# EDA Page
if page == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Add descriptions before each visualization
    st.markdown("### 1. Median Pollutant Levels per Month (2013-2017)")
    st.write("This visualization shows how pollutant levels vary across months for each year.")
    st.image('EDA/median pollutant levels per month for each year.png')
    
    st.markdown("### 2. Yearly Trend of Pollutants")
    st.write("This plot demonstrates the overall trend of pollutants across years.")
    st.image('EDA/median pollutant levels per year.png')

    st.markdown("### 3. Correlation Analysis")
    st.write("Heat map showing correlations between different pollutants and weather parameters.")
    st.image('EDA/Pearson correlation matrix.png')

    st.markdown("### 4. PM2.5 Analysis")
    st.write("Detailed analysis of PM2.5 concentrations and their patterns.")
    st.image('EDA/PM 2.5.png')

    st.markdown("### 5. Weekly Pollutant Patterns")
    st.write("How pollutant levels vary across different days of the week.")
    st.image('EDA/Pollutant Levels by Day of the Week.png')

    st.markdown("### 6. Daily Pollutant Patterns")
    st.write("Hour-by-hour analysis of pollutant levels throughout the day.")
    st.image('EDA/pollutant levels by hour of day.png')

# Model Page
elif page == "Model":
    st.title("Air Quality Prediction Models")
    
    try:
        # Load models
        decision_tree_model = joblib.load("models/DecisionTreeClassifier.joblib")
        logistic_regression_model = joblib.load("models/LogisticRegression.joblib")
        random_forest_model = joblib.load("models/RandomForestClassifier.joblib")

        # Function to predict AQI
        def predict_aqi(model, input_data):
            try:
                prediction = model.predict(input_data)
                return prediction[0]
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return None

        # User input section
        st.subheader("Enter Parameters for Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            No = st.selectbox("NO", [0, 1])
            year = st.selectbox('Year', [2013, 2014, 2015, 2016, 2017])
            month = st.selectbox("Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            day = st.slider('Day', 1, 31)
            hour = st.slider('Hour', 0, 23)
            PM2_5 = st.number_input("PM2.5", min_value=0.0, max_value=999.0, step=0.1)
            TEMP = st.number_input("Temperature (°C)", min_value=-18.0, max_value=42.0, step=0.1)

        with col2:
            PM10 = st.number_input("PM10", min_value=2.0, max_value=999.0, step=0.1)
            SO2 = st.number_input("SO2", min_value=2.8, max_value=999.0, step=0.1)
            NO2 = st.number_input("NO2", min_value=1.0, max_value=280.0)
            CO = st.number_input("CO", min_value=100.0, max_value=1000.0, step=0.5)
            PRES = st.number_input("Pressure (hPa)", min_value=982.4, max_value=1042.8, step=0.1)

        with col3:
            O3 = st.number_input("O3", min_value=0.2, max_value=1071.0, step=0.1)
            DEWP = st.number_input("Dew Point (°C)", min_value=-43.4, max_value=29.1)
            RAIN = st.number_input("Rain (mm)", min_value=0.0, max_value=72.5, step=0.1)
            wd = st.number_input("Wind Direction (°)", min_value=0, max_value=360, step=1)
            WSPM = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=13.2, step=0.1)
            station = st.number_input("Station ID", min_value=0, max_value=100, step=1)
            day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, step=1)

        # Create input DataFrame
        input_data = pd.DataFrame({
            'No': [No], 'year': [year], 'month': [month], 'day': [day],
            'hour': [hour], 'PM2.5': [PM2_5], 'PM10': [PM10], 'SO2': [SO2],
            'NO2': [NO2], 'CO': [CO], 'O3': [O3], 'TEMP': [TEMP],
            'PRES': [PRES], 'DEWP': [DEWP], 'RAIN': [RAIN], 'wd': [wd],
            'WSPM': [WSPM], 'station': [station], 'day_of_week': [day_of_week]
        })

        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ("Decision Tree", "Logistic Regression", "Random Forest")
        )

        if st.button("Predict Air Quality"):
            # Make prediction based on selected model
            if model_choice == "Decision Tree":
                prediction = predict_aqi(decision_tree_model, input_data)
            elif model_choice == "Logistic Regression":
                prediction = predict_aqi(logistic_regression_model, input_data)
            else:
                prediction = predict_aqi(random_forest_model, input_data)

            if prediction is not None:
                st.success(f"Predicted Air Quality Category: {prediction}")
                
                # Add interpretation
                st.info("""
                Air Quality Categories:
                - Excellent: Very clean air
                - Good: Satisfactory air quality
                - Slightly Polluted: Acceptable but may affect sensitive groups
                - Lightly Polluted: May cause health concerns
                - Moderately Polluted: Health warnings for sensitive groups
                - Heavily Polluted: Health alerts for everyone
                - Severely Polluted: Emergency conditions
                """)

    except Exception as e:
        st.error(f"Error in model operations: {str(e)}")

# Data Overview Page
elif page == "Data Overview":
    st.title("Beijing Air Quality Analysis")
    st.write("Dataset Overview (2013-2017)")
    
    st.markdown("""
    ### About the Dataset
    This comprehensive dataset includes hourly air pollutants data from 12 nationally-controlled 
    air-quality monitoring sites in Beijing. The data spans from March 1st, 2013 to February 28th, 2017.
    
    ### Data Sources
    - Air quality data: Beijing Municipal Environmental Monitoring Center
    - Meteorological data: China Meteorological Administration
    """)
    
    st.subheader("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Air Quality Parameters:**
        - PM2.5: Fine particulate matter
        - PM10: Inhalable particulate matter
        - SO2: Sulfur dioxide
        - NO2: Nitrogen dioxide
        - CO: Carbon monoxide
        - O3: Ozone
        """)
        
    with col2:
        st.markdown("""
        **Meteorological Parameters:**
        - TEMP: Temperature
        - PRES: Pressure
        - DEWP: Dew point temperature
        - RAIN: Precipitation
        - wd: Wind direction
        - WSPM: Wind speed
        """)
    
    st.subheader("Dataset Statistics")
    st.markdown("""
    - Time period: 2013-2017
    - Monitoring stations: 12
    - Measurement frequency: Hourly
    - Parameters measured: 18
    """)

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Created by: [Your Name]")
    st.sidebar.write("Module: CMP7005")
