import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Air Quality Index Prediction", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model", "Data Overview"])

# EDA Page
if page == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Add tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Monthly Pollutants", 
        "Yearly Trends", 
        "Correlation Matrix",
        "PM2.5 Analysis",
        "Weekly Patterns",
        "Daily Patterns"
    ])
    
    with tab1:
        st.header("Median Pollutant Levels per Month")
        st.image('median pollutant levels per month for each year.png')
        st.markdown("""
        This visualization shows how different pollutant levels vary across months for each year.
        Key observations:
        - Higher pollution levels typically occur in winter months
        - Summer months generally show lower pollutant concentrations
        """)

    with tab2:
        st.header("Yearly Pollution Trends")
        st.image('median pollutant levels per year.png')
        st.markdown("""
        Analysis of pollution trends across years shows:
        - Overall declining trend in most pollutants
        - Seasonal patterns remain consistent year over year
        """)

    with tab3:
        st.header("Correlation Between Pollutants")
        st.image('Pearson correlation matrix.png')
        st.markdown("""
        The correlation matrix reveals:
        - Strong positive correlations between PM2.5 and PM10
        - Temperature shows negative correlation with most pollutants
        - Pressure and wind speed show interesting relationships with pollutant levels
        """)

    with tab4:
        st.header("PM2.5 Detailed Analysis")
        st.image('PM 2.5.png')
        st.markdown("""
        PM2.5 is a critical pollutant:
        - Shows strong seasonal variations
        - Higher concentrations during winter months
        - Clear daily patterns visible
        """)

    with tab5:
        st.header("Weekly Pollution Patterns")
        st.image('Pollutant Levels by Day of the Week.png')
        st.markdown("""
        Analysis of weekly patterns shows:
        - Weekday vs weekend variations
        - Traffic-related pollution patterns
        - Industrial activity influence
        """)

    with tab6:
        st.header("Daily Pollution Cycles")
        st.image('pollutant levels by hour of day.png')
        st.markdown("""
        24-hour pollution cycles reveal:
        - Peak pollution hours
        - Impact of rush hour traffic
        - Nighttime vs daytime patterns
        """)

# Model Page
elif page == "Model":
    st.title("Model")
    # Load models
    decision_tree_model = joblib.load("DecisionTreeClassifier.joblib")
    logistic_regression_model = joblib.load("LogisticRegression.joblib")
    

    def normalize_input(data):
        ranges = {
            'CO': (100.0, 1000.0),
            'PM10': (2.0, 999.0),
            'O3': (0.2, 1071.0),
            'PM2.5': (0.0, 999.0),
            'NO2': (1.0, 280.0),
            'SO2': (2.8, 999.0),
            'TEMP': (-18.0, 42.0),
            'PRES': (982.4, 1042.8),
            'DEWP': (-43.4, 29.1),
            'RAIN': (0.0, 72.5),
            'wd': (0, 360),
            'WSPM': (0.0, 13.2),
            'station': (0, 100),
            'day_of_week': (0, 6),
            'No': (0, 1),
            'year': (2013, 2017),
            'month': (1, 12),
            'day': (1, 31),
            'hour': (0, 23)
        }
        
        normalized = data.copy()
        for column in data.columns:
            min_val, max_val = ranges[column]
            normalized[column] = (data[column] - min_val) / (max_val - min_val)
        return normalized


    def predict_aqi(model, input_data):
        normalized_data = normalize_input(input_data)
        prediction = model.predict(normalized_data)
        return prediction[0]


    st.title("AQI Prediction")
    
    try:
        # Input fields for the features
        No = st.selectbox("NO",[0,1])
        year = st.selectbox('Pick year', [2013,2014,2015,2016,2017])
        month = st.selectbox("Month",[1,2,3,4,5,6,7,8,9,10,11,12])
        day = st.slider('Day', 0, 31)
        hour = st.slider('Hour', 0, 24)
        PM2_5 = st.number_input("PM2.5", min_value=0.0, max_value=999.0, step=0.1)
        PM10 = st.number_input("PM10", min_value=2.0, max_value=999.0, step=0.1)
        SO2 = st.number_input("SO2", min_value=2.8, max_value=999.0, step=0.1)
        NO2 = st.number_input("NO2", min_value=1.0, max_value=280.0)
        CO = st.number_input("CO", min_value=100.0, max_value=1000.0, step=0.5)
        O3 = st.number_input("O3", min_value=0.2, max_value=1071.0, step=0.1)
        TEMP = st.number_input("Temperature (TEMP)", min_value=-18.0, max_value=42.0, step=0.1)
        PRES = st.number_input("Pressure (PRES)", min_value=982.4, max_value=1042.8, step=0.1)
        DEWP = st.number_input("Dew Point (DEWP)", min_value=-43.4, max_value=29.1)
        RAIN = st.number_input("Rain", min_value=0.0, max_value=72.5, step=0.1)
        wd = st.number_input("Wind Direction (wd)", min_value=0, max_value=360, step=1)
        WSPM = st.number_input("Wind Speed (WSPM)", min_value=0.0, max_value=13.2, step=0.1)
        station = st.number_input("Station ID", min_value=0, max_value=100, step=1)
        day_of_week = st.number_input("Day of Week (day_of_week)", min_value=0, max_value=6, step=1)

        # Creating input data as DataFrame
        input_data = pd.DataFrame({
            'No': [No],
            'year': [year],
            'month': [month],
            'day': [day],
            'hour': [hour],
            'PM2.5': [PM2_5],
            'PM10': [PM10],
            'SO2': [SO2],
            'NO2': [NO2],
            'CO': [CO],
            'O3': [O3],
            'TEMP': [TEMP],
            'PRES': [PRES],
            'DEWP': [DEWP],
            'RAIN': [RAIN],
            'wd': [wd],
            'WSPM': [WSPM],
            'station': [station],
            'day_of_week': [day_of_week]
        })


        model_choice = st.selectbox("Select model for prediction", ("Decision Tree", "Logistic Regression"))

        # Add calculate button
        if st.button('Calculate AQI'):
            # Predict AQI
            if model_choice == "Decision Tree":
                prediction = predict_aqi(decision_tree_model, input_data)
            else:
                prediction = predict_aqi(logistic_regression_model, input_data)

            # Define AQI categories
            aqi_categories = {
                0: "Good",
                1: "Moderate",
                2: "Unhealthy for Sensitive Groups",
                3: "Unhealthy",
                4: "Very Unhealthy",
                5: "Hazardous",
                6: "Severe"
            }
            
            # Display prediction 
            st.subheader(f"Predicted AQI: {prediction} - {aqi_categories[prediction]}")

    except Exception as e:
        st.error(f"Error in model processing: {str(e)}")

# Data Overview Page
elif page == "Data Overview":
    st.title("Data Overview")
    st.write("Information about the dataset")
    st.write("View the notebook below:")

    # Load and display the HTML file
    html_path = "st20313528.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)
