import streamlit as st
import joblib
import pandas as pd

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model", "Data Overview"])

# EDA Page
if page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("1. Median pollutant levels per month for each year")
    st.image('median pollutant levels per month for each year.png')

    st.markdown("2. Median pollutant levels per year")
    st.image('median pollutant levels per year.png')

    st.markdown("3. Pearson correlation matrix")
    st.image('Pearson correlation matrix.png')

    st.markdown("4. PM 2.5")
    st.image('PM 2.5.png')

    st.markdown("5. Pollutant Levels by Day of the Week")
    st.image('Pollutant Levels by Day of the Week.png')

    st.markdown("6. Pollutant levels by hour of day")
    st.image('pollutant levels by hour of day.png')

# Model Page
elif page == "Model":
    st.title("Model")
    # Load models
    decision_tree_model = joblib.load("DecisionTreeClassifier.joblib")
    logistic_regression_model = joblib.load("LogisticRegression.joblib")

    # Function to predict AQI based on selected model
    def predict_aqi(model, input_data):
        prediction = model.predict(input_data)
        return prediction[0]

    # Streamlit User Inputs for AQI Prediction
    st.title("AQI Prediction")

    # Input fields for the features
    No = st.selectbox("NO",[0,1])
    year=st.selectbox('Pick year', [2013,2014,2015,2016,2017])
    # year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
    month = st.selectbox("Month",[1,2,3,4,5,6,7,8,9,10,11,12])
    day = st.slider('Day', 0, 31)
    hour = st.slider('Hour', 0, 24)
    PM2_5 = st.number_input("PM2.5", min_value=0.0, max_value=999.0,step=0.1)
    PM10 = st.number_input("PM10", min_value=2.0, max_value=999.0,step=0.1)
    SO2 = st.number_input("SO2", min_value=2.8,max_value=999.0,step=0.1)
    NO2 = st.number_input("NO2", min_value=1.0, max_value=280.0)
    CO = st.number_input("CO", min_value=100.0, max_value=1000.0,step=0.5)
    O3 = st.number_input("O3", min_value=0.2, max_value=1071.0, step=0.1)
    TEMP = st.number_input("Temperature (TEMP)", min_value=-18.0, max_value=42.0,step=0.1)
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

    # Model selection
    model_choice = st.selectbox("Select model for prediction", ("Decision Tree", "Logistic Regression"))

    # Predict AQI
    if model_choice == "Decision Tree":
        prediction = predict_aqi(decision_tree_model, input_data)
    elif model_choice == "Logistic Regression":
        prediction = predict_aqi(logistic_regression_model, input_data)

    # Display prediction
    st.subheader(f"Predicted AQI: {prediction}")
    
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
