import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to load models with error handling
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

# Function to preprocess input data
def preprocess_input(input_data, scaler=None):
    """
    Preprocess input data using the provided scaler
    If no scaler is provided, use StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()
    
    # Ensure input is converted to numpy array and reshaped if needed
    input_array = np.array(input_data).reshape(1, -1)
    
    try:
        # If scaler is fitted, transform the data
        if hasattr(scaler, 'scale_'):
            return scaler.transform(input_array)
        else:
            # If scaler is not fitted, fit and transform
            return scaler.fit_transform(input_array)
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return input_array

# Function to predict AQI with error handling and preprocessing
def predict_aqi(model, input_data, scaler=None):
    try:
        # Preprocess input data
        processed_data = preprocess_input(input_data, scaler)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Additional debugging information
        st.write("Raw Input:", input_data)
        st.write("Processed Input:", processed_data)
        
        # If the model has prediction probabilities, show them
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            st.write("Prediction Probabilities:", probabilities)
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main Streamlit App
def main():
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
        st.title("AQI Prediction")

        # Load models
        decision_tree_model = load_model("DecisionTreeClassifier.joblib")
        logistic_regression_model = load_model("LogisticRegression.joblib")
        
        # Load scaler (if available)
        try:
            scaler = joblib.load("standard_scaler.joblib")
        except Exception:
            st.warning("No pre-fitted scaler found. Using default scaling.")
            scaler = StandardScaler()

        # Input fields for the features
        col1, col2 = st.columns(2)

        with col1:
            # First column of inputs
            No = st.selectbox("NO", [0, 1])
            year = st.selectbox('Pick year', [2013, 2014, 2015, 2016, 2017])
            month = st.selectbox("Month", list(range(1, 13)))
            day = st.slider('Day', 0, 31)
            hour = st.slider('Hour', 0, 24)
            PM2_5 = st.number_input("PM2.5", min_value=0.0, max_value=999.0, step=0.1)
            PM10 = st.number_input("PM10", min_value=2.0, max_value=999.0, step=0.1)
            SO2 = st.number_input("SO2", min_value=2.8, max_value=999.0, step=0.1)

        with col2:
            # Second column of inputs
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

        # Creating input data as list
        input_data = [
            No, year, month, day, hour, 
            PM2_5, PM10, SO2, NO2, CO, O3, 
            TEMP, PRES, DEWP, RAIN, 
            wd, WSPM, station, day_of_week
        ]

        # Model selection
        model_choice = st.selectbox("Select model for prediction", 
                                    ("Decision Tree", "Logistic Regression"))

        # Predict button
        if st.button("Predict AQI"):
            # Choose the model
            if model_choice == "Decision Tree" and decision_tree_model:
                prediction = predict_aqi(decision_tree_model, input_data, scaler)
            elif model_choice == "Logistic Regression" and logistic_regression_model:
                prediction = predict_aqi(logistic_regression_model, input_data, scaler)
            else:
                st.error("Model not loaded correctly")
                prediction = None

            # Display prediction
            if prediction is not None:
                st.subheader(f"Predicted AQI: {prediction}")

    # Data Overview Page
    elif page == "Data Overview":
        st.title("Data Overview")
        st.write("Information about the dataset")
        st.write("View the notebook below:")

        # Load and display the HTML file
        try:
            html_path = "st20313528.html"
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=700, scrolling=True)
        except Exception as e:
            st.error(f"Error loading HTML file: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
