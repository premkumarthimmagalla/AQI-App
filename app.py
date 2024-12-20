import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Comprehensive Logging Function
def log_preprocessing_details(input_data, scaler):
    """
    Detailed logging of preprocessing steps
    """
    st.write("Raw Input Data:", input_data)
    
    # Convert input to numpy array
    input_array = np.array(input_data).reshape(1, -1)
    
    # Log array details
    st.write("Input Array Shape:", input_array.shape)
    st.write("Input Array Dtype:", input_array.dtype)
    
    # Log scaler details
    if scaler is not None:
        st.write("Scaler Type:", type(scaler))
        
        # Check if scaler has been fitted
        if hasattr(scaler, 'mean_'):
            st.write("Scaler Mean:", scaler.mean_)
            st.write("Scaler Scale:", scaler.scale_)
            
            # Log how many features scaler is expecting
            st.write("Scaler Number of Features:", len(scaler.mean_))
    
    # Attempt preprocessing with detailed error handling
    try:
        # Use transform if possible
        if hasattr(scaler, 'transform'):
            processed_data = scaler.transform(input_array)
        else:
            # Fallback to fit_transform
            processed_data = scaler.fit_transform(input_array)
        
        st.write("Processed Data:", processed_data)
        st.write("Processed Data Shape:", processed_data.shape)
        
        return processed_data
    except Exception as e:
        st.error(f"Preprocessing Error: {str(e)}")
        return input_array

# Predict Function with Extensive Logging
def predict_aqi(model, input_data, scaler=None):
    try:
        # Log preprocessing details
        processed_data = log_preprocessing_details(input_data, scaler)
        
        # Predict with additional logging
        st.write("Model Type:", type(model))
        
        # Check model attributes
        st.write("Model Attributes:")
        st.write("Classes:", getattr(model, 'classes_', 'Not Available'))
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            st.write("Prediction Probabilities:", probabilities)
            
            # Log class-wise probabilities
            for i, prob in enumerate(probabilities[0]):
                st.write(f"Class {i} Probability: {prob}")
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        return None

# Main Streamlit App (Same as previous version)
def main():
    # Sidebar for navigation
    st.sidebar.title("Debugging Mode")
    page = st.sidebar.radio("Go to", ["Model Prediction", "Model Inspection"])

    if page == "Model Prediction":
        st.title("AQI Prediction with Detailed Debugging")

        # Load models with error handling
        try:
            decision_tree_model = joblib.load("DecisionTreeClassifier.joblib")
            logistic_regression_model = joblib.load("LogisticRegression.joblib")
        except Exception as e:
            st.error(f"Model Loading Error: {str(e)}")
            decision_tree_model = None
            logistic_regression_model = None

        # Try loading scaler
        try:
            scaler = joblib.load("standard_scaler.joblib")
            st.success("Scaler loaded successfully")
        except Exception as e:
            st.warning(f"Scaler Loading Error: {str(e)}")
            scaler = StandardScaler()

        # Input fields (simplified for debugging)
        st.write("### Enter Input Features")
        
        # Create input fields dynamically
        feature_names = [
            'No', 'year', 'month', 'day', 'hour', 
            'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 
            'TEMP', 'PRES', 'DEWP', 'RAIN', 
            'wd', 'WSPM', 'station', 'day_of_week'
        ]
        
        input_data = []
        for feature in feature_names:
            # Determine appropriate input method based on feature
            if feature in ['No', 'year', 'month', 'day', 'hour', 'station', 'day_of_week']:
                value = st.number_input(feature, value=0)
            else:
                value = st.number_input(feature, value=0.0, format="%.2f")
            input_data.append(value)

        # Model selection
        model_choice = st.selectbox("Select model for prediction", 
                                    ("Decision Tree", "Logistic Regression"))

        # Predict button
        if st.button("Predict AQI"):
            st.write("### Prediction Details")
            
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

    elif page == "Model Inspection":
        st.title("Model and Scaler Inspection")
        
        # File uploader for model and scaler
        uploaded_model = st.file_uploader("Upload Model File", type=['.joblib', '.pkl'])
        uploaded_scaler = st.file_uploader("Upload Scaler File", type=['.joblib', '.pkl'])

        # Inspect uploaded files
        if uploaded_model is not None:
            try:
                model = joblib.load(uploaded_model)
                st.write("Model Type:", type(model))
                
                # Check model attributes
                st.write("Model Attributes:")
                st.write("Classes:", getattr(model, 'classes_', 'Not Available'))
                
                if hasattr(model, 'feature_importances_'):
                    st.write("Feature Importances:", model.feature_importances_)
            except Exception as e:
                st.error(f"Model Inspection Error: {str(e)}")

        if uploaded_scaler is not None:
            try:
                scaler = joblib.load(uploaded_scaler)
                st.write("Scaler Type:", type(scaler))
                
                # Check scaler attributes
                if hasattr(scaler, 'mean_'):
                    st.write("Scaler Mean:", scaler.mean_)
                    st.write("Scaler Scale:", scaler.scale_)
            except Exception as e:
                st.error(f"Scaler Inspection Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
