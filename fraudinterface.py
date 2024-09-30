# fraud_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Step 1: Load the pre-trained models and scaler
@st.cache_resource
def load_models():
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('gradient_boosting_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
    except:
        xgb_model = None
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return rf_model, gb_model, xgb_model, scaler

# Step 2: Predict function
def predict_fraud(model, scaler, input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction[0], probability

# Step 3: Streamlit interface
def main():
    st.title("Credit Card Fraud Detection System")
    
    # Step 3.1: Load pre-trained models and scaler
    rf_model, gb_model, xgb_model, scaler = load_models()
    
    # Step 3.2: Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_options = ["Random Forest", "Gradient Boosting"]
    if xgb_model is not None:
        model_options.append("XGBoost")
    selected_model = st.sidebar.selectbox("Choose a model", model_options)
    
    # Step 3.3: Select model
    if selected_model == "Random Forest":
        model = rf_model
    elif selected_model == "Gradient Boosting":
        model = gb_model
    else:
        model = xgb_model
    
    # Step 3.4: User input for prediction
    st.subheader("Enter Transaction Details")
    
    # Create input fields for each feature
    input_data = {}
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    for feature in feature_names:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, format="%.6f")
    
    # Create a button to trigger prediction
    if st.button("Predict"):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction, probability = predict_fraud(model, scaler, input_df)
        
        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Fraud Detected! Probability: {probability:.2%}")
        else:
            st.success(f"No Fraud Detected. Probability of fraud: {probability:.2%}")
    
    # Step 3.5: Display feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    fig = go.Figure(go.Bar(x=feature_importance['importance'], y=feature_importance['feature'], orientation='h'))
    fig.update_layout(title=f"{selected_model} - Top 10 Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()