import streamlit as st
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the trained model using MLflow
tracking_dir = r'C:\Users\user\PYTHON_Data Science\Bank_Anomaly_Prediction_with_Data_Engineering\mlruns'
model_name = "Anomaly_Detection_Model"
model_version = "latest"    # Use 'latest' to get the most recent model version
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)


# Define Streamlit app interface
st.title('Anomaly Detection App')
st.write("Upload a CSV file containing the data you'd like to check for anomalies.")

uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Here is a preview of the data you've uploaded:")
    st.write(data.head())

    # Prepare data for prediction
    data_for_prediction = data.copy()

    # Make predictions
    # Convert inputs to DataFrame for easier manipulation
    data_for_prediction = pd.DataFrame(data)

    # Extract features for scaling
    values_to_be_scaled = ['Age', 'Estimated Income', 'Superannuation Savings',
        'Amount of Credit Cards', 'Credit Card Balance', 'Bank Loans',
        'Checking Accounts', 'Saving Accounts', 'Foreign Currency Account',
        'Business Lending', 'Contact_to_Meeting_Days', 'Bank_Joined_Days',
        'Fee_Rank', 'Loyalty_Rank', 'Occupation_encoded']

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the continuous columns
    data_for_prediction[values_to_be_scaled] = scaler.fit_transform(data_for_prediction[values_to_be_scaled])

    # Make predictions on processed input data
    prediction = model.predict(data_for_prediction)
    data['Is_Anomaly'] = prediction

    # Display predictions
    st.write("Here are the predictions. If Is_Anomaly equals to - 1, input data is predicted as anomaly")
    st.write(data)

    # Highlight anomalies
    st.write(f'Is this an anomaly? {"No" if prediction == 1 else " YES, please be careful with this Bank client"}')

# Run the Streamlit app using: streamlit run "C:\Users\user\PYTHON_Data Science\Bank_Anomaly_Prediction_with_Data_Engineering\app.py"
