### This code does the following:

# Fetch the data from ETL
# Encode the categorical features using dummies, ordinal and target Encoding
# Scale the numerical features (including ordinal and target encoded)
# Trains Unsupervised Anomaly Detection model with Isolation Forest
# Compares point and contextual anomalies
# Cluster the data with DBSCAN and fetched anomalies into clusters
# Get predictions


# Import nesseccary libriaries for data handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import mlflow


# Feature Processing and Model (Scikit-learn processing, etc. )
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.inspection import DecisionBoundaryDisplay

# Model registering
import os

# API 
import io
import requests
from google.cloud import storage, aiplatform


# Importing the ELT_app module for data ingestion
from ETL import main as data_ingest

# Call the function
data_ingest()

data_cleaned = pd.read_csv('data/data_cleaned.csv')

# Data Encoding
# Convert categorical features into numerical using OrdinalEncoder
def ordinal_encoding(data):

    # Set order of values in each column by their rank
    # for [H' ,'MR','L' meaning 'High','Mid Range','Low' respectively ]
    fee_order = ['H','MR','L']
    loyalty_order = ['Jade', 'Silver', 'Gold', 'Platinum'] 
    
    # Initialize the OrdinalEncoder with the custom order ordinal_encoder
    ordinal_encoder = OrdinalEncoder(categories=[fee_order, loyalty_order])
    data[['Fee_Rank','Loyalty_Rank']] = ordinal_encoder.fit_transform(data[['Fee Structure ID','Loyalty Classification']])
    data[['Fee_Rank','Loyalty_Rank']] = data[['Fee_Rank','Loyalty_Rank']].astype(int)
    
    print("Ordinal Encoding Completed")
    return data 
    


# Encode data
# Create dummy columns for categorical features 
def dummy_encoding(data):
    # Create dummies (True - False) columns for features with low cardinality
    data = pd.get_dummies(data, columns = ['Sex', 'Nationality', 'Properties Owned', 'Banking Relationship']) 
    print("Dummy Encoding Completed")
    print(data.columns)
    return data


# Encode features with targeted feature
def target_encoding(data, categorical_col, target_col):

    # Assuming that Income is mostly infuenced by Occupation, the values of Occupation feature
    # with medium cardinality are mapped with Income mean
    target_mean = data.groupby(categorical_col)[target_col].mean()
    data_encoded = data.copy()
    data[categorical_col + '_encoded'] = data[categorical_col].map(target_mean)
    
    print("Target Encoding Completed")
    return data  


# Drop columns that are not needed for the model
def drop_columns(data):
    data.drop(columns = ['Fee Structure ID', 'Loyalty Classification', 'Occupation'], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    print("Columns dropped")
    return data


# Standardize the continuous features 
# The values in Properties Owned and Amount of Credit Card can be treated as already having a rank, no need to scale
def standardize_data(data):
    scaler = StandardScaler() 
    values_to_be_scaled = ['Age', 'Estimated Income', 'Superannuation Savings',
       'Amount of Credit Cards', 'Credit Card Balance', 'Bank Loans',
       'Checking Accounts', 'Saving Accounts', 'Foreign Currency Account',
       'Business Lending', 'Contact_to_Meeting_Days', 'Bank_Joined_Days',
       'Fee_Rank', 'Loyalty_Rank', 'Occupation_encoded']

    data[values_to_be_scaled] = scaler.fit_transform(data[values_to_be_scaled])
    
    print("Standardization completed")

    print(data.head(3))
    return data


# Train Unsupervised Anomaly Detection Model with Isolation Forest
def train_model(data):  

    # Model Configuration
    print('Training model in progress...')

    model_name = "Anomaly_Detection_Model"
    # Define the tracking directory
    tracking_dir = r'C:\Users\user\PYTHON_Data Science\Bank_Anomaly_Prediction_with_Data_Engineering\mlruns'
    

    # Create the tracking directory if it doesn't exist
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)

    # Set the tracking URI
    tracking_URI = mlflow.set_tracking_uri(f'file:///{tracking_dir}')

    # Initialize the Hyperparameter Tuned Isolation Forest model
    experiment_name = f"Isolation_Forest_Model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    mlflow.set_experiment(experiment_name) 

    with mlflow.start_run() as run:

        experiment_id = run.info.experiment_id
        print(f"Creating Experiment: {experiment_id}")

        params = {
            'n_estimators': 100,
            'max_samples' : 256, 
            'max_features' : 1.0,
            'contamination': 0.01,
            'bootstrap': True,
            'n_jobs' : -1,
            'random_state': 42
        }

        model = IsolationForest(**params).fit(data)

        
         # Log the anomaly scores as metrics
        anomaly_predictions = model.predict(data)
        anomaly_scores = model.decision_function(data)
        for i, score in enumerate(anomaly_scores):
            mlflow.log_metric(f"anomaly_score_sample_{i}", score)

        # Log model and metrics
        mlflow.sklearn.log_model(model,"model",registered_model_name=model_name, input_example=data)

        # Print the model path (just example path as mlflow saves under artifacts)
        model_path = os.path.join(tracking_dir, experiment_name, "model")
        print(f"Model path: {model_path}")

    print(f"Training Model {model_name}, experiment ID: {experiment_id} completed successfully!")
    return model
  

def main():
    ordinal_encoded_data = ordinal_encoding(data_cleaned)
    
    dummy_data = dummy_encoding(ordinal_encoded_data)
   
    target_encoded_data = target_encoding(dummy_data, 'Occupation', 'Estimated Income')
   
    data_dropped = drop_columns(target_encoded_data) 
   
    scaled_data = standardize_data(data_dropped)

    trained_model = train_model(scaled_data)
               

if __name__ == '__main__':
    main()