### This code does the following:

# Loads the data from Google Cloud Storage.
# Saves the initial dataframes to CSV files in a folder.
# Fetches the exchange eur_to_mkd rate from the NBRM API.
# Converts MKD to EUR in the fact_table.
# Merges the fact_table with other dimension tables.
# Drops unnecessary columns from the fact_table.
# Saves the cleaned and merged fact_table to a folder as data_cleaned.csv.



import os
import io
from google.cloud import storage
import requests
import pandas as pd
import datetime
from datetime import datetime, timedelta
import resources as resources

# Set the resources
key_path = resources.key_path
bucket_name = resources.bucket_name
txt1_blob_name = resources.txt1_blob_name
txt2_blob_name = resources.txt2_blob_name
excel_blob_name = resources.excel_blob_name

"""Downloading data from Google Cloud Storage, cleaning and merging it with the data from the API"""

# Connect to Google Cloud with a handler
def initialize_gcs_client(key_path):
    try: 
        # Set the environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        # Initialize the Google Cloud Storage client
        gcs_client = storage.Client()
        return gcs_client
        
    except Exception as e: 
        print(f"Error connecting to Google Cloud: {e}")
        return None

def get_blob_url(gcs_client, bucket_name, blob_name):
    try: 
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Set URL to be valid for 30 days
        expiration_time = timedelta(days=30)
        return blob.generate_signed_url(expiration=expiration_time), blob_name, bucket_name
    except Exception as e:
        print(f"Error getting blob URL: {e}")
        return None, None, None

def load_data_from_gcs(gcs_client, bucket_name, txt1_blob_name, txt2_blob_name, excel_blob_name):
    try: 
        # Get URLs for the blobs
        excel_url, _, _ = get_blob_url(gcs_client, bucket_name, excel_blob_name)
        txt1_url, _, _ = get_blob_url(gcs_client, bucket_name, txt1_blob_name)
        txt2_url, _, _ = get_blob_url(gcs_client, bucket_name, txt2_blob_name)
    
        # Download the files using requests
        excel_response = requests.get(excel_url)
        txt1_response = requests.get(txt1_url)
        txt2_response = requests.get(txt2_url)

        # Check for successful response status
        if excel_response.status_code == 200:
            excel_content = excel_response.content
        else:
            raise Exception(f"Failed to download Excel file from {excel_url}")

        if txt1_response.status_code == 200:
            txt1_content = txt1_response.content
        else:
            raise Exception(f"Failed to download text file from {txt1_url}")

        if txt2_response.status_code == 200:
            txt2_content = txt2_response.content
        else:
            raise Exception(f"Failed to download text file from {txt2_url}")

        print("Reading files...")

        # Read the files into pandas DataFrames directly from response contents for 
        # later processing and for inheriting the schema
        with pd.ExcelFile(io.BytesIO(excel_content)) as excel_file:  
            fact_table = pd.read_excel(excel_file, "Clients - Banking")  
            dim_nationality = pd.read_excel(excel_file, "Nationality")  
            dim_client_name = pd.read_excel(excel_file, "Clients")
    
        dim_banking_contact = pd.read_csv(io.BytesIO(txt1_content), delimiter=';')
        dim_investment_advisor = pd.read_csv(io.BytesIO(txt2_content), delimiter=',')

        return fact_table, dim_nationality, dim_client_name, dim_banking_contact, dim_investment_advisor
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None  # Return a tuple with None values


def save_tables_as_csv(fact_table, dim_nationality, dim_client_name, dim_banking_contact, dim_investment_advisor):
    try:
        # Check if all dataframes are provided
        fact_table ,dim_nationality ,dim_client_name ,dim_banking_contact ,dim_investment_advisor == None, None, None, None, None
        if (fact_table ,dim_nationality ,dim_client_name ,dim_banking_contact ,dim_investment_advisor):
        # Create directories if they don't exist
            os.makedirs('data/csv_data', exist_ok=True)

        def save_csv(dataframe, filename):
            # Define the file path
            filepath = f'data/csv_data/{filename}'
            
            # Check if the file already exists and remove it if it does
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f'Updating file...{filename}')
                
            # Save the dataframe as a CSV file
            dataframe.to_csv(filepath, index=False)
            
            return filepath

        # Save the tables as CSV files and collect their file paths
        print('Downloading raw tables as CSV files...')

        fact_table_path = save_csv(fact_table, 'fact_table.csv')
        dim_nationality_path = save_csv(dim_nationality, 'dim_nationality.csv')
        dim_client_name_path = save_csv(dim_client_name, 'dim_client_name.csv')
        dim_banking_contact_path = save_csv(dim_banking_contact, 'dim_banking_contact.csv')
        dim_investment_advisor_path = save_csv(dim_investment_advisor, 'dim_investment_advisor.csv')

        print(f'Successfully downloaded tables as CSV files.\nYou can find them under the data/csv_data directory.')

    except Exception as e:
        print(f"Error saving data: {e}")
        return None, None, None, None, None  # Return a tuple with None values  
####################################################

# Transfomration functions
def convert_dates(data):
    date_columns = ['Joined Bank','Last Contact', 'Last Meeting']
    data[date_columns] = data[date_columns].apply(pd.to_datetime, unit='D', origin='1899-12-30')
    return data


def diffrences_in_dates(data):
    # Calculate the days passed between the last meating and the last contact
    data['Contact_to_Meeting_Days'] = (data['Last Meeting'] - data['Last Contact']).dt.days

    # Calculate the days since client joined the bank
    data['Bank_Joined_Days'] = (pd.to_datetime('today') - data['Joined Bank']).dt.days
    return data


# Function to clean tables
def clean_tables(data, *args):
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    tables_to_be_cleaned = [*args]
    tables_cleaned = [clean_tables(table) for table in tables_to_be_cleaned]
    return data


def get_rate_from_api():

    try: 
        # Get today's date
        date = datetime.today().strftime('%d.%m.%Y')

        # NBRM API (Macedonian National bank)
        nbrm_url = f'https://www.nbrm.mk/KLServiceNOV/GetExchangeRates?StartDate={date}&EndDate={date}&format=json'

        # Send the GET request
        response = requests.get(nbrm_url)
        # print(response.status_code) # check if successful

        # Parse the JSON response
        if response.status_code == 200:
            kursna_lista = response.json()
            
            eur_to_mkd_rate = None
            for rate in kursna_lista:
                if rate['oznaka'] == 'EUR':
                    eur_to_mkd_rate = rate['sreden']
                    break 
                else:
                    print(f"EUR to MKD rate not found in the response. Error: {response.status_code}.")
            return eur_to_mkd_rate
        eur_to_mkd_rate = 61.5
        return eur_to_mkd_rate
    except Exception as e:
        print(f"Failed to fetch data from the API. Error: {e}. Use the default rate EUR - MKD = 61.5. ")
    

def convert_mkd_to_eur(data):
    # Select all features in MKD
    currency_values_in_MKD = ['Estimated Income', 'Superannuation Savings','Credit Card Balance', 'Bank Loans', 
                        'Bank Deposits','Checking Accounts','Saving Accounts','Business Lending']
    
    # Fetch the today's rate
    eur_to_mkd_rate = get_rate_from_api()

    # Divide all numerical columns by the rate
    data = data.apply(lambda x: x / eur_to_mkd_rate if x.name in currency_values_in_MKD else x)
    return data


def merge_fact_dim_tables(df, dim_client, dim_nat, dim_b_contact, dim_inv_adv):
    # In order to preserve all the observations in the fact table, count all the occurrences 
    # of a ClientID in both tables 
    df['Occurrence'] = df.groupby('Client ID').cumcount() + 1 
    dim_client['Occurrence'] = dim_client.groupby('Client ID').cumcount() + 1

    # Add the names to IDs by the order of occurrence
    fact_data = df.merge(dim_client, how='left', on=['Client ID', 'Occurrence']) \
                  .merge(dim_nat, on='NationalityID', how='left') \
                  .merge(dim_b_contact, on='Banking Contact ID', how='left') \
                  .merge(dim_inv_adv, left_on='AdvisorID', right_on='ID', how='left') 
    
    # Drop the 'Occurrence' and 'ID' columns
    fact_data.drop(columns=['Occurrence'], axis=1, inplace=True)
    fact_data.drop(columns=['ID'], axis=1, inplace=True)
    fact_data.reset_index(drop=True, inplace=True)
    
    return fact_data


# Drop the columns with high cardinality
def drop_data(data): 
    columns=['Client ID', 'Joined Bank', 'Banking Contact ID', 'NationalityID', 'AdvisorID', 
            'Last Contact', 'Last Meeting', 'Risk Weighting', 'Name', 'Banking Contact', 'Investment Advisor', 'Bank Deposits' ]
    data.drop(columns = columns, axis=1, inplace=True)
    data.reset_index(drop=True)
    return data


def save_data(data, file_path='data/data_cleaned.csv'):
    
    if 'data_cleaned.csv' in os.listdir('data'):
        os.remove('data/data_cleaned.csv')
        print('Updating data...')

    data.to_csv(file_path, index=False)
    
    return data


"""Calling the main funtion"""

# Main function
def main():

    # Initialize the Google Cloud Storage client
    gcs_client = initialize_gcs_client(resources.key_path)
    if gcs_client:
        print('Checking Google Cloud Storage...')  

        # Get the dataframes
        fact_table, dim_nationality, dim_client_name, dim_banking_contact, dim_investment_advisor = \
            load_data_from_gcs(gcs_client, bucket_name, txt1_blob_name, txt2_blob_name, excel_blob_name)

        if fact_table is not None:
            
            print('Making data transformations...')
        
            # Convert dates
            convert_dates(fact_table) 

            # Calculate the diffrences in dates
            diffrences_in_dates(fact_table)

            # Clean the tables
            clean_tables(fact_table, dim_nationality, dim_client_name, dim_banking_contact, dim_investment_advisor)
                       
            # Get the rate from the API
            get_rate_from_api()

            # Convert MKD to EUR
            convert_mkd_to_eur(fact_table)

            # Merge the tables
            merge_fact_dim_tables(fact_table, dim_client_name, dim_nationality, dim_banking_contact, dim_investment_advisor)
            
            # Drop the data
            drop_data(merge_fact_dim_tables(fact_table, dim_client_name, dim_nationality, dim_banking_contact, dim_investment_advisor))
            print('Data transformations completed.')

            # Save the cleaned fact_table to the same path
            save_data(drop_data(merge_fact_dim_tables(fact_table, dim_client_name, dim_nationality, dim_banking_contact, dim_investment_advisor)))
            print(f'Data ready to use. Saved as data_cleaned.csv in the data directory.\n \n')

            # Save the tables as CSV files
            save_tables_as_csv(fact_table, dim_nationality, dim_client_name, dim_banking_contact, dim_investment_advisor)  

        else:
            print("Failed to load data from Google Cloud Storage.")  

    else: 
        print("Error initializing Google Cloud Storage client")



if __name__ == "__main__":
    main()

    