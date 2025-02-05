import pandas as pd

# Input data
inputs = [
    {
        'Age': 42,
        'Estimated Income': 137256.32,
        'Superannuation Savings': 49416.00,
        'Amount of Credit Cards': 2,
        'Credit Card Balance': 2866.23,
        'Bank Loans': 856.56,
        'Checking Accounts': 1001.27,
        'Saving Accounts': 25648.23,
        'Foreign Currency Account': 79562.46,
        'Business Lending': 564.89,
        'Contact_to_Meeting_Days': 76,
        'Bank_Joined_Days': 6524,
        'Fee_Rank': 0,
        'Loyalty_Rank': 1,
        'Sex_F': 1,
        'Sex_M': 0,
        'Nationality_Asian': 0,
        'Nationality_European': 1,
        'Nationality_Indian': 0,
        'Nationality_Maori': 0,
        'Nationality_Pacific Islander': 0,
        'Properties Owned_0.0': 0,
        'Properties Owned_1.0': 1,
        'Properties Owned_2.0': 0,
        'Properties Owned_3.0': 0,
        'Banking Relationship_Commercial': 1,
        'Banking Relationship_Institutional': 0,
        'Banking Relationship_Investments Only': 0,
        'Banking Relationship_Private Bank': 0,
        'Banking Relationship_Retail': 0,
        'Occupation_encoded': 166458.62
    }
]

# Create DataFrame
df = pd.DataFrame(inputs)

# Save to CSV
df.to_csv('input_data.csv', index=False)
print("CSV file 'input_data.csv' created successfully!")
