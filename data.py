import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the dataset (assuming it's in CSV format)
df = pd.read_csv('car_dataset.csv')

# Separate the 'Car' and 'Car ID' column (which we want to preserve as the target)
car_column = df['Car']
index_column = df['Car ID']

# Separate numerical and categorical columns
numerical_columns = ['Price ($)', 'Year of Manufacture', 'Resale Value ($)', 
                     'Fuel Economy (km/l)', 'Performance Rating', 'User Rating', 
                     'Safety Rating', 'Comfort Level', 'Maintenance Cost ($/yr)', 
                     'Warranty Period (years)', 'Seating Capacity', 'Battery Capacity (kWh)']

categorical_columns = ['Body Type', 'Engine Type', 'Transmission Type', 'Drive Type']

# Handle missing values for numerical columns (fill missing Battery Capacity with median)
df['Battery Capacity (kWh)'] = df['Battery Capacity (kWh)'].fillna(df['Battery Capacity (kWh)'].median())

# Handle missing values for categorical columns by filling with the mode
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates if any
df = df.drop_duplicates()

# Normalize numerical data using Min-Max Scaling
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_columns = encoder.fit_transform(df[categorical_columns])
df_encoded = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the encoded columns with the normalized numerical columns
df_final = pd.concat([df[numerical_columns], df_encoded], axis=1)

# Add back the 'Car' column (target) to the cleaned dataset
df_final['Car'] = car_column
df_final['Car ID'] = index_column
df_final = df_final.drop_duplicates()

# Save the cleaned and processed dataset
df_final.to_csv('cleaned_dataset.csv', index=False)

print("Dataset cleaning and preparation completed.")
