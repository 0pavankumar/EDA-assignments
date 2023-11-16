'''
Name: PAVAN KUMAR 
Batch Id: AUG 4th 
Topic: Data Pre-Processing


feature analysis

1.index - index - quantitative - irrelevant index doesnot provide useful info

2.ANIMALS - names - nominal - relevant data provide info

3.gender - gender - categorical - relevant data provide info

4. Homly - YES or NO - categorical - relevant data provide info

5.Types - A, B , C, D-  categorical - relevant data provide info

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
df = pd.read_csv(r"C:\Users\1pava\Documents\360digitmg\EDA\DATA SETS\Animal_category.csv")

# Count and display the number of duplicate rows in the DataFrame
df.duplicated().sum()

# Display the count of unique values in the 'Types' column
df['Types'].value_counts()

# Display the count of unique values in the 'Animals' column
df['Animals'].value_counts()

# Drop the 'Index' column from the DataFrame
df = df.drop(columns="Index", axis=1)

# encoding
#method 1
# Initialize a LabelEncoder for encoding categorical variables
from sklearn.preprocessing import LabelEncoder as le
le = le()

# Create a dictionary to store the mapping of column names to their assigned values
column_name = {}

# Encode each object-type column and capture the mapping
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
        column_name[col] = dict(zip(df[col], le.inverse_transform(df[col]))

# Print the dictionary containing the mapping of column names to their assigned values
print(column_name)

#method 2
# Perform one-hot encoding on the categorical columns
df_encoded = pd.get_dummies(df, columns=["Animals", "Gender", "Homly", "Types"])

# Display the DataFrame with one-hot encoded columns
print(df_encoded)


#method 3

from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Select the columns you want to one-hot encode
columns_to_encode = ["Animals", "Gender", "Homly", "Types"]

# Fit and transform the selected columns using the OneHotEncoder
encoded_data = encoder.fit_transform(df[columns_to_encode])

# Create a DataFrame from the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(columns_to_encode))

# Concatenate the encoded DataFrame with the original DataFrame, excluding the columns you encoded
df_encoded = pd.concat([df, encoded_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(columns=columns_to_encode)

# Display the DataFrame with one-hot encoded columns
print(df_encoded)




















