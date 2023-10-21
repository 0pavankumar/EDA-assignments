# Missing Values Imputation

import numpy as np
import pandas as pd

# load the dataset
claimants_data = pd.read_csv(r"C:\Users\D\Desktop\New Assignments  Keys\Datasets\claimants.csv")
claimants_data

# check for count of NA'sin each column
claimants_data.isna().sum()

###### There are 4 columns that have missing data ---Create an imputer object that fills 'Nan' values of CLMSEX,CLMINSUR,SEATBELT,CLMAGE
###### Mean and Median imputer are used for numeric data (CLMAGE)
###### mode is used for discrete (categorical) data (CLMSEX,CLMINSUR,SEATBELT)

# for Mean,Meadian,Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
claimants_data["CLMAGE"] = pd.DataFrame(mean_imputer.fit_transform(claimants_data[["CLMAGE"]]))
claimants_data["CLMAGE"].isnull().sum() # all 2 records replaced by mean 

# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
claimants_data["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(claimants_data[['CLMAGE']]))
claimants_data["CLMAGE"].isnull().sum() # all 2 records replaced by median 

# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
claimants_data["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(claimants_data[['CLMSEX']]))
claimants_data["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(claimants_data[['CLMINSUR']]))
claimants_data["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(claimants_data[['SEATBELT']]))
claimants_data.isnull().sum() 

###### Now, finally our data has no NA values

