import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df= pd.read_csv("C:/Users/1pava/Documents/360digitmg/EDA/DATA SETS/claimants.csv")
df.info()

df.isnull().sum()


# Mean Imputer 

mi = SimpleImputer(missing_values=np.nan, strategy="mean")
df["CLMAGE"] = pd.DataFrame(mi.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isnull().sum()



# median Imputation technique
me= mi = SimpleImputer(missing_values=np.nan, strategy="median")
df["CLMAGE"] = pd.DataFrame(mi.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isnull().sum()


# mode imputation technique
mo = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df["CLMSEX"] = pd.DataFrame(mo.fit_transform(df[["CLMSEX"]]))
df["CLMINSUR"] = pd.DataFrame(mo.fit_transform(df[["CLMINSUR"]]))
df["SEATBELT"] = pd.DataFrame(mo.fit_transform(df[["SEATBELT"]]))

df.info()

























