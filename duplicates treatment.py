
import pandas as pd
import numpy as np
df= pd.read_csv("C:/Users/1pava/Documents/360digitmg/EDA/DATA SETS/Online Retail.csv", encoding= 'unicode_escape')#, 
df

df.head()
df.tail() 

df.isna().sum()
df.isnull().sum()
df.info()
df.shape

df.describe()
df.describe

df.dtypes
df["UnitPrice"] = df["UnitPrice"].astype('int')
df.info()
df.duplicated().sum()

#imputation technique
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df['CustomerID']= pd.DataFrame(mean_imputer.fit_transform(df[['CustomerID']]))
df['CustomerID'].isnull().sum()

df['CustomerID'] = df['CustomerID'].astype('int')
df['CustomerID'].dtype

## Duplication

"Q2. Check for the duplicate values, and handle the duplicate values (ex. drop)"

#Identify duplicates records in the data
duplicate = df.duplicated()
sum(duplicate) 

#Removing Duplicates
data1 = df.drop_duplicates() 

"Q3. Do the data analysis (EDA)?"
"Such as histogram, boxplot, scatterplot etc "
import seaborn as sns
#Graphical Representation
numeric_columns = df.columns[1:-1]
# Create bar plots for each numeric column

df.columns


#Quantity
plt.hist(df.Quantity);plt.show() #histogram

#UnitPrice
plt.hist(df.UnitPrice)#;plt.show() #histogram

#CustomerID
plt.hist(df.CustomerID);plt.show() #histogram

#all boxplots

df.plot(kind= 'box', sharey = False, subplots= True, fig = (2,2))