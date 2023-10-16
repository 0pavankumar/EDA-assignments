# Outlier Treatment 

# 1st of all import all the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("C:/Users/1pava/Documents/360digitmg/EDA/DATA SETS/Boston.csv")
df

df.dtypes # finding data types
df.isna().sum() # finding is there any null values in the dataset or not but there is no null values in the dataset.

# let's find outliers 
sns.boxplot(df.crim);plt.title('Boxplot');plt.show()
sns.boxplot(df.zn);plt.title('Boxplot');plt.show()
sns.boxplot(df.indus);plt.title('Boxplot');plt.show() # no outliers present
sns.boxplot(df.chas);plt.title('Boxplot');plt.show() # we leave this because it is a categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
sns.boxplot(df.nox);plt.title('Boxplot');plt.show() # no outliers present
sns.boxplot(df.rm);plt.title('Boxplot');plt.show()
sns.boxplot(df.age);plt.title('Boxplot');plt.show() # no outliers present
sns.boxplot(df.dis);plt.title('Boxplot');plt.show()
sns.boxplot(df.rad);plt.title('Boxplot');plt.show() # no putliers present
sns.boxplot(df.tax);plt.title('Boxplot');plt.show()  # no putliers present
sns.boxplot(df.ptratio);plt.title('Boxplot');plt.show() # it's a binary type data
sns.boxplot(df.black);plt.title('Boxplot');plt.show()
sns.boxplot(df.lstat);plt.title('Boxplot');plt.show()
sns.boxplot(df.medv);plt.title('Boxplot');plt.show()

###### So, we have only 7 variables which has outliers
# 1) crim 2) zn 3) rm 4) dis 5) black 6) lstat 7) medv

#1) crim
# Detection of Outliers
IQR = df['crim'].quantile(0.75) - df['crim'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['crim'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['crim'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trimm the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_df = np.where(df['crim'] > upper_limit,True,np.where(df['crim'] < lower_limit,True,False))
# if value is greater than upper limit consider it as outliers and if the value is less than lower limit consider it as outliers
df_trimmed = df.loc[~(outliers_df),] # ~ means not - it shows all false value (not outliers)
df.shape, df_trimmed.shape # we trim 12 outliers

sns.boxplot(df_trimmed.crim);plt.title('Boxplot');plt.show()

#we still have outiers

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['crim'] > upper_limit, upper_limit,
                                         np.where(df['crim'] < lower_limit, lower_limit,
                                                  df['crim'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

#we see no outiers

"2) zn"

# Detection of Outliers
IQR = df['zn'].quantile(0.75) - df['zn'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['zn'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['zn'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trimm the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_df = np.where(df['zn'] > upper_limit,True,np.where(df['zn'] < lower_limit,True,False))
# if value is greater than upper limit consider it as outliers and if the value is less than lower limit consider it as outliers
df_trimmed = df.loc[~(outliers_df),] # ~ means not - it shows all false value (not outliers)
df.shape, df_trimmed.shape # we trim 49 outliers

sns.boxplot(df_trimmed.zn);plt.title('Boxplot');plt.show()

#we still have outiers

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
df['df1_replaced']= pd.DataFrame(np.where(df['zn'] > upper_limit, upper_limit,
                                         np.where(df['zn'] < lower_limit, lower_limit,
                                                  df['zn'])))
                                 
sns.boxplot(df.df1_replaced);plt.title('Boxplot');plt.show()

#we see no outiers

"3) rm"

# Detection of Outliers
IQR = df['rm'].quantile(0.75) - df['rm'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['rm'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['rm'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

####################### Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
df['df2_replaced']= pd.DataFrame(np.where(df['rm'] > upper_limit, upper_limit,
                                         np.where(df['rm'] < lower_limit, lower_limit,
                                                  df['rm'])))
                                 
sns.boxplot(df.df2_replaced);plt.title('Boxplot');plt.show()

#we see no outliers

"4) dis"

# Detection of Outliers
IQR = df['dis'].quantile(0.75) - df['dis'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['dis'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['dis'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

###################### 3. Winsorization #####################################
pip install feature_engine
# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['dis'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['dis']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.dis);plt.title('Boxplot');plt.show()

"5) black"

# Detection of Outliers
IQR = df['black'].quantile(0.75) - df['black'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['black'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['black'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['black'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['black']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.black);plt.title('Boxplot');plt.show()

"6) lstat"

# Detection of Outliers
IQR = df['lstat'].quantile(0.75) - df['lstat'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['lstat'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['lstat'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['lstat'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['lstat']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.lstat);plt.title('Boxplot');plt.show()

"7) medv"

# Detection of Outliers
IQR = df['medv'].quantile(0.75) - df['medv'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['medv'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['medv'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['medv'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['medv']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.medv);plt.title('Boxplot');plt.show()

# Now Our data is Outlier free

