import pandas as pd 
import matplotlib as plt

file_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


df = pd.read_csv(file_name, names = headers)
print(df.head())

# As we can see, several question marks appeared in the dataframe; those are missing values which may hinder our further analysis.

# So, how do we identify all those missing values and deal with them?

# Steps for working with missing data:
#1. identify missing data
#2. deal with missing data
#3. correct data format
##1--->identify the missing values
print(df.head())

"""Convert "?" to NaN
In the car dataset, missing data comes with the question mark "?". We replace "?" with NaN (Not a Number), 
which is Python's default missing value marker, for reasons of computational speed and convenience. Here we use the function:
.replace(A, B, inplace = True) 
to replace A by B"""
import numpy as np
df.replace("?",np.nan, inplace = True)
print(df.head())

missing_data = df.isnull()
print(missing_data)   #"True" stands for missing value, while "False" stands for not missing value.
print(missing_data.columns.values.tolist())
for columns in missing_data.columns.values.tolist():
    print(columns)
    print(missing_data[columns].value_counts())

"""
Based on the summary above, each column has 205 rows of data, seven columns containing missing data:

"normalized-losses": 41 missing data
"num-of-doors": 2 missing data
"bore": 4 missing data
"stroke" : 4 missing data
"horsepower": 2 missing data
"peak-rpm": 2 missing data
"price": 4 missing data

"""


##2---> DEAL WITH THE MISSING DATA
"""
there are two methods
1. DROP DATA
a. drop the whole row
b. drop the whole column

2. REPLACE DATA
a. replace it by mean
b. replace it by frequency
c. replace it based on other functions
"""

# Whole columns should be dropped only if most entries in the column are empty. 
# In our dataset, none of the columns are empty enough to drop entirely
"""
Replace by mean:
"normalized-losses": 41 missing data, replace them with mean
"stroke": 4 missing data, replace them with mean
"bore": 4 missing data, replace them with mean
"horsepower": 2 missing data, replace them with mean
"peak-rpm": 2 missing data, replace them with mean

Replace by frequency:
"num-of-doors": 2 missing data, replace them with "four".
Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur

Drop the whole row:
"price": 4 missing data, simply delete the whole row
Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; 
therefore any row now without price data is not useful to us
"""

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)   #calculating the average value
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace=True)     #replacing the null value with average value

avg_stroke = df['stroke'].astype('float').mean(axis = 0)
print("Average of stroke", avg_stroke)
df['stroke'].replace(np.nan, avg_stroke, inplace = True)

avg_bore = df['bore'].astype('float').mean(axis = 0)
print("Average of bore", avg_bore)
df['bore'].replace(np.nan, avg_bore, inplace = True)

avg_horsepower = df['horsepower'].astype('float').mean(axis = 0)
print("Average of horsepower", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace = True)

avg_peak_rpm = df['peak-rpm'].astype('float').mean(axis = 0)
print("Average of peak-rpm",avg_peak_rpm)
df['peak-rpm'].replace(np.nan, avg_peak_rpm, inplace = True)

df['num-of-doors'].value_counts()
print(df['num-of-doors'].value_counts())
frequent_num_of_doors = df['num-of-doors'].value_counts().idxmax()              #finding most frequent value
df['num-of-doors'].replace(np.nan, frequent_num_of_doors, inplace = True)

print(df.head(10))
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)
print(df.head())



"""
The last step in data cleaning is checking and making sure that all data is in the correct format (int, float, text or other).

In Pandas, we use

.dtype() to check the data type

.astype() to change the data type
"""

print(df.dtypes)


#change the data types 

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[['price']] = df[['price']].astype('float')
df[['normalized-losses']] = df[['normalized-losses']].astype('int')
df[['peak-rpm']] = df[['peak-rpm']].astype('float')


print(df.dtypes)



"""
What is Standardization?
-->Standardization is the process of transforming data into a common format which allows the researcher to make the meaningful comparison.
"""
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()




"""
Normalization is the process of transforming values of several variables into a similar range. 
Typical normalizations include scaling the variable so the variable average is 0, 
scaling the variable so the variance is 1, or scaling variable so the variable values range from 0 to 1
"""
#To demonstrate normalization, let's say we want to scale the columns "length", "width" and "height"
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
#Here we can see, we've normalized "length", "width" and "height" in the range of [0,1].



"""
Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
In our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 57 unique values. 
What if we only care about the price difference between cars with high horsepower, 
medium horsepower, and little horsepower (3 types)? Can we rearrange them into three ‘bins' to simplify analysis?
We will use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins
"""
df['horsepower'] = df['horsepower'].astype(int, copy = True) #convert the horsepower to right format

import matplotlib as plt #checking the horsepower on plot
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
print("done")

"""
We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.
start_value=min(df["horsepower"])
end_value = max(df["horsepower"])

"""
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)
df["horsepower-binned"].value_counts()
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


"""
DUMMY VARIABLES
What is an indicator variable?
An indicator variable (or dummy variable) is a numerical variable used to label categories. 
They are called 'dummies' because the numbers themselves don't have inherent meaning.
Why we use indicator variables?
So we can use categorical variables for regression analysis in the later modules.
Example
We see the column "fuel-type" has two unique values, "gas" or "diesel"
Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.
"""
print(df.columns)

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True) #change column names for clarity
dummy_variable_1.head()
# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()

#merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

df.to_csv('clean_df.csv') #clean csv