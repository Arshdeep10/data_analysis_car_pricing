
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'

df = pd.read_csv(path)
print(df.head)
print(df.dtypes)
print(df.corr())

# Engine size as potential predictor variable of price
snb.regplot(x = 'engine-size',y = 'price', data = df)
plt.ylim(0,)
plt.show()
#here the correlation between engine-size and price is 0.872 near to 1 means positive correlation as
#the engine size increases the price will also increase
print(df[['engine-size', 'price']].corr())


# noe see the correlation between the highway- mpg and price
snb.regplot(x='highway-mpg', y= 'price', data= df)
plt.show()
print(df[['highway-mpg', 'price',]].corr())# the correlation is similar to -0.7... (can be used as good predictor)so have negitive correlation

#now the correlation between the peak-rpm and the price
snb.regplot(x="peak-rpm", y="price", data=df)
plt.show()
print(df[['peak-rpm', 'price']].corr())# thecorrelation is -0.1 so not a good predicter


"""
These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories.
The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots.
"""
#check the relationship between "body-style" and "price".
snb.boxplot(x = 'body-style', y = 'price', data = df)
plt.show()
print("We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price. Let\\'s examine engine \"engine-location\" and \"price\\\":")

#check the relationship between engine location and price
snb.boxplot(x= 'engine-location', y='price', data = df)
plt.show()
print("here the distribution of engine location is dinticnt enough to take it as a good predictor")

#check the distribution between the drive-wheels and price
snb.boxplot(x = 'drive-wheels', y='price',data = df)
plt.show()
print("not too distinct distribution but can be a predictor of the price")


print(df.describe(include = ['object']))
engine_location_counts = df['engine-location'].value_counts().to_frame()
engine_location_counts.rename(columns = {'engine-location' : 'value_counts'}, inplace = True)
engine_location_counts.index.name = 'engine-location'
engine_location_counts.head()





print('BASICS OF GROUPING')
#if we groupby the drive-wheels we have found there are three fwr,rear,4wd
groupbyone = df[['drive-wheels','engine-location', 'price']]
groupbyone = groupbyone.groupby(['drive-wheels','engine-location'],as_index = False).mean()
groupbyone
#to visualise in easy way we will use pivot table that can be formed by the keyword pivot
groupbyone = groupbyone.pivot(index='drive-wheels', columns = 'engine-location')
groupbyone
#fill the missing values with zero
groupbyone = groupbyone.fillna(0)
#on graph
plt.pcolor(groupbyone,cmap = 'RdBu')
plt.colorbar()
plt.show()

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1
grouped_test1_pivot = grouped_test1.pivot(index = 'drive-wheels', columns = 'body-style')
grouped_test1_pivot



print("CORRELATION AND CASUATION, COFFICIENT OF CORRELATIO, P-VALUE")
from scipy import stats
pearsonr_coff, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("the pearsonr cofficient of the wheel-base", pearsonr_coff, "ans the p-value",p_value)
print("as by the pvalue the correlation is very significant as p_value < 0.001 but the linear relationship is not too strong as pearsonr_cofficient = 0.5...")
#correlation between horse power and price
pearsonr_coff, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("pearsonr_cofficient = ", pearsonr_coff, "and p-value = ", p_value)
print("the linear relationship betwwn horsepower and price is strong and the pvalue(<0.001) shows correlation is too strong")

#correlation between lenght of car and price
pearsonr_coff, p_value = stats.pearsonr(df['length'],df['price'])
print("pearsonr_cofficient", pearsonr_coff,"pvalue",p_value)
print("the cofficient of pearsonr correlation is 0.690~7 so +ve correlatoin and the pvalue gives that the correaltion is significant")

#width and price
pearsonr_coff, p_value = stats.pearsonr(df['width'], df['price'])
print("cofficient personr = ", pearsonr_coff, "pvalue", p_value)
print("the linear relationship is moderatly strong and the correlation is very significant")


#engine size and price
pearsonr_coff, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("pearsonr_cofficient = ", pearsonr_coff, 'pvalue = ',p_value)
print("the linear relation between the variables is strong and the correlation is significant too")

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   

