import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)


#SIMPLE LINEAR REGRESSION 
X = df[['engine-size']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
yhat = lm.predict(X)
print(yhat)
lm.coef_
lm.intercept_
# r squared for simple regression problem
r2 = lm.score(X, Y)
print('the r2 of simple linear regression model = ', r2)
#meansquareerror
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(yhat,Y)
print('mean squared error = ', mse)

#now predictinf the outsource values
new_input = np.arange(1,100,1).reshape(-1, 1)
y_hat = lm.predict(new_input)
y_hat[0:5]
plt.plot(new_input,y_hat)
plt.show()



#MULTIPLE LINEAR REGRESSION 
Z = df[['horsepower','highway-mpg','curb-weight', 'engine-size']]
lm.fit(Z,df['price'])
lm.intercept_
lm.coef_
yhat = lm.predict(Z)
print(yhat)
#R squared for multiple linear regression
r2 = lm.score(z,Y)
print('r2 = ',r2)
#mean square error
mse = mean_squared_error(yhat,Y)
print('mean_squared_error = ', mse)

#MODEL VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt

width = 12
height = 10
plt.figure(figsize = (height, width))
sns.regplot(x= 'highway-mpg', y = 'price', data = df)
plt.ylim(0,)
plt.show()

#lets plot to peak-rpm and price and compare with above example
plt.figure(figsize=(height,width))
sns.regplot(x='peak-rpm', y= 'price', data = df)
plt.ylim(0,)
plt.show()

# VISUALIZATION OF MULTIPLE LINEAR REGRESSION MODELS
height = 10
width = 12
plt.figure(figsize=(width,height))
ax1 = sns.distplot(df['price'], hist=False, color='r', label='actual values')
sns.distplot(yhat, hist=False, color='b', label='predicted values', ax = ax1)
plt.show()


#POLYNOMIAL REGRESSION AND PIPELINE
x = df['highway-mpg']
y= df['price']

f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p)

#rsquared
r2 = lm.score(p(x),y)
print('r2 for polynomial regression', r2)
#mse
mse = mean_squared_error(df['price'], p(x))
print('mean_squared_error for simple polynomial regression = ', mse)
#for ploting the polynomial regression
def plotpolly(model, independent_variable, dependent_variable, name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(name)
    plt.ylabel('Price of Cars')
    plt.show()

plotpolly(p, x, y, 'highway-mpg')

#NOW POLYNOMIAL REGRESSION IN MULTIVARIABLES
#eg.- 𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋1𝑋2+𝑏4𝑋21+𝑏5𝑋22   having x and y two variables
#for this we import the library
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)
z_pr = pr.fit_transform(Z)
Z.shape
z_pr.shape


#USING PIPELINE 
#We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures()), ('model', LinearRegression())]
pipe = Pipeline(input)
pipe
pipe.fit(Z,y)
y_pipe = pipe.predict(Z)
y_pipe[0:4]













