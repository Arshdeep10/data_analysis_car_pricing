import pandas as pd
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
data = pd.read_csv(path, header = None)
print(data.head())
#to know the data types of all the coloumns
print(data.dtypes)
print(data.info)





# Read Data
# We use pandas.read_csv() function to read the csv file. 
# In the bracket, we put the file path along with a quotation mark, so that pandas will read the file into a data frame from that address. 
# The file path can be either an URL or your local file address.
# Because the data does not include headers, 
# we can add an argument headers = None inside the read_csv() method, so that pandas will not automatically set the first row as a header.
# You can also assign the dataset to any variable you create.
import pandas as pd
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
data = pd.read_csv(path, header = None)

# Add Headers
# Take a look at our dataset; pandas automatically set the header by an integer from 0.
# To better describe our data we can introduce a header, this information is available at: https://archive.ics.uci.edu/ml/datasets/Automobile
# Thus, we have to add headers manually.
# Firstly, we create a list "headers" that include all column names in order. 
# Then, we use dataframe.columns = headers to replace the headers by the list we created.
# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)
data.columns = headers
data.head(10)

# Find the name of the columns of the dataframe
print(data.columns)

# The main types stored in Pandas dataframes are object, float, int, bool and datetime64. In order to
print(data.dtypes)  #returns a Series with the data type of each column.

#If we would like to get a statistical summary of each column, 
#such as count, column mean value, column standard deviation, etc. We use the describe method:
data.describe()           #This method will provide various summary statistics, excluding NaN (Not a Number) values.
# However, what if we would also like to check all the columns including those that are of type object.
print(data.describe(include = "all"))           #Now, it provides the statistical summary of all the columns, including object-typed attributes.


data.info()       #It provide a concise summary of your DataFrame.
