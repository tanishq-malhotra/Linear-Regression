from regression import LinearRegression
import pandas as pd

data = pd.read_csv('USA_Housing.csv')
X = data['Avg. Area Income']
Y = data['Price']

lg = LinearRegression()
lg.fit(X,Y)
