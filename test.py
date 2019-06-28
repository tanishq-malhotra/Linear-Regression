from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('USA_Housing.csv')
X = data[['Avg. Area Income']]
Y = data[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)

lg = LinearRegression()
lg.fit(X_train, y_train)

pred = lg.predict(X_test)

lg.score(pred,y_test)