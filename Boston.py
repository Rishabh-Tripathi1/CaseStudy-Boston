import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

data = pd.read_csv('BostonHousing.csv')

linr = LinearRegression()

data['medv'] = np.log1p(data['medv'])

x = data.drop(['medv','b'], axis=1)
y = data['medv']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
linr.fit(x_train, y_train)
y_pred = linr.predict(x_test)
print(mean_squared_error(y_test, y_pred))


# Mean Squared Error:  0.031129333980953557
