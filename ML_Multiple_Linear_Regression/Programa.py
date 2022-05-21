import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def estandarizacion(data, column_1, column_2):
    x = data[column_1]
    y = data[column_2]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    x_standard = ((x - mean_x) / std_x).tolist()
    y_standard = ((y - mean_y) / std_y).tolist()
    return x_standard, y_standard


data = pd.read_csv("Rental_Data.csv")

df = pd.DataFrame(data)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck',
        'has_washer_dryer', 'has_doorman', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[["rent"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=6)
mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

dictionary = {index: [float(y_test.to_numpy()[index]), float(y_predict[index])] for index in range(len(y_test.to_numpy()))}
data = pd.DataFrame(dictionary).transpose()

y_test_standard, y_predict_standard = estandarizacion(data, 0, 1)

print("Train score: ")
print(round(mlr.score(x_train, y_train), 5))
print("Test score: ")
print(round(mlr.score(x_test, y_test), 5))
print("Error cuadrático medio: ")
print(round(mean_squared_error(y_test_standard, y_predict_standard), 5))

plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")
plt.show()
