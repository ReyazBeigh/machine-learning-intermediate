from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import model_selection

csv_path = "./input/melb_data.csv"

data = pd.read_csv(csv_path)

features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

# predictors
X = data[features]

# Target
y = data.Price

print("Splitting the data")
train_X, test_X, train_y, test_y = train_test_split(X, y)


model = XGBRegressor()

print("fitting the model")
model.fit(train_X, train_y)

# print(model)
print("Prediction")
prediction = model.predict(test_X)

print("MEA is ", mean_absolute_error(prediction, test_y))


print("After tunning the parameters")

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
print("Fitting")
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)
print("Prediction")
prediction = model.predict(test_X)
print("MEA is ", mean_absolute_error(prediction, test_y))