from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

csv_path = "./input/melb_data.csv"

data = pd.read_csv(csv_path)

features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

# predictors
X = data[features]

# Target
y = data.Price


pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(
)), ('model', RandomForestRegressor(n_estimators=50, random_state=0))])


print("Scoring now with cross_val_score")

score = -1 * cross_val_score(pipeline, X, y, cv=5,
                             scoring='neg_mean_absolute_error')

print("MAE Is: ")
print(score)

print("Mean MAE of cross validation scores is: ", score.mean())
