import pandas as pd 
from sklearn.model_selection import train_test_split

csv_path = "input/melb_data.csv"

mlb_data = pd.read_csv(csv_path)

#prediction target
y= mlb_data.Price

#prediction features
p_f = mlb_data.drop(['Price'], axis=1)
X = p_f.select_dtypes(exclude=['object'])# here we are exclusing the string columns

#split data into training and testing sets

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train,X_test,y_train,y_test):
    model = RandomForestRegressor(n_estimators=10,random_state=0)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test,preds)

cols_with_missing_values = [col for col in train_X.columns if train_X[col].isnull().any()]


clean_X_train = train_X.drop(cols_with_missing_values, axis=1)
clearn_X_test = test_X.drop(cols_with_missing_values, axis=1)

print("MAE from the approach of dropping the cols with missing values ")
print(score_dataset(clean_X_train,clearn_X_test,train_y,test_y))

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
#impute the dataset
imputed_X_train = pd.DataFrame(imputer.fit_transform(train_X))
imputed_X_test = pd.DataFrame(imputer.transform(test_X))


#imputation remove column sets, we need to put them back

imputed_X_test.columns = test_X.columns
imputed_X_train.columns = train_X.columns

print("MAE for Approach 2 (Imputation)")
print(score_dataset(imputed_X_train,imputed_X_test,train_y,test_y))

#approach 3: Impute the missing values and keep track of the missing values, adding true to columns that have missing vlaues in next extended column and false otherwise 

train_X_plus = train_X.copy()
test_X_plus = test_X.copy()

for col in cols_with_missing_values:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    test_X_plus[col + '_was_missing'] = test_X_plus[col].isnull()


#impute the missing values
imputer = SimpleImputer()

imputed_train_X_plus = pd.DataFrame(imputer.fit_transform(train_X_plus))
imputed_test_X_plus = pd.DataFrame(imputer.transform(test_X_plus))

imputed_train_X_plus.columns = train_X_plus.columns
imputed_test_X_plus.columns = test_X_plus.columns

print("MAE using approach 3, Imputation plus tracking the missing cols")
print(score_dataset(imputed_train_X_plus,imputed_test_X_plus,train_y,test_y))


# Shape of the data used sofar 
print("Shape of the data used sofar")
print(train_X.shape) # rows, columns

print("Missing value shapes")
missing_value_count = train_X.isnull().sum()
rmissing = missing_value_count[missing_value_count>0]
print(rmissing.sum())