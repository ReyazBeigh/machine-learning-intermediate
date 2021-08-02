import pandas as pd

csv_path = "./input/melb_data.csv"

pd_data = pd.read_csv(csv_path)

y= pd_data.Price

X = pd_data.drop(["Price"], axis=1)

from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

#Drop columns with missing values
missing_value_columns = [col for col in train_X.columns if train_X[col].isnull().any()]

train_X = train_X.drop(missing_value_columns, axis=1)
test_X = test_X.drop(missing_value_columns, axis=1)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in train_X.columns if train_X[cname].nunique() < 10 and train_X[cname].dtype == "object"]

# Select numerical columns

numerical_cols = [cname for cname in train_X.columns if train_X[cname].dtype in ['int64', 'float64']]

# Keep selected columns only

target_columns = low_cardinality_cols + numerical_cols

X_train_selected = train_X[target_columns].copy()
X_test_selected = test_X[target_columns].copy()


#checkout the object columns 'those having string type value'

obj_cols = (X_train_selected.dtypes == "object")

obj_cols = list(obj_cols[obj_cols].index)

print(obj_cols)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#function for comparing different approaches

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


#Score from dropping the categorical columns

X_train_selected_drop = X_train_selected.drop(obj_cols, axis=1)
X_test_selected_drop = X_test_selected.drop(obj_cols, axis=1)

print("Score from Dropping Cat. Col. approach")
drop_col_score = score_dataset(X_train_selected_drop, X_test_selected_drop, train_y, test_y)


print(drop_col_score)

#Score from Approach 2 (Ordinal Encoding)

from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
X_train_selected_encoded = X_train_selected.copy()
X_test_selected_encoded = X_test_selected.copy()


#applying ordianl encoder to the cat. cols. --> Object cols particularly

ordinal_encoder = OrdinalEncoder()

X_train_selected_encoded[obj_cols] = ordinal_encoder.fit_transform(X_train_selected[obj_cols])
X_test_selected_encoded[obj_cols] = ordinal_encoder.transform(X_test_selected_encoded[obj_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 

print(score_dataset(X_train_selected_encoded, X_test_selected_encoded, train_y, test_y))


#Score from Approach 3 (One-Hot Encoding)

from sklearn.preprocessing import OneHotEncoder

one_hot_encoer = OneHotEncoder(handle_unknown='ignore',sparse=False)

X_test_selected_oh_encoder = pd.DataFrame(one_hot_encoer.fit_transform(X_test_selected[obj_cols]))
X_train_selected_oh_encoder = pd.DataFrame(one_hot_encoer.transform(X_train_selected[obj_cols]))

# One-hot encoding removed index; put it back


X_test_selected_oh_encoder.index = X_test_selected.index
X_train_selected_oh_encoder.index = X_train_selected.index

#remove the cat. object cols from the data
X_test_oh_num_data = X_test_selected.drop(obj_cols, axis=1)
X_train_oh_num_data = X_train_selected.drop(obj_cols, axis=1)

#add back the cat. cols. after on hot encoding
X_test_oh_all_data = pd.concat([X_test_oh_num_data, X_test_selected_oh_encoder], axis=1)
X_train_oh_all_data = pd.concat([X_train_oh_num_data, X_train_selected_oh_encoder], axis=1)


print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(X_train_oh_all_data, X_test_oh_all_data, train_y, test_y))

