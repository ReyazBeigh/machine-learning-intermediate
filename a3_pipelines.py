import pandas as pd

csv_path = "./input/melb_data.csv"

pd_data = pd.read_csv(csv_path)

y= pd_data.Price

X = pd_data.drop(["Price"], axis=1)

from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in train_X.columns if train_X[cname].nunique() < 10 and train_X[cname].dtype == "object"]

# Select numerical columns

numerical_cols = [cname for cname in train_X.columns if train_X[cname].dtype in ['int64', 'float64']]

# Keep selected columns only

target_columns = low_cardinality_cols + numerical_cols

#we are remvoing the columns with high cardinality
X_train_selected = train_X[target_columns].copy()
X_test_selected = test_X[target_columns].copy()

#bundling the preprocessing and modelling steps 
#1) impute the missing values
#2) applying one-hot encoding of the categorical columns

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#prepocesser for numerical data
num_prepocessor = SimpleImputer(strategy="constant")

#preprocesser for categorical data
cat_prepocessor = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant')),('onehot',OneHotEncoder(handle_unknown='ignore'))])


#bundle the preprocessor for numerical and categorical data

preprocessor = ColumnTransformer(transformers=[('num',num_prepocessor,numerical_cols),('cat',cat_prepocessor,low_cardinality_cols)])

#defining the model 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(n_estimators=100,random_state=0)

#bundle the preprocessor and model 
print("Bundling preprocessors and model ")
pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

#preprocessing the training data and fitting the mdel
print("Fitting ")
pipeline.fit(X_train_selected,train_y)

#preprocessing the test data and predicting the test data

print("Prediction of the pipelines way")
prediction = pipeline.predict(X_test_selected)

mae =   mean_absolute_error(test_y,prediction)
print("MAE IS ",mae)