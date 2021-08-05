# Machine Learning Intermediate Course

- Demonstrate how we can handle better the missing values from the data. There might be multiple causes of missing values in the data source. 3 approaches to work around the missing values are: 
`1 - Dropping the column that has missing value(s)`
`2 - Imputing the missing values`
`3 - Imputing the missing values and keeping track of the missing values`,
`Choice 2 has been proved to be best`

- Concept of Categorical variables, Those are the fields/variables that have explicit enum values. example Color variables having values like: white,black and green, Another example, Having Breakfast can have values like, no, rarely, often, everyday. The categorical variables can have a large impact on the prediction and MAE. 3 approaches to handle those values are:
`1) - Dropping the column/field at all`
`2) - Giving them numerical values that can be sorted (Ordinal Encoding) but this can't be done every time, in some cases it won't be possbile`
`3) Creating a new column for every value and marking the presence and absence in each record Know as One-hot encoding`

Point worth noting: World is filled with Categorical Variables. If you know how to handle this common type then you will be a better data scientist!

- Introduction to Pipelines, the concept of popelines helps in maintaining the code clean, make the models easily deployable, keep the datapreprocesisng and modelling code organised

- Cross Validation Concept. In this method to validate the model we divide the given data set into multiple folds and then use on fold each time to validate the model

- the concept of new modeling method XGBoost - (Extream Gradient Boost), the underlying concept it with (1)Naive model(initital basic model) -> (2)Make Prediction -> (3)Calculate Loss -> (4)train new model-> (5) add new model to ensemble and then repeat from 2 till we have best score [internally mean squared error is used to calcualte loss]