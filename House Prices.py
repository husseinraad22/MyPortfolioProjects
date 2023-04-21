import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Select features
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# Drop rows with missing target value
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Select target
y = train_data.SalePrice

# Select predictors
X = train_data[features]

# Split data into training and validation subsets
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=1)

# Fit the model
model.fit(train_X, train_y)

# Get predictions on validation data
val_predictions = model.predict(val_X)

# Evaluate the model
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE: {:,.0f}".format(val_mae))

# Get test predictions
test_features = test_data[features].fillna(0) # replace missing values with 0
test_predictions = model.predict(test_features)

# Save test predictions to file
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
output.to_csv('submission.csv', index=False)
