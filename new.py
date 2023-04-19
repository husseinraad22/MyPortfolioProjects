# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Read the training and testing datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Check for missing values and handle them
print(train_data.isnull().sum())
train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
train_data["Cabin"].fillna("Unknown", inplace=True)

# Encode categorical variables
le = LabelEncoder()
train_data["HomePlanet"] = le.fit_transform(train_data["HomePlanet"])
train_data["Cabin"] = le.fit_transform(train_data["Cabin"])
train_data["Destination"] = le.fit_transform(train_data["Destination"])
train_data["Name"] = le.fit_transform(train_data["Name"])

# Scale numerical variables
sc = StandardScaler()
train_data[["Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] = sc.fit_transform(train_data[["Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]])

# Split the data into input and output variables
X = train_data[["PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name"]]
y = train_data["Transported"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the random forest classifier model
rf_model = RandomForestClassifier()

# Define hyperparameters for grid search
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(rf_model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Train the model with best hyperparameters
rf_model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                   max_depth=grid_search.best_params_['max_depth'],
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
rf_model.fit(X_train, y_train)

# Make predictions on test data and evaluate the model performance
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

test_data['Cabin'].fillna(test_data['Cabin'].mode()[0], inplace=True)
# Make predictions on test data
test_data["HomePlanet"] = le.fit_transform(test_data["HomePlanet"])
test_data["Cabin"] = le.fit_transform(test_data["Cabin"])
test_data["Destination"] = le.fit_transform(test_data["Destination"])
test_data["Name"] = le.fit_transform(test_data["Name"])
test_data[["Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] = sc.transform(test_data[["Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]])
#test_data.drop("Survived", axis=1, inplace=True)
test_predictions = rf_model.predict(test_data)

# Create submission CSV file
submission_df = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Transported": test_predictions})
submission_df.to_csv("submission.csv", index=False)
