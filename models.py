import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder
from xgboost import XGBClassifier

# Get iris data as a Pandas df like you would have in production
iris = load_iris(as_frame=True)
data = iris.frame
target_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
data["target"] = data["target"].map(target_mapping)
le = LabelEncoder()
X = data.drop(columns="target")
y = data["target"]
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# Define different Pipelines for Numeric vs. Categorical
numeric_transformer = Pipeline(
    steps=[
        ("numeric_imputer", SimpleImputer(strategy="mean")), 
        ("scaler", MaxAbsScaler())
        ]
)

categorical_transformer = Pipeline(
    steps=[
        ("categorical_imputer", SimpleImputer(strategy="most_frequent")), 
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]
)

numeric_features = X_train.select_dtypes(include="number").columns.values
categorical_features = X_train.select_dtypes(exclude="number").columns.values

# Combine into single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numeric", numeric_transformer, numeric_features)
    ]
)


# Define a single Pipeline with all of the steps
### Try different models and select the one with the best cross_val_score
models = [LogisticRegression(), XGBClassifier()]
accuracys = []
pipelines = []
for model in models:
    steps = [
        ('preprocessor', preprocessor),
        ('classifier', model)
    ]
    pipeline = Pipeline(steps)
    pipelines.append(pipeline)
    accuracy = pipeline.fit(X_train, y_train).score(X_test, y_test)
    print(f'''{str(model).split("(")[0]} Test Accuracy\n{accuracy:,.3f}''')
    accuracys.append(accuracy)

# Get the best model
best_model_index = accuracys.index(max(accuracys))
best_model = models[best_model_index]
steps = [
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ]
pipeline = Pipeline(steps)

# Pickle the model
with open("app/model/trained_pipeline-0.1.0.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Pass new observation where columns are:
## sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
new_observation = pd.DataFrame(np.array([[5.1, 3.0, 4.2, 1.7]]), columns=["sepal length (cm)",  "sepal width (cm)",  "petal length (cm)",  "petal width (cm)"])
prediction = pipeline.predict(new_observation)

# Print out result
api_return_value = le.inverse_transform(prediction)
print(api_return_value)