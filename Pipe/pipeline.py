# %%
# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Global Definitions
numeric_features = ['area', 'parking', 'furnishingstatus', 'bedrooms', 'stories']
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')

models = {
    'random_forest': RandomForestRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'linear_regression': LinearRegression(),
    'decision_tree': DecisionTreeRegressor(),
    'svr': SVR()
}


# %%
def load_and_split_data():
    df = pd.read_pickle("G:\My Drive\Colab Notebooks\Assignment3\SML_G2\pipeline\dataframe.pkl")
    X = df[numeric_features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(model_name, X_train, y_train):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', models[model_name])
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    model_name = 'random_forest'  # You can change this as needed
    X_train, X_test, y_train, y_test = load_and_split_data()
    pipeline = train_model(model_name, X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(f'Model: {model_name}')
    print(f'MSE: {mean_squared_error(y_test, predictions)}')
    print(f'R2 Score: {r2_score(y_test, predictions)}')