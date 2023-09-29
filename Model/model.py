import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pickle


# Load dataset
df = pd.read_pickle("G:\My Drive\Colab Notebooks\Assignment3\SML_G2\EDA\dataframe.pkl")

# Selecting features and target
features = df[['area', 'parking', 'furnishingstatus', 'bedrooms', 'stories']]
target = df['price']

def preprocess_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['area', 'parking', 'bedrooms', 'stories']),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), ['furnishingstatus'])
        ], 
        remainder='passthrough'
    )
    return X_train, X_test, y_train, y_test, preprocessor


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

def train_and_tune_models(X_train, y_train, preprocessor):
    model_params = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {'model__bootstrap': [True, False], 'model__max_depth': [10, 20, None], 'model__min_samples_split': [2, 5, 10], 'model__n_estimators': [25, 50]}
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {'model__max_depth': [5, 10, 15], 'model__min_samples_split': [2, 5, 10]}
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
        },
        'Support Vector Regressor': {
            'model': SVR(),
            'params': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
        }
    }
    
    trained_models = {}
    for name, model_info in model_params.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_info['model'])
        ])
        
        clf = GridSearchCV(pipeline, model_info['params'], cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False))
        clf.fit(X_train, y_train)
        
        trained_models[name] = clf.best_estimator_
        print(f"{name} has been trained and tuned with best parameters: {clf.best_params_}")
        
    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    for name, model in trained_models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f'{name} - MSE: {mse}, R2 Score: {r2}')


def predict_new_data(trained_models, new_data):
    predictions = {}
    for name, model in trained_models.items():
        prediction = model.predict(new_data)
        predictions[name] = prediction[0]
    return predictions


if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(features, target)
    
    # Saving preprocessed data
    with open('X_train.pkl', 'wb') as file:
        pickle.dump(X_train, file)
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file)
    with open('y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)
    with open('y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)
    
    # Save the preprocessor
    with open('preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    # Train, tune, and evaluate the models
    trained_models = train_and_tune_models(X_train, y_train, preprocessor)
    evaluate_models(trained_models, X_test, y_test)

    # Save models
    for name, model in trained_models.items():
        with open(f'{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Example new data
    new_data = pd.DataFrame({
        'area': [7420],
        'parking': [2],
        'furnishingstatus': ['furnished'],
        'bedrooms': [4],
        'stories': [3]
    })

    # Predict with the new data
    new_data_predictions = predict_new_data(trained_models, new_data)
    for model_name, prediction in new_data_predictions.items():
        print(f"{model_name} Prediction: {prediction}")