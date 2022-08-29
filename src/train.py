import joblib
import pandas as pd

from model_selection import select_model, tune_model

train = pd.read_csv('data/train.csv')

# categorize all features
y_label = 'failure'
predictor_labels = train.columns[train.columns != y_label]
to_be_dropped = ['id', 'product_code']
categorical = ['attribute_0', 'attribute_1']
numeric = [pred for pred in predictor_labels if (pred not in categorical and pred not in to_be_dropped)]

# let's only use top 5 features
# according to logreg importance
# source: https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense
best_predictors = ['loading', 'attribute_3', 'measurement_2', 'measurement_4', 'measurement_17']
categorical = []
numeric = best_predictors

X, y = train.loc[:, predictor_labels], train[y_label]

# select and tune a model using cross-validation and grid search
best_model = select_model(X, y, categorical, numeric)
best_model_tuned = tune_model(X, y, best_model)
best_model_tuned.fit(X, y)

joblib.dump(best_model_tuned, 'trained_models/best_model.pkl')