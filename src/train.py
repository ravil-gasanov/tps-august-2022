import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


train = pd.read_csv('data/train.csv')
y_label = 'failure'

all_predictors = train.columns[train.columns != y_label]
categorical = ['attribute_0', 'attribute_1']
numeric = [pred for pred in all_predictors if (pred not in categorical and pred not in ['id', 'product_code'])]

# best according to logreg importance (source: https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense)
best_predictors = ['loading', 'attribute_3', 'measurement_2', 'measurement_4', 'measurement_17']
# source: eda
indicative_nan_cols = ['measurement_3', 'measurement_5']

X, y = train.loc[:, all_predictors], train[y_label]

process_categorical = Pipeline(
    [
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False))
    ]
)

process_numeric = Pipeline(
    [
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler())
    ]
)
prep = ColumnTransformer(
    [
        #('missing_indicator', MissingIndicator(), indicative_nan_cols),
        #('process_categorical', process_categorical, categorical),
        ('process_numeric', process_numeric, best_predictors)
    ]
)

model = ("logreg", LogisticRegression())

hyperparams = {
    'logreg__penalty': ['l1'],
    'logreg__solver': ['liblinear'],
    'logreg__C': np.linspace(0.01, 0.1, 10)
}

pipe = Pipeline(steps = [('prep', prep), model])

gkf = GroupKFold(n_splits=5)
gridcv = GridSearchCV(estimator = pipe, param_grid = hyperparams, cv = gkf, scoring = 'roc_auc')
gridcv.fit(X, y, groups = X['product_code'])

print(gridcv.best_score_) # 0.5912

best_model = gridcv.best_estimator_
best_model.fit(X, y)

joblib.dump(best_model, 'trained_models/best_model.pkl')