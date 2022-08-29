import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from utils.preprocessing import make_preprocessor


MODELS = [
    ('logreg', LogisticRegression())
]

HYPERPARAMS = {
    'logreg':
            {
                'logreg__penalty': ['l1'],
                'logreg__solver': ['liblinear'],
                'logreg__C': np.logspace(-4, 4, 4)
            }
}


def select_model(X, y, categorical, numeric, cv = 5, scoring = 'roc_auc'):
    '''
    Returns the best model from a list of models (no tuning).
    Expects a utility function for scoring, i.e. greater is better.
    '''
    best_score = 0
    best_model = None
    preprocessor = ('preprocessor', make_preprocessor(categorical, numeric))

    gkf = GroupKFold(n_splits=cv)
    groups = X['product_code']

    for model in MODELS:
        full_model = Pipeline(steps = [preprocessor, model])
        scores = cross_val_score(full_model, X, y, cv=gkf, groups = groups, scoring=scoring)
        average_score = scores.mean()

        if average_score > best_score:
            best_score = average_score
            best_model = full_model
    
    print(best_score)
    return best_model



def tune_model(X, y, model, cv = 5, scoring = 'roc_auc'):
    '''
    Returns 'model' with tuned hyperparameters
    '''
    model_name = model.steps[-1][0]
    gkf = GroupKFold(n_splits=cv)
    groups = X['product_code']

    gridcv = GridSearchCV(estimator = model, param_grid = HYPERPARAMS[model_name],\
         cv = gkf, scoring = scoring)
    gridcv.fit(X, y, groups = groups)

    print(gridcv.best_score_)
    return gridcv.best_estimator_