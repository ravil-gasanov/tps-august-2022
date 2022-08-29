
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

def make_preprocessor(categorical, numeric):
    preprocessor = ColumnTransformer(
        [
            ('process_categorical', process_categorical, categorical),
            ('process_numeric', process_numeric, numeric)
        ]
    )

    return preprocessor