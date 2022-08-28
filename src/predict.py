
import joblib
import pandas as pd

y_label = 'failure'

test = pd.read_csv('data/test.csv')
best_model = joblib.load('trained_models/best_model.pkl')

predictions = pd.DataFrame(best_model.predict_proba(test)[:, 1], columns = [y_label])
submission = pd.concat([test['id'], predictions], axis = 1)

submission.to_csv('data/submission.csv', index = False)