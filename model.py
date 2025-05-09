from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd

df = pd.read_csv('creditcard.csv')
df.drop('Time', axis=1, inplace=True)

hyper_parameters = {
    'depth': 8,
    'early_stopping_rounds': 20, 
    'iterations': 150, 
    'l2_leaf_reg': 3, 
    'learning_rate': 0.2
}  # см. models_grid_search.ipynb

model = CatBoostClassifier(**hyper_parameters)

model.fit()

