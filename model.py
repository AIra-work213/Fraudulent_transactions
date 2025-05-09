from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score


'''Подготовка необходимых данных'''

df = pd.read_csv('creditcard.csv')
df.drop('Time', axis=1, inplace=True)

x = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75)


'''Обучение модели'''

hyper_parameters = {
    'depth': 8,
    'early_stopping_rounds': 20, 
    'iterations': 150, 
    'l2_leaf_reg': 3, 
    'learning_rate': 0.2
}  # см. models_grid_search.ipynb

model = CatBoostClassifier(**hyper_parameters)
'''Обучение происходило в другом месте ввиду технических ограничений'''
model.load_model('catboost_model.cbm')

'''Тестирование модели и проверка качества'''

y_pred = model.predict(x_test)
metrics = {
    'F1_score': f1_score,
    'Recall': recall_score,
    'precision': precision_score,
    'ROC_AUC': roc_auc_score
}
for name in metrics:
    print(f'{name}: {round(metrics[name](y_test, y_pred), 5)}')

