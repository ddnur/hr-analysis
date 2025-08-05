# importing libs
import numpy as np
from sklearn.metrics import make_scorer
from tqdm import tqdm
import optuna.logging as opt
import warnings
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.model_selection import GridSearchCV
import joblib

# importing pipelines from other file
from regression_pipeline import (X_train,
                                 y_train,
                                 X_test,
                                 y_test,
                                 ridge_pipe,
                                 param_select_ridge,
                                 param_select_tree,
                                 tree_pipe,
                                 param_tree,
                                 feature_selection_pipe
                                 )

# logging settings
warnings.filterwarnings('ignore')
opt.set_verbosity(opt.WARNING)


def smape(y_true, y_pred):
    """"SMAPE"""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    if np.any(denominator == 0):
        return 'Ноль в знаменателе'
    diff = numerator / denominator
    return np.mean(diff) * 100


smape_scoring = make_scorer(smape, greater_is_better=False)

# necessary variables
state = 42
n_trials = 50

grid_search_ridge = GridSearchCV(ridge_pipe,
                                 param_select_ridge,
                                 cv=5,
                                 n_jobs=-1,
                                 scoring=smape_scoring
                                 )

grid_search_ridge.fit(X_train, y_train)

best_ridge_regressor = grid_search_ridge.best_estimator_

print(f'Best Ridge SMAPE on Train Sample: {-grid_search_ridge.best_score_}')

joblib.dump(best_ridge_regressor,
            'ridge_best_model_rabota_s_zabotoj.pkl'
            )

# progressbar for tree param selection
bar_tree = tqdm(total=n_trials, desc='Tree trials progress')


def tqdm_callback_tree(study, trial):
    bar_tree.update(1)


# tree param selection
optuna_search_tree = OptunaSearchCV(tree_pipe,
                                    param_tree,
                                    random_state=state,
                                    n_trials=n_trials,
                                    n_jobs=-1,
                                    scoring=smape_scoring,
                                    callbacks=[tqdm_callback_tree]
                                    )

optuna_search_tree.fit(X_train, y_train)

grid_tree_pipe = feature_selection_pipe(optuna_search_tree.best_estimator_)
grid_search_tree = GridSearchCV(grid_tree_pipe,
                                param_select_tree,
                                cv=5,
                                n_jobs=-1,
                                scoring=smape_scoring
                                )

grid_search_tree.fit(X_train, y_train)

best_tree_regressor = grid_search_tree.best_estimator_

print(f'\nBest Tree SMAPE on Train Sample: {-grid_search_tree.best_score_}')

joblib.dump(best_tree_regressor,
            'tree_best_model_rabota_s_zabotoj.pkl'
            )

best_model_list = [best_ridge_regressor, best_tree_regressor]
best_model_scores = []

for i in best_model_list:
    y_pred_select = i.predict(X_test)
    score = smape(y_test, y_pred_select)
    best_model_scores.append(score)

best_score = 100
idx = -1
for i in best_model_scores:
    if i < best_score:
        best_score = i
        idx += 1

print('\nBest SMAPE & Model on Test Sample:\nSMAPE:', best_score)
print('\nModel:', best_model_list[idx])
