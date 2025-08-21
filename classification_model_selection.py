# importing libs
from tqdm import tqdm
from optuna_integration.sklearn import OptunaSearchCV
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import optuna.logging as opt
import warnings

# importing pipelines
from classification_pipeline import (svc_pipe,
                                     gbc_pipe,
                                     param_svc,
                                     param_gbc,
                                     X_train_quit,
                                     X_test_quit,
                                     y_train_quit,
                                     y_test_quit,
                                     catboost_pipe
                                     )

# logging settings
warnings.filterwarnings('ignore')
opt.set_verbosity(opt.WARNING)

# necessary constants
n_jobs = -1
state = 42

# progressbars
bar_svc = tqdm(total=10, desc='SVC trials progress')
bar_gbc = tqdm(total=10, desc='GBC trials progress')


def tqdm_callback_svc(study, trial):
    """SVC optimising progressbar"""
    bar_svc.update(1)


def tqdm_callback_gbc(study, trial):
    """GradientBoosting optimising progressbar"""
    bar_gbc.update(1)


# models optimising
optuna_search_svc = OptunaSearchCV(svc_pipe,
                                   param_svc,
                                   cv=5,
                                   random_state=state,
                                   n_jobs=n_jobs,
                                   scoring='roc_auc',
                                   callbacks=[tqdm_callback_svc]
                                   )

optuna_search_gbc = OptunaSearchCV(gbc_pipe,
                                   param_gbc,
                                   cv=5,
                                   random_state=state,
                                   n_jobs=n_jobs,
                                   scoring='roc_auc',
                                   callbacks=[tqdm_callback_gbc]
                                   )

# params selection
optuna_search_svc.fit(X_train_quit, y_train_quit)
best_svc_classifier = optuna_search_svc.best_estimator_
print(f'\nBest SVC ROC-AUC on train sample: {optuna_search_svc.best_score_}')

joblib.dump(best_svc_classifier,
            'svc_best_model_rabota_s_zabotoj.pkl'
            )

optuna_search_gbc.fit(X_train_quit, y_train_quit)
best_gbc_classifier = optuna_search_gbc.best_estimator_
print(f'\nBest GBC ROC-AUC on train sample: {optuna_search_gbc.best_score_}')

joblib.dump(best_gbc_classifier,
            'gbc_best_model_rabota_s_zabotoj.pkl'
            )

catboost_pipe.fit(X_train_quit, y_train_quit)
y_proba_catboost_train = catboost_pipe.predict_proba(X_train_quit)[:, 1]
roc_auc_score(y_train_quit, y_proba_catboost_train)

joblib.dump(catboost_pipe,
            'catboost_pipe_rabota_s_zabotoj.pkl'
            )

# testing models
print("\nSVC cross-val ROC-AUC", cross_val_score(best_svc_classifier, X_train_quit, y_train_quit, cv=5))
y_proba_svc = best_svc_classifier.predict_proba(X_test_quit)[:, 1]
print("Best SVC ROC-AUC on train sample", roc_auc_score(y_test_quit, y_proba_svc))

print("\nGBC cross-val ROC-AUC", cross_val_score(best_gbc_classifier, X_train_quit, y_train_quit, cv=5))
y_proba_gbc = best_gbc_classifier.predict_proba(X_test_quit)[:, 1]
print("Best GBC ROC-AUC on test sample", roc_auc_score(y_test_quit, y_proba_gbc))

print("\ncatboost cross-val ROC-AUC", cross_val_score(catboost_pipe, X_train_quit, y_train_quit, cv=5))
y_proba_catboost = catboost_pipe.predict_proba(X_test_quit)[:, 1]
print("Best catboost ROC-AUC on train sample", (roc_auc_score(y_test_quit, y_proba_catboost)))
