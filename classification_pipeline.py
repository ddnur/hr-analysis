# importing libs
import optuna.distributions as distributions

# models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import joblib

# preprocessing
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# importing datasets
from data_preprocessing import (train_quit,
                                test_features,
                                target_quit,
                                target_satisfaction
                                )
from regression_pipeline import (ohe_pipe,
                                 rank_pipe,
                                 num_pipe
                                 )

# loading models
tree_best_model = joblib.load('/Users/nurasyl/PycharmProjects/rabota_s_zabotoj/ridge_best_model_rabota_s_zabotoj.pkl')

# merging df
test_df = test_features.merge(target_satisfaction, on='id', how='left')
test_df_quit = test_df.merge(target_quit, on='id', how='right')

# adding new features
train_quit['is_high_group'] = (train_quit['supervisor_evaluation'] >= 4).astype(int)
train_quit = train_quit.drop(columns='supervisor_evaluation')
train_quit['job_satisfaction_rate'] = tree_best_model.predict(train_quit)
test_df_quit['is_high_group'] = (test_df['supervisor_evaluation'] >= 4).astype(int)

test_df_quit['job_satisfaction_rate'] = (tree_best_model
                                         .predict(test_df_quit
                                                  .drop(columns='job_satisfaction_rate'
                                                        )
                                                  )
                                         )

if __name__ == '__main__':
    print("Checking train_quit for NA:", train_quit.isna().sum())
    print("Checking test_df for NA", test_df.isna().sum())

# lists of columns
num_columns_quit = ['employment_years', 'salary', 'is_high_group', 'job_satisfaction_rate']
ohe_columns = ['dept', 'last_year_promo', 'last_year_violations']
rank_columns = ['level', 'workload']
cat_columns = ohe_columns + rank_columns

# splitting data
X_train_quit = train_quit[num_columns_quit + cat_columns]
y_train_quit = train_quit['quit']

X_test_quit = test_df_quit[num_columns_quit + cat_columns]
y_test_quit = test_df_quit['quit']

# hyperparameters distributions
param_svc = {
    'model__C': distributions.FloatDistribution(0.001, 1, log=True)
}

param_gbc = {
    'model__min_samples_split': distributions.IntDistribution(2, 20),
    'model__max_depth': distributions.IntDistribution(8, 20),
    'model__min_samples_leaf': distributions.IntDistribution(5, 10),
    'model__max_features': distributions.CategoricalDistribution(['sqrt', 'log2', None]),
    'model__ccp_alpha': distributions.FloatDistribution(0.001, 0.05, log=True)
}

# data preprocessors
data_preprocessor_svc = ColumnTransformer(
    transformers=[
        ('ohe', ohe_pipe, ohe_columns),
        ('rank', rank_pipe, rank_columns),
        ('num', num_pipe, num_columns_quit)
    ]
)

te_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('te', TargetEncoder())
    ]
)

data_preprocessor_gbc = ColumnTransformer(
    transformers=[
        ('te', te_pipe, ohe_columns),
        ('num', num_pipe, num_columns_quit)
    ]
)
data_preprocessor_catboost = ColumnTransformer(
    transformers=[
        ('cat', SimpleImputer(strategy='most_frequent'), cat_columns),
        ('num', SimpleImputer(strategy='median'), num_columns_quit)
    ]
)

# models pipelines
svc_pipe = Pipeline(
    [
        ('preprocessor', data_preprocessor_svc),
        ('model', SVC(probability=True))
    ]
)

gbc_pipe = Pipeline(
    [
        ('preprocessor', data_preprocessor_gbc),
        ('model', GradientBoostingClassifier(random_state=42))
    ]
)

catboost_pipe = Pipeline(
    [
        ('preprocessor', data_preprocessor_catboost),
        ('model', CatBoostClassifier(cat_features=[0, 1, 2, 3, 4],
                                     verbose=True
                                     )
         )
    ]
)
