# importing libs
import optuna.distributions as distributions

# models
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# feature selection
from sklearn.feature_selection import (SelectKBest,
                                       f_classif
                                       )

# preprocessing
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler
                                   )
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# importing datasets
from data_preprocessing import train_satisfaction
from data_preprocessing import test_features
from data_preprocessing import target_satisfaction

test_df = test_features.merge(target_satisfaction, on='id', how='left')
test_df['is_high_group'] = (test_df['supervisor_evaluation'] >= 4).astype(int)
train_satisfaction['is_high_group'] = (train_satisfaction['supervisor_evaluation'] >= 4).astype(int)

# creating lists of features
ohe_columns = ['dept', 'last_year_promo', 'last_year_violations']
rank_columns = ['level', 'workload']
cat_columns = ohe_columns + rank_columns
num_columns = ['employment_years', 'salary', 'is_high_group']
category_orders = [
    ['junior', 'middle', 'senior'],
    ['low', 'medium', 'high']
]

# train-test data split
X_train = train_satisfaction[num_columns + cat_columns]
y_train = train_satisfaction['job_satisfaction_rate']
X_test = test_df[num_columns + cat_columns]
y_test = test_df['job_satisfaction_rate']

# pipelines for encoders
ohe_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ]
)

ord_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ord', OrdinalEncoder(categories=category_orders,
                               handle_unknown='use_encoded_value',
                               unknown_value=-1
                               )
         )
    ]
)

# pipeline for scaler
num_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# pipeline for linear regression
data_preprocessor_ridge = ColumnTransformer(
    transformers=[
        ('ohe', ohe_pipe, ohe_columns),
        ('rank', ord_pipe, rank_columns),
        ('num', num_pipe, num_columns)
    ]
)

ridge_pipe = Pipeline(
    [
        ('preprocessor', data_preprocessor_ridge),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('model', Ridge())
    ]
)

# tree params & pipeline
param_tree = {
    'model__max_depth': distributions.IntDistribution(2, 20),
    'model__min_samples_split': distributions.IntDistribution(2, 20),
    'model__min_samples_leaf': distributions.IntDistribution(1, 10),
}

rank_pipe_tree = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value',
                               unknown_value=-1
                               )
         )
    ]
)

data_preprocessor_tree = ColumnTransformer(
    transformers=[
        ('cat', rank_pipe_tree, cat_columns),
        ('num', num_pipe, num_columns)
    ]
)

tree_pipe = Pipeline(
    [
        ('preprocessor', data_preprocessor_tree),
        ('model', DecisionTreeRegressor(random_state=42))
    ]
)


def feature_selection_pipe(best_estimator):
    pipe_grid_tree = Pipeline(
        [
            ('preprocessor', best_estimator.named_steps['preprocessor']),
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('model', best_estimator.named_steps['model'])
        ]
    )
    return pipe_grid_tree


# params for selectkbest
param_select_ridge = {
    'feature_selection__k': [*range(1, 12)]
}

param_select_tree = {
    'feature_selection__k': [*range(1, 9)]
}
