# importing libs
import pandas as pd

# reading job satisfaction info
try:
    train_satisfaction = pd.read_csv(
        'datasets/work_satisfaction/train_job_satisfaction_rate_fixed.csv'
    )
except FileNotFoundError:
    train_satisfaction = pd.read_csv(
        '/datasets/train_job_satisfaction_rate.csv'
    )

try:
    test_features = pd.read_csv(
        'datasets/work_satisfaction/test_features.csv'
    )

except FileNotFoundError:
    test_features = pd.read_csv(
        '/datasets/test_features.csv'
    )
try:
    target_satisfaction = pd.read_csv(
        'datasets/work_satisfaction/test_target_job_satisfaction_rate.csv'
    )
except FileNotFoundError:
    target_satisfaction = pd.read_csv(
        '/datasets/test_target_job_satisfaction_rate.csv'
    )


def data_check(data: pd.DataFrame | pd.Series):
    """Checks dataframes info"""

    # displaying main info
    print('Main info:')
    data.info()

    # data description
    print('\nNumerical data description:\n')
    if not data.select_dtypes(exclude='object').empty:
        print(data.select_dtypes(exclude='object').describe())
    else:
        print('Numerical data is missing')
    print("String data description:")
    if not data.select_dtypes(include='object').empty:
        print(data.select_dtypes(include='object').describe())
    else:
        print('String data is missing')

    # displaying first 10 rows
    print('\nFirst 10 rows:')
    print(data.head(10))

    # checking for explicit duplicates
    print('\nExplicit duplicates ratio:')
    print(sum(data.duplicated()) / len(data))

    # checking for implicit duplicates
    if not data.select_dtypes(include='object').empty:
        print('\nUnique values of categorical features:\n')
        for col in data.select_dtypes(include='object').columns:
            print(f'{col}: {data[col].unique()}')

    # checking for missing values
    print("\nMissing values:")
    print(data.isna().sum())

    # dataframe's shape
    print("\ndataframe's shape:")
    print(data.shape)
    print("\n")


if __name__ == '__main__':
    # checking data
    data_check(train_satisfaction)
    data_check(test_features)
    data_check(target_satisfaction)

# changing data types
train_satisfaction['salary'] = train_satisfaction['salary'].astype('float')
test_features['salary'] = test_features['salary'].astype('float')

# replacing wrong val's
test_features['level'] = test_features['level'].replace('sinior', 'senior')

if __name__ == '__main__':
    # verifying changes
    print("\nVerifying changes")
    print("\nChecking type of feature:\n", train_satisfaction.dtypes)
    print("\nChecking unique values in feature:", test_features['level'].unique())

# reading quit info
try:
    train_quit = pd.read_csv(
        'datasets/quit_predict/train_quit.csv'
    )
except FileNotFoundError:
    train_quit = pd.read_csv(
        '/datasets/train_quit.csv'
    )

try:
    test_quit = pd.read_csv(
        'datasets/quit_predict/test_target_quit.csv'
    )
except FileNotFoundError:
    test_quit = pd.read_csv(
        'datasets/test_target_quit.csv'
    )

if __name__ == '__main__':
    # checking data
    data_check(train_quit)
    data_check(test_quit)

train_quit = train_quit.replace('sinior', 'senior')
train_quit['salary'] = train_quit['salary'].astype('float')

if __name__ == '__main__':
    # verifying changes
    print("Unique Values of 'Level'", train_quit['level'].unique())
    print("Data types:", train_quit.dtypes)
