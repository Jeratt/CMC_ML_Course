from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from numpy import ndarray

"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    features = {
        'categorial': {
            13: 'actor_0_gender',
            17: 'actor_1_gender',
            21: 'actor_2_gender'
        },
        'text': {
            1: 'genres',
            3: 'directors',
            4: 'filming_locations'
        }
    }

    y_train = df_train["awards"]
    df_train.drop(['awards'], axis=1, inplace=True)
    df_train.drop(['keywords'], axis=1, inplace=True)

    df_test.drop(['keywords'], axis=1, inplace=True)

    df_train[list(features['categorial'].values())] = \
        df_train[list(features['categorial'].values())].apply(lambda x: x.astype('category'))

    df_test[list(features['categorial'].values())] = \
        df_test[list(features['categorial'].values())].apply(lambda x: x.astype('category'))

    for i, col in features['text'].items():
        vectorizer = TfidfVectorizer()
        df_train[col] = df_train[col].apply(lambda x: " ".join(x) if (type(x) == list) else "")
        df_test[col] = df_test[col].apply(lambda x: " ".join(x) if (type(x) == list) else "")
        vectorizer.fit(df_train[col])
        col_transformed_train = vectorizer.transform(df_train[col])
        col_transformed_test = vectorizer.transform(df_test[col])
        for j in range(col_transformed_train.shape[1]):
            df_train[col + "_tfidf_" + str(j)] = col_transformed_train[:, j].toarray()
        for j in range(col_transformed_test.shape[1]):
            df_test[col + "_tfidf_" + str(j)] = col_transformed_test[:, j].toarray()
        df_train.drop([col], axis=1, inplace=True)
        df_test.drop([col], axis=1, inplace=True)

    ml_best = {
        'learning_rate': 0.03043399465956341,
        'max_depth': 11,
        'n_estimators': 633
    }

    regressor = LGBMRegressor(**ml_best)
    regressor.fit(df_train, y_train, categorical_feature='auto')
    return regressor.predict(df_test)
