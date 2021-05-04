import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn import metrics
import preprocess as prepro
import math

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest F1: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")


if __name__ == '__main__':
    label_col = 'NR.AhR'
    y_train = pd.read_csv('./data/tox21_labels_train.csv.gz', usecols=[label_col])
    X_train, y_train, _ = prepro.tox21dense('./data/tox21_dense_train.csv.gz', labels=y_train, label_col=label_col, select_features='boruta')
    bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(
        objective='binary:logistic',
    ),
    scoring = 'average_precision',
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-5, 1000),
        'reg_alpha': (1e-5, 1.0),
        'gamma': (1e-5, 0.5),
        'n_estimators': (200, 5000),
        'scale_pos_weight': (1e-5, 500),
    },
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
        n_jobs = 40,
        n_points = 5,
        n_iter = 200,
        verbose = 0,
        refit = True,
        random_state = 1
    )
    result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
