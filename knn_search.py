import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
        estimator = KNeighborsClassifier(),
        scoring = 'average_precision',
        search_spaces = {
            'n_neighbors': (1, 10),
            'weights': ('uniform', 'distance'),
            'algorithm': ('auto','ball_tree','kd_tree','brute'),
            'leaf_size': (10, 200),
            'p': [1,2],
       },
        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        ),
            n_jobs = 7,
            n_points = 3,
            n_iter = 200,
            verbose = 0,
            error_score = 0.0,
            refit = False,
            random_state = 1
    )
    result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
