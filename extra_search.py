import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.ensemble import ExtraTreesClassifier
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
        estimator = ExtraTreesClassifier(),
        scoring = 'average_precision',
        search_spaces = {
            'n_estimators': (200, 5000),
            'max_features': ('auto', 'sqrt', 'log2', None),
            'max_depth': (1, 200),
            'min_samples_split': (1e-5, 1),
            'min_samples_leaf': (1, 20),
            'class_weight': ('balanced', 'balanced_subsample', None),
            'bootstrap': (True, False)
        },
        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        ),
            n_jobs = 80,
            n_points = 3,
            n_iter = 200,
            verbose = 0,
            refit = False,
            random_state = 1
    )
    result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
