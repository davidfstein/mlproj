import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import preprocess as prepro
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer

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

def score(y_true, y_pred):
    try:
        return average_precision_score(y_true, y_pred)
    except:
        return 0.0

# Performs hyperparameter tuning with Bayesian Optimization and StratifiedKFold cross val. This will search for parameters specified in the search space dict.
# The best parameters after every round of training will be printed to std_out. See the scikit-optimize package docs for more information on using Bayesian Optimization.
if __name__ == '__main__':
    label_col = 'NR.AhR'
    y_train = pd.read_csv('./data/tox21_labels_train.csv.gz', usecols=[label_col])
    X_train, y_train, _ = prepro.tox21dense('./data/tox21_dense_train.csv.gz', labels=y_train, label_col=label_col, select_features='boruta')
    bayes_cv_tuner = BayesSearchCV(
        estimator = SVC(),
        scoring = make_scorer(score),
        search_spaces = {
            'C': (.1, 2),
            'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
            'degree': (2,5),
            'gamma': ('scale', 'auto'),
            'coef0': (0, 10),
            'shrinking': (True, False),
            'probability': (True,),
            'class_weight': ('balanced', None),
       },
        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        ),
            n_jobs = 8,
            n_points = 1,
            n_iter = 200,
            verbose = 0,
            refit = False,
            random_state = 1
    )
    result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
