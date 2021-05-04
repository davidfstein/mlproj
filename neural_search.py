import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from skopt.space import Real, Categorical, Integer
from notebooks import preprocess as prepro
import math
import tensorflow.python.keras.backend as KTF


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

def create_model(shape, 
                activation1='relu',
                layer1=128,
                layer2=256,
                dropout1=0.3,
                dropout2=0.3,
                batch_size=32,
                optimizer=None, 
                lr = .1):
    if not optimizer:
         optimizer = keras.optimizers.Adamax(learning_rate=lr, name="Adamax")
    # define model
    model = Sequential()
    model.add(Dense(layer1, activation=activation1, input_dim=shape))
    model.add(Dense(layer1, activation=activation1, input_dim=shape))
    model.add(Dropout(dropout1))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model

if __name__ == '__main__':
    label_col = 'NR.AhR'
    y_train = pd.read_csv('./data/tox21_labels_train.csv.gz', usecols=[label_col])
    X_train, y_train, _ = prepro.tox21dense('./data/tox21_dense_train.csv.gz', labels=y_train, label_col=label_col, select_features='boruta')
    params_dict = {'dropout1': Real(0.01, .9999, 'uniform'),
                   'layer1': Real(64, 512, 'uniform'),
                   'activation1': Categorical(['elu', 'relu', 'selu']),
                   'lr': Real(1e-5, 1, prior='uniform'),
                   'shape': [X_train.shape[1]]
    }
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )
    bayes_cv_tuner = BayesSearchCV(
        estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=32),
        scoring = 'average_precision',
        fit_params={'class_weight': {0:1, 1:10}, 'callbacks': [callback]},
        search_spaces = params_dict,
        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        ),
            n_jobs = 1,
            n_points = 1,
            n_iter = 200,
            verbose = 0,
            refit = False,
            random_state = 1
    )
    result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
