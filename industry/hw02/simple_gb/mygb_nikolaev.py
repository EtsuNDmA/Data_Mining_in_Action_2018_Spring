from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from scipy.special import expit

# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {
    'max_depth': 8,
    'min_samples_leaf': 3,
    'min_samples_split': 25,
    'random_state': 42,
}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.19


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau

    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            grad = expit(curr_pred) - y_data
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, -grad)

            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            curr_pred += self.tau * algo.predict(X_data)
        return self

    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return expit(res) > 0.5
