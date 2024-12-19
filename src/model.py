import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor


class VKAdsRegressor(BaseEstimator, RegressorMixin):
    """
    Модель регрессии для рекламных объявлений

    !!! Подробная информация о модели находится в research/solution.ipynb
    """

    def __init__(self, train_independently: bool = False):
        self.train_independently = train_independently

        model1_params = {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.051,
            'l2_leaf_reg': 0.53,
            'border_count': 147
        }

        model2_params = {
            'n_estimators': 310,
            'max_depth': 3,
            'learning_rate': 0.16,
            'min_samples_leaf': 10
        }

        model3_params = {
            'n_estimators': 369,
            'max_depth': 3,
            'learning_rate': 0.14,
            'min_samples_leaf': 10
        }

        self.model1 = CatBoostRegressor(**model1_params, silent=True)
        self.model2 = GradientBoostingRegressor(**model2_params)
        self.model3 = GradientBoostingRegressor(**model3_params)

    def fit(self, X, y):
        self.model1.fit(X, y['at_least_one'])

        if self.train_independently:
            self.model2.fit(y[['at_least_one']].to_numpy(), y['at_least_two'])
            self.model3.fit(y[['at_least_two']].to_numpy(), y['at_least_three'])
        else:
            output1 = np.clip(self.model1.predict(X), a_min=0, a_max=1).reshape(-1, 1)
            self.model2.fit(output1, y['at_least_two'])

            output2 = self.model2.predict(output1).reshape(-1, 1)
            self.model3.fit(output2, y['at_least_three'])

        return self

    def predict(self, X):
        output1 = np.clip(self.model1.predict(X), a_min=0, a_max=1)
        output2 = np.clip(self.model2.predict(output1.reshape(-1, 1)), a_min=0, a_max=1)
        output3 = np.clip(self.model3.predict(output2.reshape(-1, 1)), a_min=0, a_max=1)

        output = pd.DataFrame()
        output['at_least_one'] = output1
        output['at_least_two'] = output2
        output['at_least_three'] = output3
        output = output.set_index(X.index)

        return output
