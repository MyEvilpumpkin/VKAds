import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor


class VKAdsRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, train_independently: bool = False):
        self.train_independently = train_independently
        self.model1 = CatBoostRegressor(silent=True)
        self.model2 = GradientBoostingRegressor()
        self.model3 = GradientBoostingRegressor()

    def _wrap_to_df(self, name, data):
        df = pd.DataFrame()
        df[name] = data
        return df

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
        output2 = self.model2.predict(output1.reshape(-1, 1))
        output3 = self.model3.predict(output2.reshape(-1, 1))

        output = pd.DataFrame()
        output['at_least_one'] = output1
        output['at_least_two'] = output2
        output['at_least_three'] = output3
        output = output.set_index(X.index)
        
        return output
