import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
import optuna
from sklearn.model_selection import cross_val_score

class VKAdsRegressor(BaseEstimator, RegressorMixin):
    """
    Модель регрессии для рекламных объявлений с настройкой гиперпараметров с помощью Optuna

    !!! Подробная информация о модели находится в research/solution.ipynb
    """

    def __init__(self, train_independently: bool = False, tune_hyperparameters: bool = False, n_trials: int = 100):
        self.train_independently = train_independently
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.model1 = CatBoostRegressor(silent=True)
        self.model2 = GradientBoostingRegressor()
        self.model3 = GradientBoostingRegressor()

    def _objective_catboost(self, trial, X, y):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
        }
        model = CatBoostRegressor(**params, silent=True)
        return np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))

    def _objective_gb(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
        model = GradientBoostingRegressor(**params)
        return np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))

    def _tune_model(self, objective, X, y):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X, y), n_trials=self.n_trials)
        return study.best_params
    def fit(self, X, y):
        if self.tune_hyperparameters:
            # Tune model1
            best_params = self._tune_model(self._objective_catboost, X, y['at_least_one'])
            self.model1 = CatBoostRegressor(**best_params, silent=True)

        # Fit model1
        self.model1.fit(X, y['at_least_one'])

        if self.train_independently:
            if self.tune_hyperparameters:
                # Tune model2 and model3
                best_params2 = self._tune_model(self._objective_gb, y[['at_least_one']].to_numpy(), y['at_least_two'])
                best_params3 = self._tune_model(self._objective_gb, y[['at_least_two']].to_numpy(), y['at_least_three'])
                self.model2 = GradientBoostingRegressor(**best_params2)
                self.model3 = GradientBoostingRegressor(**best_params3)

            # Fit model2 and model3
            self.model2.fit(y[['at_least_one']].to_numpy(), y['at_least_two'])
            self.model3.fit(y[['at_least_two']].to_numpy(), y['at_least_three'])
        else:
            output1 = np.clip(self.model1.predict(X), a_min=0, a_max=1).reshape(-1, 1)

            if self.tune_hyperparameters:
                # Tune model2
                best_params2 = self._tune_model(self._objective_gb, output1, y['at_least_two'])
                self.model2 = GradientBoostingRegressor(**best_params2)

            # Fit model2
            self.model2.fit(output1, y['at_least_two'])

            output2 = self.model2.predict(output1).reshape(-1, 1)

            if self.tune_hyperparameters:
                # Tune model3
                best_params3 = self._tune_model(self._objective_gb, output2, y['at_least_three'])
                self.model3 = GradientBoostingRegressor(**best_params3)

            # Fit model3
            self.model3.fit(output2, y['at_least_three'])

        return self

    def predict(self, X):
        # The predict method remains unchanged
        output1 = np.clip(self.model1.predict(X), a_min=0, a_max=1)
        output2 = self.model2.predict(output1.reshape(-1, 1))
        output3 = self.model3.predict(output2.reshape(-1, 1))

        output = pd.DataFrame()
        output['at_least_one'] = output1
        output['at_least_two'] = output2
        output['at_least_three'] = output3
        output = output.set_index(X.index)

        return output

