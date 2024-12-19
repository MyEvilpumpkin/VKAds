import os
import sys
import joblib

from preprocessing import read_data, features_targets_split
from model import VKAdsRegressor


def save_model(model_dir, model):
    """
    Сохранение модели
    """

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))


def train_model(data_dir, model_dir):
    """
    Тренировка модели и сохранение в файл
    """

    data = read_data(data_dir, 'preprocessed_data_train')
    features, targets = features_targets_split(data)

    model = VKAdsRegressor(train_independently=True)
    model.fit(features, targets)

    save_model(model_dir, model)


if __name__ == '__main__':
    data_dir_path = sys.argv[1]
    model_dir_path = sys.argv[2]
    train_model(data_dir_path, model_dir_path)
