import os
import sys
import joblib

from preprocessing import read_data, features_targets_split
from metrics import get_smoothed_mean_log_accuracy_ratio


def read_model(model_dir):
    """
    Чтение модели
    """

    return joblib.load(os.path.join(model_dir, 'model.joblib'))


def test_model(data_dir, model_dir):
    """
    Тестирование модели и вывод на метрики
    """

    data = read_data(data_dir, 'preprocessed_data_test')
    features, targets = features_targets_split(data)

    model = read_model(model_dir)
    result = model.predict(features)

    # Используем предложенную метрику
    print(f'Metric = {get_smoothed_mean_log_accuracy_ratio(targets, result)}')


if __name__ == '__main__':
    data_dir_path = sys.argv[1]
    model_dir_path = sys.argv[2]
    test_model(data_dir_path, model_dir_path)
