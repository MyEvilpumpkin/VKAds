import os
import sys
import joblib

import pandas as pd

from metrics import get_smoothed_mean_log_accuracy_ratio


def read_data(dir):
    return pd.read_csv(os.path.join(dir, 'preprocessed_data_test.tsv'), sep='\t')


def features_targets_split(data):
    target_columns = ['at_least_one', 'at_least_two', 'at_least_three']

    features = data.drop(columns=target_columns)
    targets = data[target_columns]

    return features, targets


def read_model(dir):
    return joblib.load(os.path.join(dir, 'model.joblib'))


def print_metric(targets, result):
    print(f'Metric = {get_smoothed_mean_log_accuracy_ratio(targets, result)}')


def test_model(data_dir, model_dir):
    data = read_data(data_dir)

    features, targets = features_targets_split(data)

    model = read_model(model_dir)

    result = model.predict(features)

    print_metric(targets, result)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]
    test_model(data_dir, model_dir)
