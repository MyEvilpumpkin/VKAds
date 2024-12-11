import os
import sys
import joblib

import pandas as pd

from model import VKAdsRegressor


def read_data(dir):
    return pd.read_csv(os.path.join(dir, 'preprocessed_data_train.tsv'), sep='\t')


def features_targets_split(data):
    target_columns = ['at_least_one', 'at_least_two', 'at_least_three']

    features = data.drop(columns=target_columns)
    targets = data[target_columns]

    return features, targets


def save_model(dir, model):
    os.makedirs(dir, exist_ok=True)
    joblib.dump(model, os.path.join(dir, 'model.joblib'))


def train_model(data_dir, model_dir):
    data = read_data(data_dir)

    features, targets = features_targets_split(data)

    model = VKAdsRegressor()
    model.fit(features, targets)

    save_model(model_dir, model)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]
    train_model(data_dir, model_dir)
