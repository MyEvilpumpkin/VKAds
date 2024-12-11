import sys

from preprocessing import process_data
from training import train_model
from testing import test_model


if __name__ == '__main__':
    data_dir_path = sys.argv[1]
    model_dir_path = sys.argv[2]

    process_data(data_dir_path)
    train_model(data_dir_path, model_dir_path)
    test_model(data_dir_path, model_dir_path)
