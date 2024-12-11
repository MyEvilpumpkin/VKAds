import sys

from preprocessing import process_data
from training import train_model
from testing import test_model


if __name__ == '__main__':
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]

    process_data(data_dir)
    train_model(data_dir, model_dir)
    test_model(data_dir, model_dir)
