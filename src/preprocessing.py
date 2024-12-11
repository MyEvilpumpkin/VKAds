import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(dir, name):
    return pd.read_csv(os.path.join(dir, f'{name}.tsv'), sep='\t')


def write_data(dir, name, data):
    data.to_csv(os.path.join(dir, f'{name}.tsv'), sep='\t')


def process_history(history):
    user_cpms = {}
    publisher_users = {}

    for _, row in history.iterrows():
        user_id = str(int(row['user_id']))
        publisher = str(int(row['publisher']))
        cpm = row['cpm']

        if user_id not in user_cpms:
            user_cpms[user_id] = []
        user_cpms[user_id].append(cpm)

        if publisher not in publisher_users:
            publisher_users[publisher] = set()
        publisher_users[publisher].add(user_id)
    
    return user_cpms, publisher_users


def peak_hours(hour_start, hour_end):
    peak_hours_count = 0
    for i in range(hour_start, hour_end + 1):
        hour_norm = i % 24
        if hour_norm >= 8 and hour_norm <= 22:
            peak_hours_count += 1
    return peak_hours_count


def process_ad_users(user_ids, user_cpms):
    users = user_ids.split(',')
    counts = []
    cpms = []
    for user in users:
        if user in user_cpms:
            counts.append(len(user_cpms[user]))
            cpms.extend(user_cpms[user])

    return sum(counts), np.mean(counts), np.mean(cpms)


def process_ad_publishers(publisher_ids, publisher_users):
    publishers = publisher_ids.split(',')
    counts = []
    for publisher in publishers:
        if publisher in publisher_users:
            counts.append(len(publisher_users[publisher]))

    return sum(counts), np.mean(counts)


def split_data(data):
    return train_test_split(data, test_size=0.33, random_state=42)


def process_data(data_dir):
    history = read_data(data_dir, 'history')
    ads = read_data(data_dir, 'validate')
    target = read_data(data_dir, 'validate_answers')

    user_cpms, publisher_users = process_history(history)

    ads['publisher_size'] = ads['publishers'].apply(
        lambda publishers: len(publishers.split(','))
    )
    ads['peak_hours'] = ads.apply(
        lambda row: peak_hours(row['hour_start'], row['hour_end']), axis=1
    )
    ads['cpm_x_peak_hours'] = ads['cpm'] * ads['peak_hours']
    ads['publisher_size_x_peak_hours'] = ads['publisher_size'] * ads['peak_hours']
    ads[['users_power', 'mean_users_power', 'mean_cpm_per_users']] = ads['user_ids'].apply(
        lambda user_ids: process_ad_users(user_ids, user_cpms)
    ).apply(pd.Series)
    ads[['active_users_in_publishers', 'mean_active_users_in_publishers']] = ads['publishers'].apply(
        lambda publisher_ids: process_ad_publishers(publisher_ids, publisher_users)
    ).apply(pd.Series)

    ads = ads.drop(columns=['user_ids', 'publishers'])
    ads = ads.drop(columns=['hour_start', 'hour_end'])
    
    ads = ads.merge(target, left_index=True, right_index=True)

    train_data, test_data = split_data(ads)

    write_data(data_dir, 'preprocessed_data_train', train_data)
    write_data(data_dir, 'preprocessed_data_test', test_data)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    process_data(data_dir)
