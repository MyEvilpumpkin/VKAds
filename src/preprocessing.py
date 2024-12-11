import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(data_dir, name):
    """
    Чтение данных
    """

    return pd.read_csv(os.path.join(data_dir, f'{name}.tsv'), sep='\t')


def write_data(data_dir, name, data):
    """
    Запись данных
    """

    data.to_csv(os.path.join(data_dir, f'{name}.tsv'), sep='\t')


def features_targets_split(data):
    """
    Разделение данных на фичи и таргеты
    """

    target_columns = ['at_least_one', 'at_least_two', 'at_least_three']

    features = data.drop(columns=target_columns)
    targets = data[target_columns]

    return features, targets


def process_history(history):
    """
    Обработка таблицы с историей

    !!! Подробная информация о предобработке находится в research/solution.ipynb
    """

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
    """
    Вычисление пиковых часов

    !!! Подробная информация о предобработке находится в research/solution.ipynb
    """

    peak_hours_count = 0
    for i in range(hour_start, hour_end + 1):
        hour_norm = i % 24
        if 8 <= hour_norm <= 22:
            peak_hours_count += 1
    return peak_hours_count


def process_ad_users(user_ids, user_cpms):
    """
    Обработка целевых пользователей для рекламного объявления

    !!! Подробная информация о предобработке находится в research/solution.ipynb
    """

    users = user_ids.split(',')
    counts = []
    cpms = []
    for user in users:
        if user in user_cpms:
            counts.append(len(user_cpms[user]))
            cpms.extend(user_cpms[user])

    return sum(counts), np.mean(counts), np.mean(cpms)


def process_ad_publishers(publisher_ids, publisher_users):
    """
    Обработка целевых площадок для рекламного объявления

    !!! Подробная информация о предобработке находится в research/solution.ipynb
    """

    publishers = publisher_ids.split(',')
    counts = []
    for publisher in publishers:
        if publisher in publisher_users:
            counts.append(len(publisher_users[publisher]))

    return sum(counts), np.mean(counts)


def process_data(data_dir):
    """
    Обработка данных

    !!! Подробная информация о предобработке находится в research/solution.ipynb
    """

    # Читаем данные
    history = read_data(data_dir, 'history')
    ads = read_data(data_dir, 'validate')
    target = read_data(data_dir, 'validate_answers')

    # Обрабатываем историю
    user_cpms, publisher_users = process_history(history)

    # Обрабатываем рекламу (генерируем фичи)
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

    # Объединяем с таргетами
    ads = ads.merge(target, left_index=True, right_index=True)

    # Делим данные
    train_data, test_data = train_test_split(ads, test_size=0.33, random_state=42)

    # Сохраняем в соответствующие файлы
    write_data(data_dir, 'preprocessed_data_train', train_data)
    write_data(data_dir, 'preprocessed_data_test', test_data)


if __name__ == '__main__':
    data_dir_path = sys.argv[1]
    process_data(data_dir_path)
