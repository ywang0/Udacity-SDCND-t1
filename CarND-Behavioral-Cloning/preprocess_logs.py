"""
Usage: python preprocess_logs.py

Utility to preporcess the driving_log.csv to add two columns:'pre_state' and
'avg_steering'. The modified log is saved as 'processed_driving_log.csv'.
"""

import numpy as np
import pandas as pd
import os


log_dirs = ['udacity', 'counter_clockwise', 'clockwise',
            'curves', 'recovery', 'turn_after_bridge_2',
            'track2', 'track2_opposite', 'track2_recovery',
            'track2_curves', 'track2_curves_2',
           ]
data_dir = 'data/'
log_csv = 'driving_log.csv'
p_log_csv = 'processed_driving_log.csv'


def average_steerings(steerings, size=1):
    n_steerings = len(steerings)
    avg_steerings = []

    for i in range(n_steerings):
        if i < size:
            avg_steering = np.mean(steerings[0: i+size+1])
        elif i > n_steerings-size-1:
            avg_steering = np.mean(steerings[i-size:])
        else:
            avg_steering = np.mean(steerings[i-size: i+size+1])
        avg_steerings.append(avg_steering)

    return np.array(avg_steerings)


def preprocess_logs(log_csv, processed_log_csv):
    tdf = pd.read_csv(log_csv)
    pre_state = [[0., 0., 0.,]]
    for i in range(1, tdf.shape[0], 1):
        pre_state.append(list(tdf.iloc[i-1][['steering', 'throttle', 'speed']]))
    tdf['pre_state'] = pre_state
    tdf['avg_steering'] = average_steerings(tdf.steering)
    tdf.to_csv(processed_log_csv, index=False)


if __name__ == '__main__':

    for log_dir in log_dirs:
        log_path = os.path.join(data_dir, log_dir, log_csv)
        processed_log_path = os.path.join(data_dir, log_dir, p_log_csv)
        preprocess_logs(log_path, processed_log_path)
