import os
import numpy as np
import pandas as pd
from datetime import datetime, time
from sklearn.preprocessing import StandardScaler

from src.router import route_features
from src.masking import random_masker


attacks = [
    {'id': 1, 'start_time': '28/12/2015 10:29:14', 'end_time': '28/12/2015 10:44:53', 'points': ['MV-101']},
    {'id': 2, 'start_time': '28/12/2015 10:51:08', 'end_time': '28/12/2015 10:58:30', 'points': ['P-102']},
    {'id': 3, 'start_time': '28/12/2015 11:22:00', 'end_time': '28/12/2015 11:28:22', 'points': ['LIT-101']},
    {'id': 4, 'start_time': '28/12/2015 11:47:39', 'end_time': '28/12/2015 11:54:08', 'points': ['MV-504']},
    {'id': 6, 'start_time': '28/12/2015 12:00:55', 'end_time': '28/12/2015 12:04:10', 'points': ['AIT-202']},
    {'id': 7, 'start_time': '28/12/2015 12:08:04', 'end_time': '28/12/2015 12:15:33', 'points': ['LIT-301']},
    {'id': 8, 'start_time': '28/12/2015 13:10:10', 'end_time': '28/12/2015 13:26:13', 'points': ['DPIT-301']},
    {'id': 10, 'start_time': '28/12/2015 14:16:20', 'end_time': '28/12/2015 14:19:00', 'points': ['FIT-401']},
    {'id': 11, 'start_time': '28/12/2015 14:19:00', 'end_time': '28/12/2015 14:28:20', 'points': ['FIT-401']},
    {'id': 13, 'start_time': '29/12/2015 11:11:25', 'end_time': '29/12/2015 11:15:17', 'points': ['MV-304']},
    {'id': 14, 'start_time': '29/12/2015 11:35:40', 'end_time': '29/12/2015 11:42:50', 'points': ['MV-303']},
    {'id': 16, 'start_time': '29/12/2015 11:57:25', 'end_time': '29/12/2015 12:02:00', 'points': ['LIT-301']},
    {'id': 17, 'start_time': '29/12/2015 14:38:12', 'end_time': '29/12/2015 14:50:08', 'points': ['MV-303']},
    {'id': 19, 'start_time': '29/12/2015 18:10:43', 'end_time': '29/12/2015 18:15:01', 'points': ['AIT-504']},
    {'id': 20, 'start_time': '29/12/2015 18:15:43', 'end_time': '29/12/2015 18:22:17', 'points': ['AIT-504']},
    {'id': 21, 'start_time': '29/12/2015 18:29:58', 'end_time': '29/12/2015 18:42:00', 'points': ['MV-101', 'LIT-101']},
    {'id': 22, 'start_time': '29/12/2015 22:55:18', 'end_time': '29/12/2015 23:03:00', 'points': ['UV-401', 'AIT-502', 'P-501']},
    {'id': 23, 'start_time': '30/12/2015 01:42:34', 'end_time': '30/12/2015 01:54:10', 'points': ['P-602', 'DPIT-301', 'MV-302']},
    {'id': 24, 'start_time': '30/12/2015 09:51:08', 'end_time': '30/12/2015 09:56:28', 'points': ['P-203', 'P-205']},
    {'id': 25, 'start_time': '30/12/2015 10:01:31', 'end_time': '30/12/2015 10:12:01', 'points': ['LIT-401', 'P-401']},
    {'id': 26, 'start_time': '30/12/2015 17:04:56', 'end_time': '30/12/2015 17:29:00', 'points': ['P-101', 'LIT-301']},
    {'id': 27, 'start_time': '31/12/2015 01:17:08', 'end_time': '31/12/2015 01:45:18', 'points': ['P-302', 'LIT-401']},
    {'id': 28, 'start_time': '31/12/2015 01:45:18', 'end_time': '31/12/2015 11:15:27', 'points': ['P-302']},
    {'id': 29, 'start_time': '31/12/2015 15:32:00', 'end_time': '31/12/2015 15:34:00', 'points': ['P-201', 'P-203', 'P-205']},
    {'id': 30, 'start_time': '31/12/2015 15:47:02', 'end_time': '31/12/2015 16:07:10', 'points': ['LIT-101', 'P-101', 'MV-201']},
    {'id': 31, 'start_time': '31/12/2015 22:05:34', 'end_time': '31/12/2015 22:11:40', 'points': ['LIT-401']},
    {'id': 32, 'start_time': '1/01/2016 10:36:00', 'end_time': '1/01/2016 10:46:36', 'points': ['LIT-301']},
    {'id': 33, 'start_time': '1/01/2016 14:21:12', 'end_time': '1/01/2016 14:28:35', 'points': ['LIT-101']},
    {'id': 34, 'start_time': '1/01/2016 17:12:40', 'end_time': '1/01/2016 17:14:20', 'points': ['P-101']},
    {'id': 35, 'start_time': '1/01/2016 17:18:56', 'end_time': '1/01/2016 17:26:56', 'points': ['P-101', 'P-102']},
    {'id': 36, 'start_time': '1/01/2016 22:16:01', 'end_time': '1/01/2016 22:25:43', 'points': ['LIT-101']},
    {'id': 37, 'start_time': '2/01/2016 11:17:02', 'end_time': '2/01/2016 11:25:27', 'points': ['P-501', 'FIT-502']},
    {'id': 38, 'start_time': '2/01/2016 11:31:38', 'end_time': '2/01/2016 11:36:18', 'points': ['AIT-402', 'AIT-502']},
    {'id': 39, 'start_time': '2/01/2016 11:43:48', 'end_time': '2/01/2016 11:50:28', 'points': ['FIT-401', 'AIT-502']},
    {'id': 40, 'start_time': '2/01/2016 11:51:42', 'end_time': '2/01/2016 11:56:38', 'points': ['FIT-401']},
    {'id': 41, 'start_time': '2/01/2016 13:13:02', 'end_time': '2/01/2016 13:40:56', 'points': ['LIT-301']},
]

for attack in attacks:
    attack['start_time_dt'] = datetime.strptime(attack['start_time'], '%d/%m/%Y %H:%M:%S')
    attack['end_time_dt'] = datetime.strptime(attack['end_time'], '%d/%m/%Y %H:%M:%S')


def _ensure_csv(path):
    csv_path = os.path.splitext(path)[0] + '.csv'
    if os.path.exists(csv_path):
        return csv_path
    df = pd.read_excel(path, header=None)
    df.to_csv(csv_path, index=False, header=False)
    return csv_path


def _load_matrix(path, input_size):
    csv_path = _ensure_csv(path)
    ts = np.loadtxt(csv_path, delimiter=',', skiprows=2, usecols=0, dtype=str)
    data = np.loadtxt(csv_path, delimiter=',', skiprows=2, usecols=range(1, input_size + 1))
    columns = np.loadtxt(csv_path, delimiter=',', usecols=range(1, input_size + 1), dtype=str, max_rows=1)
    columns = list(map(str.strip, columns))
    return ts, data.astype(np.float32), columns


def _label_from_attacks(ts):
    labels = np.zeros(len(ts), dtype=np.int32)
    for i, t in enumerate(ts):
        t = t.strip()
        try:
            dt = datetime.strptime(t, '%d/%m/%Y %I:%M:%S %p')
        except ValueError:
            dt = datetime.strptime(t, '%d/%m/%Y %H:%M:%S')
        for attack in attacks:
            if attack['start_time_dt'] <= dt <= attack['end_time_dt']:
                labels[i] = 1
                break
    return labels


def load_swat_windows(data_root, config):
    if isinstance(data_root, dict):
        if 'SWAT' in data_root:
            data_root = data_root['SWAT']
        else:
            data_root = next(iter(data_root.values()), None)

    window = config['window_size']
    stride = config['stride']
    test_stride = config.get('test_stride', stride)
    input_size = int(config.get('input_size', 51))

    print('Training: SWAT Dataset (SARAD-style)')

    train_path = os.path.join(data_root, 'train.xlsx')
    test_path = os.path.join(data_root, 'test.xlsx')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'SWAT train file not found: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'SWAT test file not found: {test_path}')

    train_ts, train_raw, _ = _load_matrix(train_path, input_size)
    test_ts, test_raw, _ = _load_matrix(test_path, input_size)
    _ = train_ts  # kept for parity with SARAD loader shape

    test_labels = _label_from_attacks(test_ts)

    train_scaler = StandardScaler()
    test_scaler = StandardScaler()
    train_total_norm = train_scaler.fit_transform(train_raw)
    test_total_norm = test_scaler.fit_transform(test_raw)

    (train_phy, train_res, test_phy, test_res), topo, _ = route_features(
        train_total_norm, test_total_norm
    )

    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1
        return np.array(
            [data[i * current_stride : i * current_stride + window] for i in range(num_windows)],
            dtype=np.float32,
        )

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)

    v1, v2, v3, v4 = random_masker(train_w_phy)
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4], axis=1)
    rv1, = random_masker(train_res_w, mask_rates=(0.25,))

    train_final = {
        'phy_views': phy_views,
        'res_views': rv1,
        'phy_anchor': train_w_phy,
        'res_orig': train_res_w,
        'topology': topo,
    }

    test_final = {
        'phy': create_windows(test_phy, test_stride),
        'res': create_windows(test_res, test_stride),
        'topology': topo,
    }

    actual_test_len = (test_final['phy'].shape[0] - 1) * test_stride + window
    if test_labels.size < actual_test_len:
        raise ValueError(
            f'SWAT labels shorter than expected: {test_labels.size} < {actual_test_len}'
        )
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, train_scaler.mean_, train_scaler.scale_
