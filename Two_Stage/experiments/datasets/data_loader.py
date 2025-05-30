# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader functions to read various tabular datasets."""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from tensorflow.keras.utils import to_categorical
from scipy.stats import mode
import matplotlib.pyplot as plt

import sys
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def load_activity():
    """Loads the Activity dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

    cache_filepath_train_x = os.path.join(DATA_DIR, "activity/X_train.txt")
    cache_filepath_train_y = os.path.join(DATA_DIR, "activity/y_train.txt")
    cache_filepath_test_x = os.path.join(DATA_DIR, "activity/X_test.txt")
    cache_filepath_test_y = os.path.join(DATA_DIR, "activity/y_test.txt")
    with open(cache_filepath_train_x, "r") as fp:
        x_train = np.genfromtxt(fp.readlines(), encoding="UTF-8")
    with open(cache_filepath_test_x, "r") as fp:
        x_test = np.genfromtxt(fp.readlines(), encoding="UTF-8")
    with open(cache_filepath_train_y, "r") as fp:
        y_train = np.genfromtxt(fp.readlines(), encoding="UTF-8")
    with open(cache_filepath_test_y, "r") as fp:
        y_test = np.genfromtxt(fp.readlines(), encoding="UTF-8")

    x = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    x_train = x[: len(y_train)]
    x_test = x[len(y_train):]

    print("Data loaded...")
    print("Data shapes:")
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    is_classification = True
    num_classes = 6

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train - 1, dtype=np.int32).iloc[:, 0]
    y_test = pd.DataFrame(y_test - 1, dtype=np.int32).iloc[:, 0]

    return (x_train, x_test, y_train, y_test, is_classification, num_classes)

def load_DSA():
    """
  加载DSA数据集，并划分测试集和训练集。

  参数:
  data_dir (str): 存放数据集的根目录路径。
  test_size (float): 测试集所占的比例，默认是0.2，即20%。

  返回:
  tuple: 包含训练特征、测试特征、训练标签、测试标签、是否为分类任务、类别数量的元组。
  """
    data_dir = "/Users/mac/Desktop/data"
    test_size = 0.2
    activities = [f"a{i:02d}" for i in range(1, 20)]  # a01, a02, ..., a19
    subjects = [f"p{i}" for i in range(1, 9)]  # p1, p2, ..., p8
    sessions = [f"s{i:02d}" for i in range(1, 61)]  # s01, s02, ..., s60

    data = []
    labels = []

    for activity in activities:
        for subject in subjects:
            subject_dir = os.path.join(data_dir, activity, subject)

            for session in sessions:
                file_path = os.path.join(subject_dir, f"{session}.txt")

                if os.path.exists(file_path):
                    # 读取文件中的数据
                    with open(file_path, "r") as file:
                        raw_data = np.loadtxt(file, delimiter=",")

                        # 每个数据文件包含 125 行 x 45 列
                        assert raw_data.shape == (125, 45), f"Unexpected data shape in {file_path}"

                        # 数据预处理（将数据展平或其它处理方式）
                        data.append(raw_data.flatten())

                        # 标签：提取活动编号，假设活动编号是从文件路径中提取的
                        labels.append(int(activity[1:]) - 1)

    # 转换为 NumPy 数组
    data = np.array(data)
    labels = np.array(labels)
    # 检查 data 是否为一维数组
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # 重塑为二维数组

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # 打印数据形状
    print("Data loaded and processed...")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # 假设这是分类任务，类别数量需要根据实际数据集进行调整
    is_classification = True
    num_classes = len(np.unique(labels))

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train - 1, dtype=np.int32).iloc[:, 0]
    y_test = pd.DataFrame(y_test - 1, dtype=np.int32).iloc[:, 0]

    return (x_train, x_test, y_train, y_test, is_classification, num_classes)
    #return (
    #    pd.DataFrame(x_train), pd.DataFrame(x_test), pd.Series(y_train), pd.Series(y_test), is_classification,
    #    num_classes)


def read_data(filepath):
    # 读取数据
    df = pd.read_csv(filepath, header=None, names=['user-id', 'activity', 'timestamp', 'X', 'Y', 'Z'])

    # 清理数据
    df['Z'].replace(regex=True, inplace=True, to_replace=r';', value=r'')
    df['Z'] = df['Z'].apply(lambda x: np.float64(x) if x != '' else np.nan)

    # 标签编码
    label_encode = LabelEncoder()
    df['activityEncode'] = label_encode.fit_transform(df['activity'].values.ravel())

    # 线性插值填补空值
    interpolation_fn = interp1d(df['activityEncode'], df['Z'], kind='linear')
    null_list = df[df['Z'].isnull()].index.tolist()
    for i in null_list:
        y = df['activityEncode'][i]
        value = interpolation_fn(y)
        df['Z'] = df['Z'].fillna(value)

    # 划分数据集
    df_train = df[df['user-id'] <= 27]
    df_test = df[df['user-id'] > 27]

    # 归一化
    df_train[['X', 'Y', 'Z']] = df_train[['X', 'Y', 'Z']].apply(lambda col: (col - col.min()) / (col.max() - col.min()))
    df_test[['X', 'Y', 'Z']] = df_test[['X', 'Y', 'Z']].apply(lambda col: (col - col.min()) / (col.max() - col.min()))


    return df_train, df_test, label_encode


# 时间序列分割函数
def segments(df, time_steps, step, label_name):
    N_FEATURES = 3
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['X'].values[i:i + time_steps]
        ys = df['Y'].values[i:i + time_steps]
        zs = df['Z'].values[i:i + time_steps]

        label = mode(df[label_name][i:i + time_steps])[0]

        segments.append([xs, ys, zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

def load_WISDM(time_steps=80, step=40, label_name='activity'):
    """
    加载和预处理WISDM数据集，划分训练集和测试集。

    参数:
    filepath (str): 数据文件的路径。
    time_steps (int): 每个时间片段的步长，默认80。
    step (int): 每次分割的数据步长，默认40。
    label_name (str): 标签的列名，默认 'activity'。

    返回:
    tuple: 包含训练特征、测试特征、训练标签、测试标签、是否为分类任务、类别数量的元组。
    """
    filepath='/Users/mac/Desktop/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    df_train, df_test, label_encode = read_data(filepath)

    # 时间序列切割
    x_train, y_train = segments(df_train, time_steps, step, 'activityEncode')
    x_test, y_test = segments(df_test, time_steps, step, 'activityEncode')

    # 获取输入的形状
    time_period, sensors = x_train.shape[1], x_train.shape[2]
    num_classes = label_encode.classes_.size

    # 重塑输入数据形状
    input_shape = time_period * sensors
    x_train = x_train.reshape(x_train.shape[0], input_shape)
    x_test = x_test.reshape(x_test.shape[0], input_shape)

    x_train = pd.DataFrame(x_train)  # 将 x_train 转换为 DataFrame
    x_test= pd.DataFrame(x_test)  # 将 x_test 转换为 DataFrame

    # 进行标签独热编码
    y_train_hot = to_categorical(y_train, num_classes)
    y_test_hot = to_categorical(y_test, num_classes)

    y_train = pd.DataFrame(y_train - 1, dtype=np.int32).iloc[:, 0]
    y_test = pd.DataFrame(y_test - 1, dtype=np.int32).iloc[:, 0]

    print("Data loaded and processed...")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # 返回数据
    return x_train, x_test, y_train, y_test, True, num_classes


def preprocess_opportunity_data(window_size=500, step_size=250, threshold=0.5):
    """
    预处理 Opportunity 数据集，进行滑动窗口分割和缺失值处理。

    参数：
    - file_paths: 数据文件路径列表，包括所有的 .dat 文件。
    - window_size: 滑动窗口的大小（默认为500 ms）。
    - step_size: 滑动窗口的步长（默认为250 ms）。
    - threshold: 缺失值阈值（默认为0.5，表示超过50%的缺失值的列会丢弃）。

    返回：
    - 处理后的训练和测试数据集（x_train, x_test, y_train, y_test）。
    """
    base_path='/Users/mac/Desktop/数据集相关论文/OpportunityUCIDataset/dataset/'
    # 1. 读取数据文件
    subjects = ["S1", "S2", "S3", "S4"]
    files = [f"{s}-ADL{adl}.dat" for s in subjects for adl in range(1, 6)] + \
            [f"{s}-Drill.dat" for s in subjects]

    full_data = []
    for file in files:
        df = pd.read_csv(base_path + file, header=None, delimiter=",", skiprows=1, on_bad_lines="skip",
                         low_memory=False)

        # 提取传感器数据（2-102列）和标签（244列）
        sensor_data = df.iloc[:, 1:102].astype(float)  # 获取第2列到第102列 (索引1到101)
        labels = df.iloc[:, 243]  # 获取第244列（索引243）

        # 过滤有效标签（1, 2, 4, 5是有效标签）
        valid_mask = labels.isin([1, 2, 4, 5])
        full_data.append(pd.concat([sensor_data[valid_mask], labels[valid_mask]], axis=1))

    # 合并所有文件的数据
    full_df = pd.concat(full_data, ignore_index=True)

    # 2. 滑动窗口分割（500ms窗口，250ms步长）
    window_size = 15  # 假设30Hz采样率，500ms=15样本
    step = 7  # 250ms=7样本
    x, y = [], []

    # 获取传感器数据和标签
    data = full_df.iloc[:, :-1].values  # 传感器数据：所有列（除最后一列）
    labels = full_df.iloc[:, -1].values  # 标签数据：最后一列

    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        label = labels[i + window_size - 1]

        # 3. 缺失值处理
        missing = np.isnan(window).sum()
        if missing > (window.size * 0.5):  # 如果超过50%缺失，跳过
            continue

        # 前向填充并补零剩余缺失
        df_window = pd.DataFrame(window).ffill().fillna(0)
        x.append(df_window.values)
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    # 4. 划分数据集并编码标签
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    le = LabelEncoder()
    y_train_hot = to_categorical(le.fit_transform(y_train))
    y_test_hot = to_categorical(le.transform(y_test))
    num_classes = y_train_hot.shape[1]

    print("Data loaded and processed...")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train_hot.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test_hot.shape}")

    return x_train, x_test, y_train_hot, y_test_hot, True, num_classes


