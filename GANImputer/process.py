# from loguru import logger
import pandas as pd
import numpy as np
import os


def sample_data(p, rows, cols):
    random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (random_matrix < p)
    return binary_random_matrix


def process_data(name, miss_rate):
    # Random drop data
    data = pd.read_table(f"data/{name}.csv", sep=",")
    attrs = list(data.columns.values)
    data_x = np.array(data)
    data_x = data_x.astype(float)
    row, col = data_x.shape
    sign = sample_data(1-miss_rate, row, col)
    data_miss = data_x.copy()
    data_miss[sign == 0] = np.nan

    # Minmax standardization
    data_x = data_x.astype(np.float64)
    data_miss = data_miss.astype(np.float64)
    for i in range(col):
        min_val = np.nanmin(data_miss[:, i])
        max_val = np.nanmax(data_miss[:, i])
        data_x[:, i] = (data_x[:, i] - min_val) / (max_val - min_val + 1e-6)
        data_miss[:, i] = (data_miss[:, i] - min_val) / (max_val - min_val + 1e-6)
    data_x[data_x > 100*np.nanmax(data_miss)] = 1
    # logger.info(f"The model is built on {name} data set")
    # logger.info(f"The missing rate is {miss_rate}")
    data_df = pd.DataFrame(data_miss, columns=attrs)
    sign_df = pd.DataFrame(sign, columns=attrs)
    std_df = pd.DataFrame(data_x, columns=attrs)
    data_df.to_csv(f"data/{name}_clean.csv", index=False)
    sign_df.to_csv(f"data/{name}_sign.csv", index=False)
    std_df.to_csv(f"data/{name}_std.csv", index=False)
