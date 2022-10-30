import os
import json
import pandas as pd
import yfinance as yf
import numpy as np
import datetime 

def get_stock_data(code, start, end):
    data = yf.download(code, start, end)
    data = data.drop(data[data.Volume < 10].index)
    data.index.name = "Date"
    return data


def stock_download(
    dic,
    start="2022-01-01",
    end="2022-10-30",
    download_dir="data/stocks/",
):
    os.makedirs(download_dir, exist_ok=True)
    count = 0
    stock_dict = {}
    for symbol in dic:
      data = get_stock_data(symbol, start, end)
      data.to_csv(download_dir + f"{symbol}.csv")

def make_DL_dataset(data, data_len, n_stock):
    times = []
    dataset = np.array(data.iloc[:data_len, :]).reshape(1, -1, n_stock)
    times.append(data.iloc[:data_len, :].index)

    for i in range(1, len(data) - data_len + 1):
        addition = np.array(data.iloc[i : data_len + i, :]).reshape(1, -1, n_stock)
        dataset = np.concatenate((dataset, addition))
        times.append(data.iloc[i : data_len + i, :].index)
    return dataset, times

def data_split(data, train_len, pred_len, split_date, n_stock):
    train_cnt = sum(data.index < split_date)
    return_train, times_train = make_DL_dataset(
        data[:train_cnt], train_len + pred_len, n_stock
    )
    return_test, times_test = make_DL_dataset(
        data[train_cnt - train_len:], train_len + pred_len, n_stock
    )

    x_tr = np.array([x[:train_len] for x in return_train])
    y_tr = np.array([x[-pred_len:] for x in return_train])
    times_tr = np.unique(
        np.array([x[-pred_len:] for x in times_train]).flatten()
    ).tolist()

    x_te = np.array([x[:train_len] for x in return_test])
    y_te = np.array([x[-pred_len:] for x in return_test])
    times_te = np.unique(
        np.array([x[-pred_len:] for x in times_test]).flatten()
    ).tolist()

    return x_tr, y_tr, x_te, y_te, times_tr, times_te


if __name__ == "__main__":
  dic = ["aapl", "pdd", "tsla", "meta", "jpm", "amd", "ngg", "biib", "ba", "ryaay"]
  stock_pair = stock_download(
      dic, download_dir='data/stocks/'
  )
  
  json.dump(dic, open("data/stock.json", "w", encoding="UTF-8"))