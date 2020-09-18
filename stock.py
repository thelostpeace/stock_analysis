import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import tushare as ts
import datetime
import argparse


def check_stock_data(name):
    files = glob.glob(name)

    return (len(files) != 0)

def get_stock_data(name, store_file):
    api = ts.pro_api(token='9a29d45bfd6a127f24365620f9bb730f557abaacff59656b0218b843')
    today = datetime.date.today()
    if check_stock_data(store_file):
        data = pd.read_csv(store_file)
        saved_date = datetime.datetime.strptime(str(data.iloc[0].trade_date), '%Y%m%d')
        delta = datetime.timedelta(days=1)
        start_date = saved_date + delta
        extend_data = ts.pro_bar(ts_code=name, api=api, start_date=start_date.strftime("%Y%m%d"), end_date=today.strftime("%Y%m%d"), adj='qfq')
        print("extend data length: %d" % len(extend_data))
        data = extend_data.append(data)
    else:
        data = pd.DataFrame()
        end_date = today.strftime("%Y%m%d")
        while True:
            tmp = ts.pro_bar(ts_code=name, api=api, end_date=end_date, adj='qfq')
            print("get data length: %d, end_date: %s" % (len(tmp), end_date))
            end_date = tmp.iloc[-1].trade_date
            data = data.append(tmp)
            if len(tmp) < 5000:
                break

    data.to_csv(store_file, index=False)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, default='000519.SZ')
    parser.add_argument('--update_data', action='store_true')
    parser.set_defaults(update_data=False)
    args = parser.parse_args()
    filename = "data/%s.csv" % args.stock
    if args.update_data:
        data = get_stock_data(stock, filename)
    else:
        data = pd.read_csv(filename)
    data = data.dropna(axis=0)
