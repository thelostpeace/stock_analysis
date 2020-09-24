import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import tushare as ts
import datetime
import argparse
import math

predict_days = 5

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
            end_date = datetime.datetime.strptime(str(tmp.iloc[-1].trade_date), '%Y%m%d')
            delta = datetime.timedelta(days=1)
            end_data = (end_date - detla).strftime("%Y%m%d")
            data = data.append(tmp)
            if len(tmp) < 5000:
                break

    data.to_csv(store_file, index=False)

    return data

def add_preday_info(data):
    new_data = data.reset_index(drop=True)
    extend = pd.DataFrame()
    pre_open = []
    pre_high = []
    pre_low = []
    pre_change = []
    pre_pct_chg = []
    pre_vol = []
    pre_amount = []

    for idx in range(len(new_data) - 1):
        pre_open.append(new_data.iloc[idx + 1].open)
        pre_high.append(new_data.iloc[idx + 1].high)
        pre_low.append(new_data.iloc[idx + 1].low)
        pre_change.append(new_data.iloc[idx + 1].change)
        pre_pct_chg.append(new_data.iloc[idx + 1].pct_chg)
        pre_vol.append(new_data.iloc[idx + 1].vol)
        pre_amount.append(new_data.iloc[idx + 1].amount)

    pre_open.append(0.)
    pre_high.append(0.)
    pre_low.append(0.)
    pre_change.append(0.)
    pre_pct_chg.append(0.)
    pre_vol.append(0.)
    pre_amount.append(0.)

    new_data['pre_open'] = pre_open
    new_data['pre_high'] = pre_high
    new_data['pre_low'] = pre_low
    new_data['pre_change'] = pre_change
    new_data['pre_pct_chg'] = pre_pct_chg
    new_data['pre_vol'] = pre_vol
    new_data['pre_amount'] = pre_amount

    # fill predicting target
    days = [[] for i in range(predict_days)]
    for idx in range(predict_days - 1, len(new_data)):
        for i in range(len(days)):
            days[i].append(new_data.iloc[idx - i].pct_chg)

    # fill invalid days with 0.
    for i in range(len(days)):
        for idx in range(predict_days - 1):
            days[i].insert(0, 0.)

    # extend pandas frame
    for i in range(len(days)):
        col = "pct_chg%d" % (i + 1)
        new_data[col] = days[i]

    return new_data

def add_ma_info(data):
    new_data = data.reset_index(drop=True)
    days = [5, 10, 15, 20, 30, 50, 100, 200]
    # add simple ma info
    smas = [[] for i in range(len(days))]
    count = 0
    for day in days:
        for idx in range(len(data) - day + 1):
            smas[count].append(new_data.iloc[idx : idx + day].close.sum() / day)
        count += 1

    count = 0
    for day in days:
        for idx in range(day - 1):
            smas[count].append(0.)
        count += 1

    for day, sma in zip(days, smas):
        new_data['sma%d' % day] = sma

    sma_cols = ["sma%d" % d for d in days]
    pre_smas = [[] for i in range(len(sma_cols))]
    count = 0
    for col in sma_cols:
        pre_smas[count] = new_data.iloc[1:][col].tolist()
        pre_smas[count].append(0.)
        count += 1

    for col, pre_sma in zip(sma_cols, pre_smas):
        new_data["pre_%s" % col] = pre_sma

    # add exponential ma info
    # scaling = s / (1 + d), s is smoothing, typically 2, d is ma days
    # ema(t) = v * scaling + ema(t - 1) * (1 - scaling), v is time(t)'s price
    emas = [[] for i in range(len(days))]
    count = 0
    for day in days:
        scaling = 2. / (1 + day)
        for idx in range(len(data) - 1, -1, -1):
            if idx == len(data) - 1:
                emas[count].append(new_data.iloc[idx].close)
                continue
            emas[count].append(new_data.iloc[idx].close * scaling + (1 - scaling) * emas[count][-1])
        count += 1

    for day, ema in zip(days, emas):
        # reverse ema
        new_data['ema%d' % day] = ema[-1::-1]

    ema_cols = ['ema%d' % d for d in days]
    pre_emas = [[] for i in range(len(days))]
    count = 0
    for col in ema_cols:
        pre_emas[count] = new_data.iloc[1:][col].tolist()
        pre_emas[count].append(0.)
        count += 1

    for col, pre_ema in zip(ema_cols, pre_emas):
        new_data["pre_%s" % col] = pre_ema

    return new_data

def add_rsi_info(data):
    new_data = data.reset_index(drop=True)
    '''
        RSI = 100 - 100 / (1 + RS)
        RS = average up / average down
        average up = sum(up moves) / N
        average downn = sum(down moves) / N
    '''
    # calculate ups and downs
    ups = []
    downs = []
    for idx in range(len(data) - 1):
        if new_data.iloc[idx].close > new_data.iloc[idx + 1].close:
            ups.append(new_data.iloc[idx].close - new_data.iloc[idx + 1].close)
            downs.append(0.)
        else:
            ups.append(0.)
            downs.append(new_data.iloc[idx + 1].close - new_data.iloc[idx].close)
    ups.append(0.)
    downs.append(0.)
    # period of RSI
    N = [2,3,4,5,6]
    for n in N:
        # calculate ema
        up_emas = []
        down_emas = []
        scaling = 2. / (1 + n)
        for idx in range(len(ups) - 1, -1, -1):
            if idx == len(data) - 1:
                up_emas.append(ups[-1])
                down_emas.append(downs[-1])
                continue
            up_emas.append(ups[idx] * scaling + (1 - scaling) * up_emas[-1])
            down_emas.append(downs[idx] * scaling + (1 - scaling) * down_emas[-1])

        # reverse ema
        up_emas = up_emas[-1::-1]
        down_emas = down_emas[-1::-1]
        rsi = []
        for idx in range(len(data) - n):
            rsi.append(100. - 100. / (1. + up_emas[idx] / down_emas[idx]))

        for idx in range(n):
            rsi.append(0.)

        new_data['rsi%d' % n] = rsi
        pre_rsi = rsi[1:]
        pre_rsi.append(0.)
        new_data['pre_rsi%d' % n] = pre_rsi

    return new_data

def add_crossover_info(data):
    # this project is for short-swing trading, so I just
    # track 5-day period ema crossover with 10-day, 15-day, 20-day,
    # 30-day, 50-day, 100-day, 200-day,
    # -1 for breakdowns, 0 for normal, 1 for breakouts
    new_data = data.reset_index(drop=True)
    tracking_day = 'ema5'
    cross_day = ['ema10', 'ema15', 'ema20', 'ema30', 'ema50', 'ema100', 'ema200']
    cross_cols = ['cross5-10', 'cross5-15', 'cross5-20', 'cross5-30', 'cross5-50', 'cross5-100', 'cross5-200']
    for ema, cross_col in zip(cross_day, cross_cols):
        prestatus = 0
        if new_data.iloc[-2][tracking_day] >= new_data.iloc[-2][ema]:
            prestatus = 1
        else:
            prestatus = -1
        crossover = []
        crossover.append(prestatus)
        for idx in range(len(new_data) - 2, -1, -1):
            if prestatus == -1:
                if new_data.iloc[idx][tracking_day] >= new_data.iloc[idx][ema]:
                    crossover.append(1)
                    prestatus = 1
                else:
                    crossover.append(0)
            elif prestatus == 1:
                if new_data.iloc[idx][tracking_day] >= new_data.iloc[idx][ema]:
                    crossover.append(0)
                else:
                    crossover.append(-1)
                    prestatus = -1

        new_data[cross_col] = crossover[-1::-1]

    precross_cols = ['pre_cross5-10', 'pre_cross5-15', 'pre_cross5-20', 'pre_cross5-30', 'pre_cross5-50', 'pre_cross5-100', 'pre_cross5-200']
    for cross_col, precross_col in zip(cross_cols, precross_cols):
        vals = new_data.iloc[1:][cross_col].tolist()
        vals.append(0)
        new_data[precross_col] = vals

    return new_data

def add_features(data):
    new_data = add_preday_info(data)
    new_data = add_ma_info(new_data)
    new_data = add_rsi_info(new_data)
    new_data = add_crossover_info(new_data)

    return new_data

class Model:
    def __init__(self):
        self.features = ['pre_open', 'pre_high', 'pre_low', 'pre_close', 'pre_change', 'pre_pct_chg', 'pre_vol', 'pre_amount', 'pre_sma5', 'pre_sma10', 'pre_sma15', 'pre_sma20', 'pre_sma30', 'pre_sma50', 'pre_sma100', 'pre_sma200', 'pre_ema5', 'pre_ema10', 'pre_ema15', 'pre_ema20', 'pre_ema30', 'pre_ema50', 'pre_ema100', 'pre_ema200', 'pre_rsi2', 'pre_rsi3', 'pre_rsi4', 'pre_rsi5', 'pre_rsi6', 'pre_cross5-10', 'pre_cross5-15', 'pre_cross5-20', 'pre_cross5-30', 'pre_cross5-50', 'pre_cross5-100', 'pre_cross5-200']
        self.targets = ['pct_chg%d' % (i + 1) for i in range(predict_days)]
        self.predict_features = ['open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol', 'amount', 'sma5', 'sma10', 'sma15', 'sma20', 'sma30', 'sma50', 'sma100', 'sma200', 'ema5', 'ema10', 'ema15', 'ema20', 'ema30', 'ema50', 'ema100', 'ema200', 'rsi2', 'rsi3', 'rsi4', 'rsi5', 'rsi6', 'cross5-10', 'cross5-15', 'cross5-20', 'cross5-30', 'cross5-50', 'cross5-100', 'cross5-200']

        self.params = {
            'n_estimators': range(100, 1000, 100),
            'max_depth': range(3, 7, 1),
            'gamma': np.arange(0, 5, 0.5),
            'min_child_weight': range(1, 10, 1),
            'subsample': np.arange(0.6, 1, 0.1),
            'colsample_bytree': np.arange(0.1, 1, 0.1)
        }

        self.xgb = XGBRegressor(learning_rate=0.02, objective='reg:squarederror', n_jobs=6)
        self.models = []
        for i in range(predict_days):
            model = RandomizedSearchCV(self.xgb, param_distributions=self.params, n_iter=10, n_jobs=6, cv=KFold(shuffle=True, random_state=1992), verbose=3, random_state=1992)
            self.models.append(model)

        self.days = predict_days

    def train(self, data):
        for i in range(self.days):
            print("start training pct_chg%d data length: %d" % (i + 1, len(data)))
            self.models[i].fit(data[self.features].to_numpy(), data[self.targets[i]].to_numpy())
            print("done training pct_chg%d data length: %d" % (i + 1, len(data)))

    def predict(self, data):
        pct_chg = []
        print("today:", data[self.predict_features].iloc[0].to_frame())
        for i in range(self.days):
            pct_chg.append(self.models[i].predict(data[self.predict_features].iloc[0].to_numpy().reshape(1, -1))[0])

        return pct_chg

    def plot(self, data, days, stock):
        dots = days + self.days
        x = range(1, dots + 1)
        y = [d for d in reversed(data.iloc[0:days]['close'].tolist())]
        pct_chg = self.predict(data)
        print(pct_chg)
        print("last day close: %s" % y[-1])
        for chg in pct_chg:
            y.append(y[-1] * (1 + chg / 100))
            print("predict: %s" % y[-1])

        sns.lineplot(x=x, y=y)
        sns.lineplot(x=x[:days], y=y[:days])
        plt.savefig('%s.png' % stock)

    def feature_importance(self):
        count = 1
        for model in self.models:
            #print(model.best_estimator_.feature_importances_)
            feature_map = list(zip(self.predict_features, model.best_estimator_.feature_importances_))
            fi = sorted(feature_map, key=lambda v : v[1], reverse=True)
            fi = [v[0] for v in fi]
            print("day%d percent change feature importance: %s" % (count, ' => '.join(fi)))
            count += 1


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
    print("data length: %d" % len(data))
    data = add_features(data)

    model = Model()
    model.train(data.iloc[predict_days - 1:-200])
    #print(model.predict(data))
    model.plot(data, 30, args.stock)
    model.feature_importance()
