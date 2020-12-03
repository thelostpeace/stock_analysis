#!/usr/bin/env python
import numpy as np
import pandas as pd
from matplotlib import rcParams
#rcParams['font.family'] = ['Nimbus Sans L']
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tushare as ts
import datetime
import argparse
import math
import ta
import os, sys, random
import smtplib
import imghdr
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import subprocess
import mplfinance as mpf

stock_index = ['000001.SH']
code2name = dict()

predict_days = 5
api = ts.pro_api(token='6fd0d52251fd78f819527832d0ad920feea9acd672d7f296a02efea3')

trading_note = """
本邮件由程序自动发送，勿回复，谢谢！

### 结论

经过这段时间的测试，选出以下指标，其中close为收盘价

 - rsi2: 即两日RSI，2周期RSI是一个比较敏感的指标，对短期阶段性的买点判断比较准确，但是对买点判断不准确，只依靠2周期RSI容易卖飞，遇到差行情很容易回撤

 - boll_wband20: 20周期的bollinger bands，对于短期趋势的判定很准确，当价格线上穿boll_wband20并且与boll_wband20同趋势的时候，是很强的上升势

 - vwap30: 30周期的volume weighted average price，当价格线上穿vwap30并且与vwap30同趋势的时候，是很强的上升势

 - kc_wband15: 15周期的keltner channel，当价格线上穿kc_wband15并且与kc_wband15同趋势的时候，是很强的上升势

 - macd: 快线5周期，慢线15周期，信号线7周期，当macd线上穿信号线的时候是上升趋势，但是有一定的延时性

 - adx15: 15周期average directional movement index, 当+DMI > -DMI的时候是上升势

 - trix2: 2周期trix，当trix2上穿价格线并且与价格线同趋势的时候，是很强的上升势

 - mi: mass index，当价格线上穿mi并且与mi同趋势的时候，是很强的上升势

 - cci5: 5周期的commodity channel index，非常敏感，当cci5 > close并且没有很明显的下降趋势的时候，是上升势

 - kst: kst oscillator, 当kst上穿信号线并且同趋势的时候，是很强的上升势，有误判情况

 - psar: parabolic stop and reverse，每次价格线上穿psar都是买点

 - tsi: true strength index，tsi上穿价格线是很强的上升势

 - wr15: 15周期williams percent range，当wr15上穿价格线并持续保持在价格线之上，是上升势

 - roc15: 15周期rate of change，当roc15上穿价格线并保持在价格线之上，是上升势

 - kama: kaufman's adaptive moving average, 当价格线上穿kama，是上升势
"""

def check_stock_data(name):
    files = glob.glob(name)

    return (len(files) != 0)

def get_stock_data(name, weekly):
    data = pd.DataFrame()
    end_date = api.daily().iloc[0]['trade_date']
    while True:
        if weekly:
            tmp = api.weekly(ts_code=name, end_date=end_date)
        else:
            tmp = ts.pro_bar(ts_code=name, api=api, end_date=end_date, adj='qfq')
        print("get data length: %d, end_date: %s" % (len(tmp), end_date))
        end_date = datetime.datetime.strptime(str(tmp.iloc[-1].trade_date), '%Y%m%d')
        delta = datetime.timedelta(days=1)
        end_date = (end_date - delta).strftime("%Y%m%d")
        data = data.append(tmp)
        if len(tmp) < 5000:
            break

    return data

def get_index_data(name, weekly):
    today = datetime.date.today().strftime("%Y%m%d")
    data = api.index_daily(ts_code=name)
    if str(data.iloc[0].trade_date) != today:
        print("today's index data is not ready, last trading day is %s" % data.iloc[0].trade_date)

    return data

def get_stock_candidates():
    today = datetime.date.today().strftime("%Y%m%d")
    last_trading_day = api.daily().iloc[0]['trade_date']
    if today != last_trading_day:
        print("today's stock data is not ready, get stock candidates of %s" % last_trading_day)
    df = api.daily_basic(trade_date=last_trading_day)
    # 选取量在20w以上的, 价格在5-50之间的
    candidates = df[(df.float_share * df.turnover_rate_f > 200000.) & (df.close > 5.) & (df.close < 50.)]["ts_code"].tolist()

    return candidates

def get_code_name_map():
    global code2name
    global api
    df = api.stock_basic()
    for code, name in zip(df['ts_code'].to_list(), df['name'].to_list()):
        code2name[code] = name

    df = api.index_basic()
    for code, name in zip(df['ts_code'].to_list(), df['name'].to_list()):
        code2name[code] = name

def calculate_index(days, K):
    global code2name
    # days 交易日，最近的在前
    last_day = api.daily().iloc[0]['trade_date']
    print('last trade day: %s' % last_day)
    open_cal = api.trade_cal(is_open='1', end_date=last_day)['cal_date'].to_list()[-1::-1][:days+20]
    data = pd.DataFrame()
    trade_date_ = []
    open_ = []
    high_ = []
    low_ = []
    close_ = []
    vol_ = []
    amount_ = []
    top_K = []
    r_top_K = []
    w_top_K = []
    for day in open_cal:
        df = api.daily(trade_date=day)
        df2 = api.daily_basic(trade_date=day)
        df = df[df.ts_code.isin(df2.ts_code.tolist())]
        df = df.sort_values('ts_code').reset_index()
        df2 = df2.sort_values('ts_code').reset_index()
        df['circ_mv'] = df2['circ_mv']
        amount = df.circ_mv.sum()
        df['weight'] = df['circ_mv'] / amount * 100
        df['open'] = df['open'] * df['weight']
        df['high'] = df['high'] * df['weight']
        df['low'] = df['low'] * df['weight']
        df['close'] = df['close'] * df['weight']
        trade_date_.append(day)
        open_.append(df.open.sum())
        high_.append(df.high.sum())
        low_.append(df.low.sum())
        close_.append(df.close.sum())
        vol_.append(df.vol.sum() / 10000.)
        amount_.append(df.amount.sum() / 100000.)
        cand = df.sort_values('weight', ascending=False).iloc[:K][['ts_code', 'weight']].to_numpy()
        top_ = ["%s%+.3f%%" % (code2name[item[0]], item[1]) for item in cand]
        w_top_K.append(top_)
        cand = df.sort_values('close', ascending=False).iloc[:K][['ts_code', 'pct_chg']].to_numpy()
        top_ = ["%s%+.2f%%" % (code2name[item[0]], item[1]) for item in cand]
        top_K.append(top_)
        cand = df.sort_values('close', ascending=True)[['ts_code', 'pct_chg']].to_numpy()
        temp = []
        count = 0
        for item in cand:
            if item[0] in code2name:
                temp.append("%s%+.2f%%" %(code2name[item[0]], item[1]))
                count += 1
            if count >= K:
                break
        r_top_K.append(temp)
        #time.sleep(0.5)
    data['Date'] = trade_date_[-1::-1]
    data['Open'] = open_[-1::-1]
    data['High'] = high_[-1::-1]
    data['Low'] = low_[-1::-1]
    data['Close'] = close_[-1::-1]
    data['Volume'] = vol_[-1::-1]
    data['Amount'] = amount_[-1::-1]

    bb = ta.volatility.BollingerBands(close=data['Close'], n=20, ndev=2)
    data['BollHBand'] = bb.bollinger_hband()
    data['BollLBand'] = bb.bollinger_lband()
    data['BollMAvg'] = bb.bollinger_mavg()

    return data.iloc[20:], (top_K, r_top_K, w_top_K)

def plot_index(df, top_K, savefile):
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df = df.set_index('Date')
    mc = mpf.make_marketcolors(up='r', down='g', ohlc='white')
    style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)
    wconfig = dict()
    apdict = mpf.make_addplot(df[['BollHBand', 'BollLBand', 'BollMAvg']])
    mpf.plot(df, type='ohlc', volume=True, style=style, title='Stock A Index', return_width_config=wconfig, ylabel='Index', figscale=1.5, tight_layout=True, addplot=apdict, scale_width_adjustment=dict(lines=0.7))
    print(wconfig)
    plt.savefig(savefile)
    plt.close('all')
    today = datetime.date.today().strftime("%Y%m%d")
    trade_date = api.trade_cal(end_date=today, is_open='1')
    print('trade date: %s' % trade_date.iloc[-1]['cal_date'])
    print('open: %.2f' % df.iloc[-1]['Open'])
    print('high: %.2f' % df.iloc[-1]['High'])
    print('low: %.2f' % df.iloc[-1]['Low'])
    print('close: %.2f' % df.iloc[-1]['Close'])
    print('volume: %.2f万手' % df.iloc[-1]['Volume'])
    print('amount: %.2f亿' % df.iloc[-1]['Amount'])
    print('percent change: %+.2f%%' % ((df.iloc[-1]['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100.))
    print("权重占比前十: %s" % ' '.join(top_K[2][0]))
    print('指数占比前十: %s' % ' '.join(top_K[0][0]))
    print('指数占比倒数前十: %s' % ' '.join(top_K[1][0]))

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
    cols = ["sma%d" % d for d in days]
    for day, col in zip(days, cols):
        new_data[col] = ta.utils.sma(new_data.iloc[-1::-1].close, periods=day)[-1::-1]
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    # add exponential ma info
    # scaling = s / (1 + d), s is smoothing, typically 2, d is ma days
    # ema(t) = v * scaling + ema(t - 1) * (1 - scaling), v is time(t)'s price
    cols = ["ema%d" % d for d in days]
    for day, col in zip(days, cols):
        new_data[col] = ta.utils.ema(new_data.iloc[-1::-1].close, periods=day)[-1::-1]
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

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
    N = [2,3,4,5,6]
    cols = ["rsi%d" % n for n in N]
    for n, col in zip(N, cols):
        new_data[col] = ta.momentum.rsi(new_data.iloc[-1::-1].close, n=n)[-1::-1]
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

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

def add_long_crossover_info(data):
    # add 50-day 100-day crossover info, I think
    # it is not important for short-swing trading,
    # but sometimes it happens, just add this feature
    new_data = data.reset_index(drop=True)
    tracking_day = 'ema50'
    cross_day = ['ema100']
    cross_cols = ['longcross']
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

    precross_cols = ['pre_longcross']
    for cross_col, precross_col in zip(cross_cols, precross_cols):
        vals = new_data.iloc[1:][cross_col].tolist()
        vals.append(0)
        new_data[precross_col] = vals

    return new_data

def add_bollinger_band_info(data):
    new_data = data.reset_index(drop=True)
    #N = [20, 14, 12, 10, 5, 4, 3, 2]
    N = [20, 10]
    for n in N:
        bb = ta.volatility.BollingerBands(close=new_data.iloc[-1::-1].close, n=n, ndev=2)
        col = 'boll_hband%d' % n
        new_data[col] = bb.bollinger_hband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

        col = 'boll_lband%d' % n
        new_data[col] = bb.bollinger_lband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

        col = 'boll_hband_ind%d' % n
        new_data[col] = bb.bollinger_hband_indicator()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

        col = 'boll_lband_ind%d' % n
        new_data[col] = bb.bollinger_lband_indicator()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

        col = 'boll_mavg%d' % n
        new_data[col] = bb.bollinger_mavg()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

        col = 'boll_pband%d' % n
        new_data[col] = bb.bollinger_pband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

        col = 'boll_wband%d' % n
        new_data[col] = bb.bollinger_wband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data['pre_%s' % col] = temp

    return new_data

def add_obv_info(data):
    new_data = data.reset_index(drop=True)
    obv = ta.volume.OnBalanceVolumeIndicator(close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol)
    new_data['obv'] = obv.on_balance_volume()
    temp = new_data.iloc[1:]['obv'].tolist()
    temp.append(np.nan)
    new_data['pre_obv'] = temp

    return new_data

def add_adi_info(data):
    new_data = data.reset_index(drop=True)
    adi = ta.volume.AccDistIndexIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol)
    new_data['adi'] = adi.acc_dist_index()
    temp = new_data.iloc[1:]['adi'].tolist()
    temp.append(np.nan)
    new_data['pre_adi'] = temp

    return new_data

def add_cmf_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        cmf = ta.volume.ChaikinMoneyFlowIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol, n=day)
        col = "cmf%d" % day
        new_data[col] = cmf.chaikin_money_flow()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_fi_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        fi = ta.volume.ForceIndexIndicator(close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol, n=day)
        col = "fi%d" % day
        new_data[col] = fi.force_index()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_eom_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        eom = ta.volume.EaseOfMovementIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, volume=new_data.iloc[-1::-1].vol, n=day)
        col = "eom%d" % day
        new_data[col] = eom.ease_of_movement()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "sma_eom%d" % day
        new_data[col] = eom.sma_ease_of_movement()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_vpt_info(data):
    new_data = data.reset_index(drop=True)
    vpt = ta.volume.VolumePriceTrendIndicator(close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol)
    col = "vpt"
    new_data[col] = vpt.volume_price_trend()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_nvi_info(data):
    new_data = data.reset_index(drop=True)
    nvi = ta.volume.NegativeVolumeIndexIndicator(close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol)
    col = "nvi"
    new_data[col] = nvi.negative_volume_index()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_vwap_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        vwap = ta.volume.VolumeWeightedAveragePrice(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, volume=new_data.iloc[-1::-1].vol, close=new_data.iloc[-1::-1].close, n=day)
        col = "vwap%d" % day
        new_data[col] = vwap.volume_weighted_average_price()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_atr_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        atr = ta.volatility.AverageTrueRange(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "atr%d" % day
        new_data[col] = atr.average_true_range()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_kc_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        kc = ta.volatility.KeltnerChannel(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "kc_mband%d" % day
        new_data[col] = kc.keltner_channel_mband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "kc_pband%d" % day
        new_data[col] = kc.keltner_channel_pband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "kc_wband%d" % day
        new_data[col] = kc.keltner_channel_wband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_dc_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        dc = ta.volatility.DonchianChannel(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "dc_mband%d" % day
        new_data[col] = dc.donchian_channel_mband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "dc_pband%d" % day
        new_data[col] = dc.donchian_channel_pband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "dc_wband%d" % day
        new_data[col] = dc.donchian_channel_wband()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_macd_info(data):
    new_data = data.reset_index(drop=True)
    macd = ta.trend.MACD(close=new_data.iloc[-1::-1].close, n_slow=15, n_fast=5, n_sign=7)
    col = "macd"
    new_data[col] = macd.macd()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "macd_diff"
    new_data[col] = macd.macd_diff()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "macd_signal"
    new_data[col] = macd.macd_signal()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp
    return new_data

def add_adx_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30,14,10]
    for day in days:
        adx = ta.trend.ADXIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "adx%d" % day
        new_data[col] = adx.adx()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "adx_neg%d" % day
        new_data[col] = adx.adx_neg()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "adx_pos%d" % day
        new_data[col] = adx.adx_pos()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_vi_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        vi = ta.trend.VortexIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "vi_diff%d" % day
        new_data[col] = vi.vortex_indicator_diff()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "vi_neg%d" % day
        new_data[col] = vi.vortex_indicator_neg()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "vi_pos%d" % day
        new_data[col] = vi.vortex_indicator_pos()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_trix_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        trix = ta.trend.TRIXIndicator(close=new_data.iloc[-1::-1].close, n=day)
        col = "trix%d" % day
        new_data[col] = trix.trix()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_mi_info(data):
    new_data = data.reset_index(drop=True)
    mi = ta.trend.MassIndex(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low)
    col = "mi"
    new_data[col] = mi.mass_index()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_cci_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        cci = ta.trend.CCIIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "cci%d" % day
        new_data[col] = cci.cci()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_dpo_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        dpo = ta.trend.DPOIndicator(close=new_data.iloc[-1::-1].close, n=day)
        col = "dpo%d" % day
        new_data[col] = dpo.dpo()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_kst_info(data):
    new_data = data.reset_index(drop=True)
    kst = ta.trend.KSTIndicator(close=new_data.iloc[-1::-1].close)
    col = "kst"
    new_data[col] = kst.kst()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "kst_diff"
    new_data[col] = kst.kst_diff()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "kst_sig"
    new_data[col] = kst.kst_sig()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_ichimoku_info(data):
    new_data = data.reset_index(drop=True)
    ichimoku = ta.trend.IchimokuIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low)
    col = "ichimoku_a"
    new_data[col] = ichimoku.ichimoku_a()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "ichimoku_b"
    new_data[col] = ichimoku.ichimoku_b()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "ichimoku_base"
    new_data[col] = ichimoku.ichimoku_base_line()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "ichimoku_conv"
    new_data[col] = ichimoku.ichimoku_conversion_line()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_psar_info(data):
    new_data = data.reset_index(drop=True)
    psar = ta.trend.PSARIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close)
    col = "psar"
    new_data[col] = psar.psar()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "psar_down"
    new_data[col] = psar.psar_down()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "psar_down_idc"
    new_data[col] = psar.psar_down_indicator()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "psar_up"
    new_data[col] = psar.psar_up()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    col = "psar_up_idc"
    new_data[col] = psar.psar_up_indicator()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_tsi_info(data):
    new_data = data.reset_index(drop=True)
    tsi = ta.momentum.TSIIndicator(close=new_data.iloc[-1::-1].close)
    col = "tsi"
    new_data[col] = tsi.tsi()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_uo_info(data):
    new_data = data.reset_index(drop=True)
    uo = ta.momentum.UltimateOscillator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close)
    col = "uo"
    new_data[col] = uo.uo()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_so_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        so = ta.momentum.StochasticOscillator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, n=day)
        col = "stoch%d" % day
        new_data[col] = so.stoch()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

        col = "stoch_signal%d" % day
        new_data[col] = so.stoch_signal()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_wr_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        wr = ta.momentum.WilliamsRIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, lbp=day)
        col = "wr%d" % day
        new_data[col] = wr.wr()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_ao_info(data):
    new_data = data.reset_index(drop=True)
    ao = ta.momentum.AwesomeOscillatorIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low)
    col = "ao"
    new_data[col] = ao.ao()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_kama_info(data):
    new_data = data.reset_index(drop=True)
    kama = ta.momentum.KAMAIndicator(close=new_data.iloc[-1::-1].close)
    col = "kama"
    new_data[col] = kama.kama()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_roc_info(data):
    new_data = data.reset_index(drop=True)
    days = [2,5,10,15,20,30]
    for day in days:
        roc = ta.momentum.ROCIndicator(close=new_data.iloc[-1::-1].close, n=day)
        col = "roc%d" % day
        new_data[col] = roc.roc()
        temp = new_data.iloc[1:][col].tolist()
        temp.append(np.nan)
        new_data["pre_%s" % col] = temp

    return new_data

def add_dr_info(data):
    new_data = data.reset_index(drop=True)
    dr = ta.others.DailyReturnIndicator(close=new_data.iloc[-1::-1].close)
    col = "dr"
    new_data[col] = dr.daily_return()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_dlr_info(data):
    new_data = data.reset_index(drop=True)
    dlr = ta.others.DailyLogReturnIndicator(close=new_data.iloc[-1::-1].close)
    col = "dlr"
    new_data[col] = dlr.daily_log_return()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_cr_info(data):
    new_data = data.reset_index(drop=True)
    cr = ta.others.CumulativeReturnIndicator(close=new_data.iloc[-1::-1].close)
    col = "cr"
    new_data[col] = cr.cumulative_return()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_mfi_info(data):
    new_data = data.reset_index(drop=True)
    mfi = ta.volume.MFIIndicator(high=new_data.iloc[-1::-1].high, low=new_data.iloc[-1::-1].low, close=new_data.iloc[-1::-1].close, volume=new_data.iloc[-1::-1].vol, n=5)
    col = "mfi"
    new_data[col] = mfi.money_flow_index()
    temp = new_data.iloc[1:][col].tolist()
    temp.append(np.nan)
    new_data["pre_%s" % col] = temp

    return new_data

def add_features(data):
    new_data = data.reset_index(drop=True)
    # previous day info
    #new_data = add_preday_info(new_data)
    # moving average info
    new_data = add_ma_info(new_data)
    # rsi info
    new_data = add_rsi_info(new_data)
    # crossover of moving average
    #new_data = add_crossover_info(new_data)
    # long crossover of moving average
    #new_data = add_long_crossover_info(new_data)
    # bollinger bands
    new_data = add_bollinger_band_info(new_data)
    # on-balance volume
    #new_data = add_obv_info(new_data)
    # accumulation/distribution index
    #new_data = add_adi_info(new_data)
    # chaikin money flow
    #new_data = add_cmf_info(new_data)
    # force index
    #new_data = add_fi_info(new_data)
    # ease of movement
    #new_data = add_eom_info(new_data)
    # volume price trend
    #new_data = add_vpt_info(new_data)
    # negative volume index
    #new_data = add_nvi_info(new_data)
    # volume weighted average price
    #new_data = add_vwap_info(new_data)
    # average true range
    #new_data = add_atr_info(new_data)
    # keltner channel
    #new_data = add_kc_info(new_data)
    # donchian channel
    #new_data = add_dc_info(new_data)
    # moving average convergence divergence
    #new_data = add_macd_info(new_data)
    # average directional movement index
    new_data = add_adx_info(new_data)
    # vortex indicator
    #new_data = add_vi_info(new_data)
    # trix indicator
    #new_data = add_trix_info(new_data)
    # mass index
    #new_data = add_mi_info(new_data)
    # commodity channel index
    #new_data = add_cci_info(new_data)
    # detrended price oscillator
    #new_data = add_dpo_info(new_data)
    # kst oscillator
    #new_data = add_kst_info(new_data)
    # ichimoku kinko hyo
    #new_data = add_ichimoku_info(new_data)
    # parabolic stop and reverse
    new_data = add_psar_info(new_data)
    # true strength index
    #new_data = add_tsi_info(new_data)
    # ultimate oscillator
    #new_data = add_uo_info(new_data)
    # stochastic oscillator
    #new_data = add_so_info(new_data)
    # williams %R
    #new_data = add_wr_info(new_data)
    # awesome oscillator
    #new_data = add_ao_info(new_data)
    # kaufman's adaptive moving average
    #new_data = add_kama_info(new_data)
    # rate of change
    #new_data = add_roc_info(new_data)
    # daily return
    #new_data = add_dr_info(new_data)
    # daily log return
    #new_data = add_dlr_info(new_data)
    # cumulative return
    #new_data = add_cr_info(new_data)
    # money flow index
    #new_data = add_mfi_info(new_data)

    return new_data

def plot_data(data, days, close, cols, filename, stock):
    x = [i for i in range(days)]
    count = 0
    plt.figure()
    fig, ax = plt.subplots(len(cols), figsize=[6.4 * 3, 4 * len(cols)])
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for col in cols:
        if 'ema' in col or 'boll_band' in col:
            vals2 = data.iloc[0:days].iloc[-1::-1][close].to_numpy()
            vals3 = data.iloc[0:days].iloc[-1::-1]['ema5'].to_numpy()
            sns.lineplot(x=x, y=vals3, ax=ax[count])
            sns.lineplot(x=x, y=vals2, ax=ax[count])
        elif 'vol' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['vol'].to_numpy()
            sns.lineplot(x=x, y=vals, ax=ax[count])
        elif 'gap' in col:
            vals2 = data.iloc[0:days].iloc[-1::-1][close].to_numpy()
            vals2 = StandardScaler().fit_transform(vals2.reshape(-1, 1)).flatten()
            sns.lineplot(x=x, y=vals2, ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['vol'].to_numpy()
            vals = StandardScaler().fit_transform(vals.reshape(-1, 1)).flatten()
            sns.lineplot(x=x, y=vals, ax=ax[count])
            max_ = max([np.amax(vals2), np.amax(vals)])
            min_ = min([np.amin(vals2), np.amin(vals)])
        else:
            vals1 = data.iloc[0:days].iloc[-1::-1][col].to_numpy()
            vals1 = StandardScaler().fit_transform(vals1.reshape(-1,1)).flatten()
            max_ = np.amax(vals1)
            min_ = np.amin(vals1)
            vals2 = data.iloc[0:days].iloc[-1::-1][close].to_numpy()
            vals3 = data.iloc[0:days].iloc[-1::-1]['ema5'].to_numpy()
            sns.lineplot(x=x, y=vals1, ax=ax[count])
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals2.reshape(-1,1)).flatten(), ax=ax[count])
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals3.reshape(-1,1)).flatten(), ax=ax[count])

        if 'cmf' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['adi'].to_numpy()
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals.reshape(-1,1)).flatten(), ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'adi'], loc='upper left')
        elif 'macd' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['macd_signal'].to_numpy()
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals.reshape(-1,1)).flatten(), ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'macd_signal'], loc='upper left')
        elif 'adx' in col:
            day = col.replace('adx', '')
            pos_ = data.iloc[0:days].iloc[-1::-1]['adx_pos%s' % day].to_numpy()
            pos_ = StandardScaler().fit_transform(pos_.reshape(-1,1)).flatten()
            max_ = np.amax(pos_)
            sns.lineplot(x=x, y=pos_, ax=ax[count])
            neg_ = data.iloc[0:days].iloc[-1::-1]['adx_neg%s' % day].to_numpy()
            neg_ = StandardScaler().fit_transform(neg_.reshape(-1,1)).flatten()
            min_ = np.amin(neg_)
            sns.lineplot(x=x, y=neg_, ax=ax[count])
            # scatter plot
            y = [min_]
            for i in range(1, days):
                if pos_[i] > neg_[i] and pos_[i-1] < neg_[i-1]:
                    y.append(max_)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            adx_ = data.iloc[0:days].iloc[-1::-1][col].to_numpy()
            adx_max = np.amax(adx_)
            adx_min = np.amin(adx_)
            adx_ = [(x - adx_min) / (adx_max - adx_min) for x in adx_]
            y = []
            for i in range(days):
                if adx_[i] > 0.8:
                    y.append(min_ + (max_ - min_) / 4. * 3)
                elif adx_[i] < 0.1:
                    y.append(min_ + (max_ - min_) / 4. * 1)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend([col, close, 'ema5', '+DMI', '-DMI', 'buy', 'signal'], loc='upper left')
        elif 'vi_diff' in col:
            day = col.replace('vi_diff', '')
            pos_ = data.iloc[0:days].iloc[-1::-1]['vi_pos%s' % day].to_numpy()
            pos_ = StandardScaler().fit_transform(neg_.reshape(-1,1)).flatten()
            sns.lineplot(x=x, y=pos_, ax=ax[count])
            neg_ = data.iloc[0:days].iloc[-1::-1]['vi_neg%s' % day].to_numpy()
            neg_ = StandardScaler().fit_transform(neg_.reshape(-1,1)).flatten()
            sns.lineplot(x=x, y=neg_, ax=ax[count])
            ax[count].legend([col, close, 'ema5', '+VI', '-VI'], loc='upper left')
        elif 'kst' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['kst_diff'].to_numpy()
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals.reshape(-1,1)).flatten(), ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['kst_sig'].to_numpy()
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals.reshape(-1,1)).flatten(), ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'kst_diff', 'kst_sig'], loc='upper left')
        elif 'stoch' in col:
            day = col.replace('stoch', '')
            vals = data.iloc[0:days].iloc[-1::-1]['stoch_signal%s' % day].to_numpy()
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals.reshape(-1,1)).flatten(), ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'stoch_signal'], loc='upper left')
        elif 'boll_wband' in col:
            day = col.replace('boll_wband', '')
            vals = data.iloc[0:days].iloc[-1::-1]['boll_mavg%s' % day].to_numpy()
            sns.lineplot(x=x, y=StandardScaler().fit_transform(vals.reshape(-1,1)).flatten(), ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'boll_mavg'], loc='upper left')
        elif 'boll_band' in col:
            day = col.replace('boll_band', '')
            vals = data.iloc[0:days].iloc[-1::-1]['boll_mavg%s' % day].to_numpy()
            sns.lineplot(x=x, y=vals, ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['boll_hband%s' % day].to_numpy()
            max_ = np.amax(vals)
            sns.lineplot(x=x, y=vals, ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['boll_lband%s' % day].to_numpy()
            min_ = np.amin(vals)
            sns.lineplot(x=x, y=vals, ax=ax[count])
            # gap
            close_ = data.iloc[0:days].iloc[-1::-1]['close'].to_numpy()
            high_ = data.iloc[0:days].iloc[-1::-1]['high'].to_numpy()
            low_ = data.iloc[0:days].iloc[-1::-1]['low'].to_numpy()
            # up gap
            y = []
            for i in range(days):
                if (high_[i] - close_[i]) / (high_[i] - low_[i]) < 0.05:
                    y.append(max_)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend(['ema5', close, 'boll_mavg', 'boll_hband', 'boll_lband', 'gap'], loc='upper left')
        elif 'boll_pband' in col:
            day = col.replace('boll_pband', '')
            vals = data.iloc[0:days].iloc[-1::-1]['boll_pband%s' % day].to_numpy()
            y = []
            for i in range(days):
                if vals[i] > 0.8:
                    y.append(max_)
                elif vals[i] < 0.1:
                    y.append((max_ + min_) / 2.)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'buy'], loc='upper left')
        elif 'rsi' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['rsi2'].to_numpy()
            y = []
            for v in vals:
                if v > 80.:
                    y.append(max_)
                elif v < 10.:
                    y.append((max_ + min_) / 2.)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'buy'], loc='upper left')
        elif 'mfi' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['mfi'].to_numpy()
            print("mfi:", vals)
            y = []
            for v in vals:
                if v > 80.:
                    y.append(1.)
                elif v < 20.:
                    y.append(0.)
                else:
                    y.append(-1.)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend([col, close, 'ema5', 'buy'], loc='upper left')
        elif 'ema' in col:
            vals = data.iloc[0:days].iloc[-1::-1]['ema10'].to_numpy()
            sns.lineplot(x=x, y=vals, ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['ema15'].to_numpy()
            sns.lineplot(x=x, y=vals, ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['ema20'].to_numpy()
            sns.lineplot(x=x, y=vals, ax=ax[count])
            vals = data.iloc[0:days].iloc[-1::-1]['ema30'].to_numpy()
            sns.lineplot(x=x, y=vals, ax=ax[count])
            # scatter break through
            period = 10
            ema_ = data.iloc[0:days].iloc[-1::-1]['ema%d' % period].to_numpy()
            close_ = data.iloc[0:days].iloc[-1::-1]['close'].to_numpy()
            min_ = np.amin(close_)
            max_ = np.amax(close_)
            y = [min_] * period
            for i in range(period, days):
                appended = False
                if close_[i] > ema_[i] and close_[i-1] < ema_[i-1]:
                    y.append(max_)
                    appended = True
                if not appended:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend(['ema5', close, 'ema10', 'ema15', 'ema20', 'ema30', 'break'], loc='upper left')
        elif 'vol' in col:
            vol_ = data.iloc[0:days].iloc[-1::-1]['vol'].to_numpy()
            close_ = data.iloc[0:days].iloc[-1::-1]['close'].to_numpy()
            max_ = np.amax(vol_)
            min_ = np.amin(vol_)
            y = [min_]
            for i in range(days - 1):
                if close_[i] > close_[i+1] and vol_[i] < vol_[i+1]:
                    y.append(max_)
                elif close_[i] < close_[i+1] and vol_[i] > vol_[i+1]:
                    y.append((max_ + min_) / 2.)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend(['vol', 'buy'], loc='upper left')
        elif 'gap' in col:
            close_ = data.iloc[0:days].iloc[-1::-1]['close'].to_numpy()
            high_ = data.iloc[0:days].iloc[-1::-1]['high'].to_numpy()
            low_ = data.iloc[0:days].iloc[-1::-1]['low'].to_numpy()
            # up gap
            y = []
            for i in range(days):
                if (high_[i] - close_[i]) / (high_[i] - low_[i]) < 0.05:
                    y.append(max_)
                elif (close_[i] - low_[i]) / (high_[i] - low_[i]) < 0.05:
                    y.append((max_ + min_) / 2.)
                else:
                    y.append(min_)
            sns.scatterplot(x=x, y=y, ax=ax[count])
            ax[count].legend(['close', 'vol', 'gap'], loc='upper left')
        else:
            ax[count].legend([col, close, 'ema5'], loc='upper left')
        count += 1
    fig.suptitle(stock, fontsize=40, fontweight='normal')
    plt.savefig(filename)
    plt.close('all')

def clear_directory(dirname):
    files = glob.glob("%s/*" % dirname)
    for fname in files:
        os.remove(fname)

def send_mail(mail_to):
    clear_directory('mail')
    cmd = 'zip -r mail/stock_png.zip pattern'
    subprocess.call(cmd.split())

    msg_from = "1285470650@qq.com"
    passwd = "vjxmrjfhpqerbaae"
    msg_to = mail_to
    print("sending mail to %s ..." % msg_to)

    today = datetime.datetime.today().strftime("%Y-%m-%d")
    msg = MIMEMultipart()
    msg['Subject'] = "%s stock trading plottings" % today
    msg['From'] = msg_from
    msg['To'] = msg_to

    msg.attach(MIMEText(trading_note))

    files = glob.glob("mail/stock_png.*")
    for fname in files:
        with open(fname, 'rb') as fp:
            data = fp.read()
            msg.attach(MIMEApplication(data, Name='pattern.zip'))

    try:
        s = smtplib.SMTP('smtp.qq.com', 587)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("sending mail success!")
    except smtplib.SMTPException as e:
        print("sending failed:", e)
    finally:
        s.quit()

'''
 过滤机制：
    1. 五日内psar与close越来越近，close有上穿趋势
'''
def filter_by_strategy1(data, days):
    # normalize close
    temp_close = data.iloc[0:days].iloc[-1::-1]['close'].to_numpy()
    close = StandardScaler().fit_transform(temp_close.reshape(-1, 1)).flatten()

    flag = False
    temp = data.iloc[0:days].iloc[-1::-1]['psar'].to_numpy()
    psar = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    close_ = close[-1:-6:-1]
    psar_ = psar[-1:-6:-1]
    # 1.当前psar大于close，五日psar与close递增取均值，当前差加上递增均值是否大于0，如果大于0，则可能上穿
    if psar_[0] > close_[0]:
        dist = close_ - psar_
        sum_ = 0.
        for i in range(4):
            sum_ += dist[i] - dist[i + 1]
        avg = sum_ / 4.
        if dist[0] + avg > 0.:
            flag = True
    # 2.当前psar小于close，是否是五日内发生的上穿现象，如果是，则趋势可能仍在
    if psar_[0] < close_[0] and psar_[-1] > close_[-1]:
        flag = True
    if not flag:
        return False

    # 取五日adx_pos, adx_neg均值做为POS和NEG，取五日的距离，相比前一天差值变小大于1天，则
    # 即使psar合格但是adx趋势还没有走到
    temp_pos = data.iloc[0:days].iloc[-1::-1]['adx_pos14'].to_numpy()
    pos = StandardScaler().fit_transform(temp_pos.reshape(-1, 1)).flatten()
    pos = pos[-1::-1]
    temp_neg = data.iloc[0:days].iloc[-1::-1]['adx_neg14'].to_numpy()
    neg = StandardScaler().fit_transform(temp_neg.reshape(-1, 1)).flatten()
    neg = neg[-1::-1]
    # 取五日均值做平滑
    new_pos = []
    new_neg = []
    for i in range(5):
        pos_ = np.sum(pos[i : i + 5]) / 5.
        neg_ = np.sum(neg[i : i + 5]) / 5.
        new_pos.append(pos_)
        new_neg.append(neg_)

    dist = np.array(new_pos) - np.array(new_neg)
    count = 0
    for i in range(4):
        if dist[i] < dist[i + 1]:
            count += 1
        if count > 1:
            return False

    return True
'''
 1. 用2周期RSI过滤，看买点
'''
def filter_by_strategy2(data, days):
    flag = False
    rsi2 = data.iloc[0:days]['rsi2'].to_numpy()
    max_ = np.amax(rsi2)
    min_ = np.amin(rsi2)
    if (data.iloc[0]['rsi2'] - min_) / (max_ - min_) < 0.1:
        flag = True
    if not flag:
        return flag

    return True

'''
 1. 取15日vwap 做整体趋势控制，再考虑kama和psar上穿情况
'''
def filter_by_strategy3(data, days):
    # filter price < 5.0.
    if data.iloc[0]['close'] < 5.:
        return False

    # filter volume < 200000.
    if data.iloc[0]['vol'] < 200000.:
        return False

    # 1. boll_wband < 0.3
    boll_wd = data.iloc[0:days]['boll_wband20'].to_numpy()
    boll_max = np.amax(boll_wd)
    boll_min = np.amin(boll_wd)
    if (boll_wd[0] - boll_min) / (boll_max - boll_min) > 0.08:
        return False

    # normalize close
    temp_close = data.iloc[0:days].iloc[-1::-1]['ema5'].to_numpy()
    close = StandardScaler().fit_transform(temp_close.reshape(-1, 1)).flatten()

    vwap_flag = False
    temp = data.iloc[0:days].iloc[-1::-1]['vwap30'].to_numpy()
    vwap = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    close_ = close[-1:-6:-1]
    vwap_ = vwap[-1:-6:-1]
    # 1.当前vwap大于close，五日psar与close递增取均值，当前差加上递增均值是否大于0，如果大于0，则可能上穿
    if vwap_[0] > close_[0]:
        dist = close_ - vwap_
        sum_ = 0.
        for i in range(4):
            sum_ += dist[i] - dist[i + 1]
        avg = sum_ / 4.
        if dist[0] + avg > 0.:
            vwap_flag = True
    # 2.当前vwap小于close，是否是五日内发生的上穿现象，如果是，则趋势可能仍在
    if vwap_[0] < close_[0] and vwap_[-1] > close_[-1]:
        vwap_flag = True

    psar_flag = False
    temp = data.iloc[0:days].iloc[-1::-1]['psar'].to_numpy()
    psar = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    close_ = close[-1:-6:-1]
    psar_ = psar[-1:-6:-1]
    # 1.当前psar大于close，五日psar与close递增取均值，当前差加上递增均值是否大于0，如果大于0，则可能上穿
    if psar_[0] > close_[0]:
        dist = close_ - psar_
        sum_ = 0.
        for i in range(4):
            sum_ += dist[i] - dist[i + 1]
        avg = sum_ / 4.
        if dist[0] + avg > 0.:
            psar_flag = True
    # 2.当前psar小于close，是否是五日内发生的上穿现象，如果是，则趋势可能仍在
    if psar_[0] < close_[0] and psar_[-1] > close_[-1]:
        psar_flag = True

    kama_flag = False
    temp = data.iloc[0:days].iloc[-1::-1]['kama'].to_numpy()
    kama = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    close_ = close[-1:-6:-1]
    kama_ = kama[-1:-6:-1]
    # 1.当前kama大于close，五日kama与close递增取均值，当前差加上递增均值是否大于0，如果大于0，则可能上穿
    if kama_[0] > close_[0]:
        dist = close_ - kama_
        sum_ = 0.
        for i in range(4):
            sum_ += dist[i] - dist[i + 1]
        avg = sum_ / 4.
        if dist[0] + avg > 0.:
            kama_flag = True
    # 2.当前kama小于close，是否是五日内发生的上穿现象，如果是，则趋势可能仍在
    if kama_[0] < close_[0] and kama_[-1] > close_[-1]:
        kama_flag = True

    if (not psar_flag) and (not kama_flag) and (not vwap_flag):
        return False

    # 1. 5日内close上穿boll_mavg20
    boll_flag = False
    temp = data.iloc[0:days].iloc[-1::-1]['boll_mavg20'].to_numpy()
    boll_mavg = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    close_ = close[-1:-6:-1]
    boll_mavg_ = boll_mavg[-1:-6:-1]
    if boll_mavg_[0] < close_[0] and boll_mavg_[-1] > close_[-1]:
        boll_flag = True
    if not boll_flag:
        return False

    return True

'''
 1. 取adx相交做为buy signal
'''
def filter_by_strategy4(data, days):
    # filter by ADX
    adx_flag = False
    adx_pos = data.iloc[0:days]['adx_pos14'].to_numpy()
    adx_pos = StandardScaler().fit_transform(adx_pos.reshape(-1, 1)).flatten()
    adx_neg = data.iloc[0:days]['adx_neg14'].to_numpy()
    adx_neg = StandardScaler().fit_transform(adx_neg.reshape(-1, 1)).flatten()
    if adx_pos[0] > adx_neg[0] and adx_pos[1] < adx_neg[1]:
        adx_flag = True
    if not adx_flag:
        return False

    return True

'''
 1. 取pband < 0.1 做为buy signal
'''
def filter_by_strategy5(data, days):
    # filter by boll_pband20
    pband = data.iloc[0:days]['boll_pband20'].to_numpy()
    if pband[0] > 0.1:
        return False

    return True

'''
 1. 取adx < 0.1 做为buy signal
'''
def filter_by_strategy6(data, days):
    # filter by adx14
    adx = data.iloc[0:days]['adx14'].to_numpy()
    max_ = np.amax(adx)
    min_ = np.amin(adx)
    if (adx[0] - min_) / (max_ - min_) > 0.1:
        return False

    # filter by price, don't pick percent of price is high
    price = data.iloc[0:days]['close'].to_numpy()
    max_ = np.amax(price)
    min_ = np.amin(price)
    if (price[0] - min_) / (max_ - min_) > 0.1:
        return False

    return True

'''
 1. 取adx < 0.1 和 psar上穿做为buy signal
'''
def filter_by_strategy7(data, days):
    # filter by adx14
    adx = data.iloc[0:days]['adx14'].to_numpy()
    max_ = np.amax(adx)
    min_ = np.amin(adx)
    if (adx[0] - min_) / (max_ - min_) > 0.1:
        return False

    # filter by psar
    psar_flag = False
    psar = data.iloc[0:days]['psar'].to_numpy()
    psar = StandardScaler().fit_transform(psar.reshape(-1, 1)).flatten()
    close = data.iloc[0:days]['close'].to_numpy()
    close = StandardScaler().fit_transform(close.reshape(-1, 1)).flatten()
    if close[0] > psar[0] and close[1] < psar[1]:
        psar_flag = True
    if not psar_flag:
        return False

    return True

'''
 1. 取adx < 0.1 and pband < 0.1做为buy signal
'''
def filter_by_strategy8(data, days):
    # filter by adx14
    adx = data.iloc[0:days]['adx14'].to_numpy()
    max_ = np.amax(adx)
    min_ = np.amin(adx)
    if (adx[0] - min_) / (max_ - min_) > 0.1:
        return False

    # filter by boll_pband20
    pband = data.iloc[0:days]['boll_pband20'].to_numpy()
    if pband[0] > 0.1:
        return False

    return True

'''
 1. 取pband < 0.1 and vol < 0.1做为buy signal (短)
'''
def filter_by_strategy9(data, days):
    # filter by vol
    vol = data.iloc[0:days]['vol'].to_numpy()
    max_ = np.amax(vol)
    min_ = np.amin(vol)
    if (vol[0] - min_) / (max_ - min_) > 0.1:
        return False

    # filter by boll_pband20
    pband = data.iloc[0:days]['boll_pband20'].to_numpy()
    if pband[0] > 0.1:
        return False

    return True

'''
 1. 取adx < 0.1 and vol < 0.1做为buy signal (中)
'''
def filter_by_strategy10(data, days):
    # filter by vol
    vol = data.iloc[0:days]['vol'].to_numpy()
    max_ = np.amax(vol)
    min_ = np.amin(vol)
    if (vol[0] - min_) / (max_ - min_) > 0.1:
        return False

    # filter by adx14
    adx = data.iloc[0:days]['adx14'].to_numpy()
    max_ = np.amax(adx)
    min_ = np.amin(adx)
    if (adx[0] - min_) / (max_ - min_) > 0.1:
        return False

    return True

'''
 1. 取价跌、量涨为信号
'''
def filter_by_strategy11(data, days):
    # filter by vol
    vol = data.iloc[0:days]['vol'].to_numpy()
    close = data.iloc[0:days]['close'].to_numpy()
    if not (vol[0] > vol[1] and close[0] < close[1]):
        return False

    # filter by adx14
    adx = data.iloc[0:days]['adx14'].to_numpy()
    max_ = np.amax(adx)
    min_ = np.amin(adx)
    if (adx[0] - min_) / (max_ - min_) > 0.1:
        return False

    return True

'''
 1. 取价涨、量跌为信号
'''
def filter_by_strategy12(data, days):
    # filter by vol
    vol = data.iloc[0:days]['vol'].to_numpy()
    close = data.iloc[0:days]['close'].to_numpy()
    if not (vol[0] < vol[1] and close[0] > close[1]):
        return False

    # close 太高没有必要
    max_ = np.amax(close)
    min_ = np.amin(close)
    if (close[0] - min_) / (max_ - min_) > 0.5:
        return False

    # normalize close
    temp_close = data.iloc[0:days]['close'].to_numpy()
    close_ = StandardScaler().fit_transform(temp_close.reshape(-1, 1)).flatten()
    temp = data.iloc[0:days]['psar'].to_numpy()
    psar_ = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    if close_[0] < psar_[0]:
        return False

    return True

'''
 1. 取价涨、量跌为信号
'''
def filter_by_strategy13(data, days):
    # filter by vol
    vol = data.iloc[0:days]['vol'].to_numpy()
    close = data.iloc[0:days]['close'].to_numpy()
    if not (vol[0] < vol[1] and close[0] > close[1]):
        return False

    # close 太高没有必要
    max_ = np.amax(close)
    min_ = np.amin(close)
    if (close[0] - min_) / (max_ - min_) > 0.5:
        return False

    # normalize close
    temp_close = data.iloc[0:days]['close'].to_numpy()
    close_ = StandardScaler().fit_transform(temp_close.reshape(-1, 1)).flatten()
    temp = data.iloc[0:days]['psar'].to_numpy()
    psar_ = StandardScaler().fit_transform(temp.reshape(-1, 1)).flatten()
    if close_[0] > psar_[0]:
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str)
    parser.add_argument('--mail_only', action='store_true')
    parser.add_argument('--mail', type=str, default='1285470650@qq.com')
    parser.add_argument('--not_filter', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--plot_index', action='store_true')
    parser.add_argument('--weekly', action='store_true')
    parser.set_defaults(mail_only=False)
    parser.set_defaults(not_filter=False)
    parser.set_defaults(random=False)
    parser.set_defaults(plot_index=False)
    parser.set_defaults(weekly=False)
    args = parser.parse_args()
    print(args)

    if not args.mail_only:
        # 获取代码名称映射
        get_code_name_map()
        # 100天 plot
        days = 100

        clear_directory("pattern")

        # plot index
        if args.plot_index:
            data, top_K = calculate_index(days, 10)
            plot_index(data, top_K, 'pattern/chives_index.png')
            sys.exit(0)

        if args.stock:
            candidates = args.stock.split(',')
        else:
            candidates = get_stock_candidates()
        print("candidates: %s" % len(candidates))
        if args.random:
            random.shuffle(candidates)

        count = 1
        limit = 0
        for cand in candidates:
            print("index %d of %d" % (count, len(candidates)))
            print("getting data for %s" % cand)
            try:
                if cand in stock_index:
                    data = get_index_data(cand, args.weekly)
                else:
                    data = get_stock_data(cand, args.weekly)
                data = data.dropna(axis=0)
                data = add_features(data)
                if cand not in stock_index and not args.not_filter and not filter_by_strategy6(data, days):
                    print("filter %s by strategy!!!" % cand)
                    continue
                png = "pattern/%s.png" % cand
                print("plotting picture for %s" % cand)
                plot_data(data, days, 'close', ['ema5', 'vol', 'psar', 'adx14', 'boll_pband20', 'boll_band20'], png, cand)
                print("="*20, "DONE", "="*20)
                limit += 1
                if args.limit != -1 and limit >= args.limit:
                    break
            except Exception as e:
                print("skip %s for exception!!! (%s)" % (cand, e))
                continue
                #raise
            finally:
                count += 1

        #send_mail(args.mail)
    else:
        send_mail(args.mail)
