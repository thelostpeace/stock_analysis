# stock_analysis

**请不要用我的token，注册tushare-pro可以获取免费token：[tushare-pro](https://tushare.pro/register?reg=393010)**

 - 2020-09-21 update: add ema info, both **sma(simple moving average)** and **ema(exponential moving average)**. it seems that ema is more sensitive than sma for ema weighting more on recent prices. for short-swing trading, you should focus on short period ema crossover, because the crossovers tell you the trends. especially shorter period ema breaking through longer period ema upward is a strong signal that the price of this stock is going up and it is time to hold this stock and wait for a right time to sell. but the buy time is earlier than this crossover. and it seems that 100-day ema is very important, maybe because 100 days is near the market cycle? the feature importance just implies this conclusion.

 ```
 day1 percent change feature importance: ema20 => ema15 => ema100 => ema30 => pct_chg => sma15 => sma30 => change => ema50 => open => sma100 => amount => ema10 => sma20 => sma5 => sma200 => sma50 => ema200 => low => sma10 => vol => close => high => ema5
 day2 percent change feature importance: sma30 => ema100 => sma100 => pct_chg => sma50 => ema200 => ema15 => sma200 => ema20 => vol => change => sma15 => ema50 => sma20 => sma10 => amount => ema10 => ema30 => sma5 => open => ema5 => high => close => low
 day3 percent change feature importance: ema100 => sma20 => sma5 => ema200 => sma10 => sma200 => ema30 => sma15 => sma100 => sma50 => pct_chg => ema15 => ema50 => high => ema10 => vol => close => ema5 => change => open => low => sma30 => amount => ema20
 day4 percent change feature importance: ema5 => ema15 => sma20 => sma5 => ema20 => ema200 => sma200 => pct_chg => sma15 => amount => sma100 => sma30 => high => sma50 => change => ema100 => open => ema30 => ema50 => ema10 => close => vol => sma10 => low
 day5 percent change feature importance: ema10 => low => ema15 => ema100 => ema5 => sma20 => ema20 => sma100 => ema30 => sma200 => sma30 => sma15 => pct_chg => sma10 => change => sma5 => close => ema200 => high => sma50 => vol => amount => open => ema50
 ```

 - 2020-09-21 update: add rsi info, it seems that ema is more importance than rsi. but 2-period rsi is a good buy time indicator. 
 - 2020-09-21 conclusion: when 2-period rsi is under 10, near 0 is better, then buy. and if 5-day ema is over 15-day ema then hold, or sell it for short-swing trading. when you think it's too high, typically 2-period rsi is over 90, and 5-day ema is turning down, it indicates that the up trend is over, never wait for the best sell time.

 - 2020-09-24 update: crossover is very strong signal for short-swing trading. especially 5-day cross 100-day or 50-day, in my current experience, 5-day and 15-day crossover is also useful signal for trading.

```
day1 percent change feature importance: ema20 => cross5-100 => cross5-50 => ema5 => ema30 => pct_chg => ema100 => ema15 => change => ema50 => sma100 => cross5-30 => amount => rsi5 => sma200 => sma10 => sma5 => low => ema10 => rsi4 => rsi2 => sma20 => rsi6 => sma50 => sma15 => sma30 => ema200 => open => high => rsi3 => vol => cross5-200 => close => cross5-10 => cross5-15 => cross5-20
day2 percent change feature importance: ema30 => sma50 => sma30 => ema15 => ema5 => cross5-10 => ema200 => pct_chg => change => amount => low => sma10 => ema20 => ema100 => rsi6 => sma5 => sma20 => sma200 => vol => rsi5 => rsi3 => rsi4 => ema50 => sma100 => sma15 => rsi2 => cross5-20 => ema10 => cross5-15 => open => close => high => cross5-100 => cross5-200 => cross5-50 => cross5-30
day3 percent change feature importance: sma100 => sma10 => sma50 => ema5 => sma15 => ema30 => ema10 => ema200 => ema15 => cross5-30 => sma200 => rsi6 => rsi5 => pct_chg => change => sma30 => ema20 => sma5 => ema100 => ema50 => vol => sma20 => amount => cross5-10 => rsi3 => rsi2 => cross5-100 => close => cross5-20 => low => rsi4 => high => open => cross5-15 => cross5-50 => cross5-200
day4 percent change feature importance: ema50 => sma10 => sma30 => amount => pct_chg => ema15 => sma50 => sma15 => cross5-15 => ema30 => sma200 => sma100 => rsi6 => rsi2 => high => ema200 => cross5-10 => sma20 => ema100 => sma5 => ema5 => rsi5 => rsi3 => open => rsi4 => ema10 => low => close => ema20 => vol => cross5-30 => change => cross5-100 => cross5-50 => cross5-20 => cross5-200
day5 percent change feature importance: sma50 => ema15 => ema10 => sma15 => ema20 => pct_chg => amount => sma20 => change => sma200 => ema5 => ema50 => low => sma10 => ema200 => cross5-30 => cross5-15 => open => sma100 => rsi2 => ema30 => sma30 => ema100 => vol => rsi5 => rsi3 => rsi6 => sma5 => close => cross5-10 => rsi4 => cross5-200 => high => cross5-20 => cross5-50 => cross5-100
```

 - 2020-09-28 update: add a lot of indicators, but **vwap30** is the only useful indicator for trading. it seems that when price crosses vwap30 and is greater than vwap30, it is time to buy. and I think vwap30 is better than rsi2.
 - 2020-09-29 update: add a lot of indicators.

 - 2020-09-30 update: 所有更新已经完毕了，国庆快乐！

 - 2020-10-09 update: make this project private
