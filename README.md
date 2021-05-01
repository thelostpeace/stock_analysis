# stock_analysis

**2021-05-01留：经过很长时间的研究之后决定公开这些杂乱的代码，当然这些代码已经不是最新的了，是不是最新的也不重要了。总的来说，A股是没有什么规律可言的，很多指标都只是一个参考而已，在研究各种指标的过程中，结合实盘和历史数据，你会对A股有越来越深的认知。如果这些杂乱的代码都够帮助你尽快开始实验，那便达到了它的目的，如果说你想用里面的一些策略去盈利，那是不可能的，因为A股是没有规律可言的，反而处处是陷阱。对于A股而言，你要认清里面的角色，有股东、有机构、有大游资、更有成群的散户。A股是一个纯粹的博弈游戏，要想在这个游戏中取得胜利，你要先明白一个道理：股价是大家合力做上去的，明白了这一点之后，你要学着去站队，学着去规避风险减少回撤保住利润。散户想要在A股里赚快钱，只有一条路，那便是利用散户的优势：船小好掉头。一路研究下来，A股确实是一个有意思的地方，但是还是建议你远离A股，它是一个吃人不吐骨头的地方，是一个费神费力费钱的地方，更是一个尔虞我诈的地方。如果你一定要玩玩A股，感受一下博弈的乐趣，建议不要投入超过10%的资金在里面，大胆去做，盈亏都无所谓。最后，祝君玩的开心，也愿君能大赚！**

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
