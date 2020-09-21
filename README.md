# stock_analysis

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
