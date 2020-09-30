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
