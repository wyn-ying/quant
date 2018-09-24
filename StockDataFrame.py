#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame, Timestamp
from qntstock.StockSeries import StockSeries

class StockDataFrame(DataFrame):
    def __init__(self, df):
        #DataFrame.__init__(self, df)
        super(StockDataFrame, self).__init__(df)
        self.df = df
        self.df['date'] = self.df['date'].apply(lambda x: Timestamp(x))


    def __call__(self):
        return self.df


    def __getitem__(self, x):
        if isinstance(x, str):
            return StockSeries(self.df[x], x)
        elif isinstance(x, list):
            return StockDataFrame(self.df[x])
        else:
            raise OSError('Error: Unexpected index for StockDataFrame.')


    def _get_name(self, name):
        return name if name[:5] != 'close' else name[6:]


    def add(self, indicator_name, *args, **kwargs):
        indicator = getattr(self, indicator_name)
        if indicator is not None:
            cur_result = indicator(*args, **kwargs)
            if isinstance(cur_result, StockDataFrame):
                self.df = pd.concat([self.df, cur_result.df], axis=1)
            elif isinstance(cur_result, StockSeries):
                col_name = cur_result.col_name
                self.df[col_name] = cur_result


    def hhv(self, n=5, col='close', inplace=False):
        hhv_s = self.df[col].rolling(center=False, window=n).max()
        col_name = self._get_name(col+'_HHV_'+str(n))
        if inplace:
            self.df[col_name] = hhv_s
        else:
            return StockSeries(hhv_s, col_name)


    def llv(self, n=5, col='close', inplace=False):
        llv_s = self.df[col].rolling(center=False, window=n).min()
        col_name = self._get_name(col+'_LLV_'+str(n))
        if inplace:
            self.df[col_name] = llv_s
        else:
            return StockSeries(llv_s, col_name)


    #NOTE: 类似用ma滤波，但是时效性和降噪性能更优
    def last_max(self, n=5, col='close', inplace=False):
        # n: 平滑天数，小于n天的波动被忽略
        s = self.df[col]
        l = len(s)
        keep_wave_days = [0 for _ in range(l)]
        last_max_s = self.df[col].copy(deep=True)
        up = True
        for i in range(1, l):
            if s[i-1] < s[i]:
                if up:
                    keep_wave_days[i] = keep_wave_days[i-1] + 1
                else:
                    keep_wave_days[i] = 1
                    up = True
            else:
                if up:
                    keep_wave_days[i] = -1
                    up = False
                else:
                    keep_wave_days[i] = keep_wave_days[i-1] - 1

        for i in range(1, l):
            if keep_wave_days[i] == -1:
                if i == 1:
                    last_max_s[i] = last_max_s[i-1]
                #NOTE:前一天那轮的连续上涨时间较短，讨论
                elif keep_wave_days[i-1] < n:
                    last_min_idx = i - 1 - abs(keep_wave_days[i-1])
                    last_max_idx = last_min_idx - abs(keep_wave_days[last_min_idx])
                    # 考虑可能的越界
                    if last_max_idx < 0:
                        print(i)
                        last_max_s[i] = s[i-1]
                    #NOTE: 【前一天(高点)】高于【上轮的高点】，讨论
                    if s[i-1] > s[last_max_idx]:
                        #last_max_s[i] = s[i-1]
                        #NOTE: 看三段的长度是否超过n，如果三短时间很短，并且涨多跌少，讨论
                        if i-1 - last_max_idx + abs(keep_wave_days[last_max_idx]) < n and \
                           abs(keep_wave_days[i-1]+keep_wave_days[last_max_idx]) > abs(keep_wave_days[last_min_idx]):
                            #NOTE: 还没涨超【上轮高点】的last_max_s，可能是震荡或慢涨没见顶，先保持之前的last_max_s
                            if s[i-1] < last_max_s[last_max_idx]:
                                last_max_s[i] = last_max_s[last_min_idx]
                            #NOTE: 涨超了【上轮高点的last_max_s，创新高了直接认
                            else:
                                last_max_s[i] = s[i-1]
                        #NOTE:超过n了，认为是确立上涨趋势,上轮下跌是短期波动，直接认新高
                        else:
                            last_max_s[i] = s[i-1]
                    #NOTE: 上一轮的高点更高，前一天认为是短时反抽，抹平
                    else:
                        last_max_s[i] = last_max_s[last_min_idx]
                #NOTE:前一天那轮的连续上涨超过n天，直接认定前一天为新高
                else:
                    last_max_s[i] = s[i-1]
            else:
                last_max_s[i] = last_max_s[i-1]

        col_name = self._get_name(col+'_LASTMAX_'+str(n))
        if inplace:
            self.df[col_name] = last_max_s
        else:
            return StockSeries(last_max_s, col_name)


    def last_min(self, n=5, col='close', inplace=False):
        # n: 平滑天数，小于n天的波动被忽略
        s = self.df[col]
        l = len(s)
        keep_wave_days = [0 for _ in range(l)]
        last_min_s = self.df[col].copy(deep=True)
        up = True
        for i in range(1, l):
            if s[i-1] < s[i]:
                if up:
                    keep_wave_days[i] = keep_wave_days[i-1] + 1
                else:
                    keep_wave_days[i] = 1
                    up = True
            else:
                if up:
                    keep_wave_days[i] = -1
                    up = False
                else:
                    keep_wave_days[i] = keep_wave_days[i-1] - 1

        for i in range(1, l):
            if keep_wave_days[i] == 1:
                if i == 1:
                    last_min_s[i] = last_min_s[i-1]
                #NOTE:前一天那轮的连续下跌时间较短，讨论
                elif -keep_wave_days[i-1] < n:
                    last_max_idx = i - 1 - abs(keep_wave_days[i-1])
                    last_min_idx = last_max_idx - abs(keep_wave_days[last_max_idx])
                    # 考虑可能的越界
                    if last_min_idx < 0:
                        print(i)
                        last_min_s[i] = s[i-1]
                    #NOTE: 【前一天(低点)】低于【上轮的低点】，讨论
                    if s[i-1] < s[last_min_idx]:
                        #last_min_s[i] = s[i-1]
                        #NOTE: 看三段的长度是否超过n，如果三短时间很短，并且跌多涨少，讨论
                        if i-1 - last_min_idx + abs(keep_wave_days[last_min_idx]) < n and \
                           abs(keep_wave_days[i-1]+keep_wave_days[last_min_idx]) > keep_wave_days[last_max_idx]:
                            #NOTE: 还没跌超【上轮低点】的last_min_s，可能是震荡或阴跌没见底，先保持之前的last_min_s
                            if s[i-1] > last_min_s[last_min_idx]:
                                last_min_s[i] = last_min_s[last_max_idx]
                            #NOTE: 跌超了【上轮低点的last_min_s，之前累积的阴跌太多，认低点
                            else:
                                last_min_s[i] = s[i-1]
                        #NOTE:超过n了，认为是下跌趋势,上轮上涨是短期波动，直接认新低
                        else:
                            last_min_s[i] = s[i-1]
                    #NOTE: 上一轮的低点更低，前一天认为是短时波动，抹平
                    else:
                        last_min_s[i] = last_min_s[last_max_idx]
                #NOTE:前一天那轮的连续下跌超过n天，直接认定前一天为新低
                else:
                    last_min_s[i] = s[i-1]
            else:
                last_min_s[i] = last_min_s[i-1]

        col_name = self._get_name(col+'_LASTMIN_'+str(n))
        if inplace:
            self.df[col_name] = last_min_s
        else:
            return StockSeries(last_min_s, col_name)


    def ma(self, n=5, col='close', inplace=False):
        ma_s = round(self.df[col].rolling(center=False, window=n).mean(), 2)
        col_name = self._get_name(col+'_MA_'+str(n))
        if inplace:
            self.df[col_name] = ma_s
        else:
            return StockSeries(ma_s, col_name)


    def expma(self, n=5, col='close', inplace=False):
        expma_s = round(self.df[col].ewm(span=n, adjust=False).mean(), 2)
        col_name = self._get_name(col+'_EXPMA_'+str(n))
        if inplace:
            self.df[col_name] = expma_s
        else:
            return StockSeries(expma_s, col_name)


    def bbi(self, nums=[3, 6, 12, 24], col='close', inplace=False):
        ma = DataFrame()
        for n in nums:
            ma[n] = round(self.ma(n, col), 2)

        bbi_s = round(ma[nums].apply(sum, axis=1)/4, 2)
        col_name = self._get_name(col+'_BBI_'+str(n))
        if inplace:
            self.df[col_name] = bbi_s
        else:
            return StockSeries(bbi_s, col_name)


    def macd(self, fast=12, slow=26, avg=9, col='close', inplace=False):
        ema_fast = round(self.df[col].ewm(span=fast, adjust=False).mean(), 3)
        ema_slow = round(self.df[col].ewm(span=slow, adjust=False).mean(), 3)
        diff = ema_fast - ema_slow
        dea = round(diff.ewm(span=avg, adjust=False).mean(), 3)
        bar = 2 * (diff - dea)
        dic = {'MACD_BAR': bar, 'MACD_DIFF': diff, 'MACD_DEA': dea}
        macd_df = pd.DataFrame(dic)
        macd_df['MACD_BAR'] = macd_df['MACD_BAR']
        macd_df['MACD_DIFF'] = macd_df['MACD_DIFF']
        macd_df['MACD_DEA'] = macd_df['MACD_DEA']
        if inplace:
            # self.df = pd.concat([self.df, macd_df], axlis=1)
            self.df[['MACD_BAR', 'MACD_DIFF', 'MACD_DEA']] = macd_df
        else:
            return StockDataFrame(macd_df)


    def kdj(self, n=9, m=3, inplace=False):
        low_list = self.df['low'].rolling(window=n, center=False).min()
        low_list = low_list.fillna(value=self.df['low'].expanding(min_periods=1).min())
        high_list = self.df['high'].rolling(window=n, center=False).max()
        high_list=high_list.fillna(value=self.df['high'].expanding(min_periods=1).max())
        rsv = round((self.df['close'] - low_list) / (high_list - low_list) * 100, 1)
        if high_list[len(high_list) - 1] - low_list[len(low_list) - 1] == 0:
            rsv[len(rsv) - 1] = 0
        else:
            rsv[len(rsv) - 1] = 33.3
        kdj_k = round(rsv.ewm(com=m - 1, adjust=False).mean(), 1)
        if not high_list[len(high_list) - 1] - low_list[len(low_list) - 1] == 0:
            kdj_k[len(kdj_k) - 1] = 11.1
        kdj_d = round(kdj_k.ewm(com=m - 1, adjust=False).mean(), 1)
        kdj_j = 3 * kdj_k - 2 * kdj_d
        dic = {'KDJ_K': kdj_k, 'KJD_D': kdj_d, 'KDJ_J': kdj_j}
        kdj_df = pd.DataFrame(dic)
        kdj_df['KDJ_K'] = kdj_df['KDJ_K']
        kdj_df['KDJ_D'] = kdj_df['KDJ_D']
        kdj_df['KDJ_J'] = kdj_df['KDJ_J']
        if inplace:
            self.df[['KDJ_K', 'KDJ_D', 'KDJ_J']] = kdj_df
        else:
            return StockDataFrame(kdj_df)


    def rsi(self, n=6, inplace=False):
        rsi_n = talib.RSI(tdf['close'].values, timeperiod=n)
        rsi_s = round(pd.Series(rsi_n, index=self.df.index).fillna(0), 1)
        col_name = 'RSI_'+str(n)
        if inplace:
            self.df[col_name] = rsi_s
        else:
            return StockSeries(rsi_s, col_name)


    def wr(self, period=14, inplace=False):
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        wr_n = talib.WILLR(high , low, close, period) * -1
        wr_s = pd.Series(wr_n, index=self.df.index)
        col_name = 'WR_'+str(period)
        if inplace:
            self.df[col_name] = wr_s
        else:
            return StockSeries(wr_s, col_name)


    def boll(self, n=20, std=2, inplace=False):
        mid = self.df['close'].rolling(center=False, window=n).mean()
        std_line = self.df['close'].rolling(center=False, window=n).std()
        ub = mid + std * std_line
        lb = mid - std * std_line
        dic = {'BOLL_MD': mid, 'BOLL_UPPER': ub, 'BOLL_LOWER': lb}
        boll_df = pd.DataFrame(dic)
        boll_df['BOLL_MD'] = boll_df['BOLL_MD']
        boll_df['BOLL_UPPER'] = boll_df['BOLL_UPPER']
        boll_df['BOLL_LOWER'] = boll_df['BOLL_LOWER']
        if inplace:
            self.df[['BOLL_MD', 'BOLL_UPPER', 'BOLL_LOWER']] = boll_df
        else:
            return StockDataFrame(boll_df)

    '''
    ##########
    '''

    def increase(self, cols, strict=True, inplace=False, sig_name=None):
        if isinstance(cols, str):
            signal = StockSeries(self.df[cols]).increase(strict)
        else:
            rdf = DataFrame()
            for col in cols:
                rdf[col] = StockSeries(self.df[col]).wave(strict).map(lambda x: True if x > 0 else False)
            signal = rdf.all(axis=1).map(lambda x: 1 if x is True else 0)
        if inplace:
            if sig_name is None:
                sig_name = 'increase_'+'_'.join(cols)
            self.df[sig_name] = signal
        else:
            return StockSeries(signal)


    def decrease(self, cols, strict=True, inplace=False, sig_name=None):
        if isinstance(cols, str):
            signal = StockSeries(self.df[cols]).increase(strict)
        else:
            rdf = DataFrame()
            for col in cols:
                rdf[col] = StockSeries(self.df[col]).wave(strict).map(lambda x: True if x < 0 else False)
            signal = rdf.all(axis=1).map(lambda x: 1 if x is True else 0)
        if inplace:
            if sig_name is None:
                sig_name = 'decrease_'+'_'.join(cols)
            self.df[sig_name] = signal
        else:
            return StockSeries(signal)


    def long(self, cols, strict=True, inplace=False, sig_name=None):
        rdf = DataFrame()
        for i in range(1,len(cols)):
            name = cols[i-1]+'_'+cols[i]
            rdf[name] = StockSeries(self.df[cols[i-1]]).upper_than(self.df[cols[i]], strict)
            rdf[name] = rdf[name].map(lambda x: True if x == 1 else False)
        signal = rdf.all(axis=1).map(lambda x: 1 if x is True else 0)
        if inplace:
            if sig_name is None:
                sig_name = 'long_'+'_'.join(cols)
            self.df[sig_name] = signal
        else:
            return StockSeries(signal)


    def short(self, cols, strict=True, inplace=False, sig_name=None):
        rdf = DataFrame()
        for i in range(1,len(cols)):
            name = cols[i-1]+'_'+cols[i]
            rdf[name] = StockSeries(self.df[cols[i-1]]).lower_than(self.df[cols[i]], strict)
            rdf[name] = rdf[name].map(lambda x: True if x == 1 else False)
        signal = rdf.all(axis=1).map(lambda x: 1 if x is True else 0)
        if inplace:
            if sig_name is None:
                sig_name = 'short_'+'_'.join(cols)
            self.df[sig_name] = signal
        else:
            return StockSeries(signal)
 
    '''
    ##########
    '''

    def combine_backward(self, order, period=30, strict=[]):
        return StockDataFrame(tss.combine_backward(self.df, order, period, strict), stat='record')


    def combine_forward(self, order, period=None, strict=[]):
        return StockDataFrame(tss.combine_forward(self.df, order, period, strict), stat='record')


    def to_coarse_grained(self, signal_cols, df_cg):
        """
        combine fine-grained signal to coarse-grained
        NOTE: x+timedelta(hours=20) for D, x+timedelta(hours=22) for W
        """
        if df_cg.date[1]-df_cg.date[0] <= self.df.date[1]-self.df.date[0]:
            raise OSError('Error: given "df_cg" is fine-grained than df.')
        #data_cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 'tor']
        #signals = [col for col in self.df.columns if col not in data_cols]
        #df = self.df[signals].copy(deep=True)
        df = self.df[signal_cols].copy(deep=True)
        ll, lb = list(df_cg.date), [Timestamp(0), ] + list(df_cg.date)
        df.date = pd.cut(self.df.date, bins=lb, labels=ll)
        df = df.groupby(df.date).sum().clip_upper(1).reset_index(drop=True)
        df = pd.concat([df_cg, df], axis=1)
        return StockDataFrame(df)


    def to_fine_grained(self, signal_cols, df_fg):
        if df_fg.date[1]-df_fg.date[0] >= self.df.date[1]-self.df.date[0]:
            raise OSError('Error: given "df_fg" is coarse-grained than df.')
        df = pd.DataFrame(0, index=df_fg.date, columns=signal_cols)

        df_idx = df_fg[['date',]]
        df_idx['idx'] = 0
        ll, lb = range(len(self.df.date)), [Timestamp(0), ] + list(self.df.date)
        df_idx.idx = pd.cut(df_idx.date, bins=lb, labels=ll)
        d_idx = df_idx.set_index('date').to_dict()['idx']
        df_data = self.df[signal_cols]
        df = df.apply(lambda x: df_data.ix[d_idx[x.name]], axis=1)
        return df


    def record_to_date(self):
        if self.stat != 'record':
            raise OSError('Error: Could not convert. The status of StockDataFrame is not record. Use method \
                    "combine_backward" or "combine_forward" to get record StockDataFrame"')
        return StockDataFrame(tss.convert_record_to_date(self.df, self.df['date']), stat='date_record')


    def record_to_signal(self):
        if self.stat != 'record':
            raise OSError('Error: Could not convert. The status of StockDataFrame is not record. Use method \
                    "combine_backward" or "combine_forward" to get record StockDataFrame"')
        return StockDataFrame(tss.convert_record_to_date(self.df, self.df['date']))


    def date_to_signal(self):
        if self.stat != 'date_record':
            raise OSError('Error: Could not convert. The status of StockDataFrame is not date_record. Use method \
                    "record_to_date" to get date_record StockDataFrame"')
        return StockDataFrame(tss.convert_date_to_signal(self.df, self.df['date']))


    def _last_min_simple(self, n=3, col='close', inplace=False):
        # n: 平滑天数，小于n天的波动被忽略
        s = self.df[col]
        l = len(s)
        keep_wave_days = [0 for _ in range(l)]
        last_min_s = self.df[col].copy(deep=True)
        up = True
        for i in range(1, l):
            if s[i-1] < s[i]:
                if up:
                    keep_wave_days[i] = keep_wave_days[i-1] + 1
                else:
                    keep_wave_days[i] = 1
                    up = True
            else:
                if up:
                    keep_wave_days[i] = -1
                    up = False
                else:
                    keep_wave_days[i] = keep_wave_days[i-1] - 1

        for i in range(1, l):
            if keep_wave_days[i] == 1:
                if i == 1:
                    last_min_s[i] = last_min_s[i-1]
                #NOTE:前一天那轮的连续下跌时间较短，讨论
                elif -keep_wave_days[i-1] < n:
                    last_max_idx = i - 1 - abs(keep_wave_days[i-1])
                    last_min_idx = last_max_idx - abs(keep_wave_days[last_max_idx])
                    # 考虑可能的越界，应该不存在，s[0] = 0
                    #if last_min_idx < 0:
                    #    print(i)
                    #    last_min_s[i] = s[i-1]
                    #NOTE: 【前一天(低点)】低于【上轮的低点】，认新低
                    if s[i-1] < s[last_min_idx]:
                        last_min_s[i] = s[i-1]
                    #NOTE: 上一轮的低点更低，前一天认为是短时波动，抹平
                    else:
                        last_min_s[i] = last_min_s[last_max_idx]
                #NOTE:前一天那轮的连续下跌超过n天，直接认定前一天为新低
                else:
                    last_min_s[i] = s[i-1]
            else:
                last_min_s[i] = last_min_s[i-1]

        if inplace:
            self.df[self._get_name(col+'_LASTMIN_'+str(n))] = last_min_s
        else:
            return StockSeries(last_min_s)


if __name__ == "__main__":
    from qntstock.stock_data import get_stock_data
    #df = get_stock_data('002302', start_date='2017-01-01')
    df = get_stock_data('002230', start_date='20141231', autype='W')
    df = StockDataFrame(df)
    _ =df._last_min_mod()
    df['MA_5'] = df.ma(5)
    df.ma(10, inplace=True)
    df.boll(inplace=True)
    print(df)
    print(type(df.long(['close', 'MA_5'])))
    print(df.increase(['MA_5',]))
    #print(df.increase('MA_5'))
    #print(df.long(['close', 'MA_5']))
    #print(df['BOLL_MD'])
    #成员.运算是哪个__xxx__函数
    print(df.columns)
