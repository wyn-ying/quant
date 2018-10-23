#/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import Timestamp
import qntstock.time_series_system as tss
from qntstock.data_loader import DataLoader
from qntstock.StockDataFrame import StockDataFrame as SDF
from qntstock.StockSeries import StockSeries as SS

class FeatureBuilder():
    def __init__(self, df, state=None):
        self.df = SDF(df)
        self.state = state

    def add(self, indicator_name, *args, **kwargs):
        if 'inplace' not in kwargs.keys():
            kwargs['inplace'] = True
        self.df.add(indicator_name, *args, **kwargs)


    def add_signal(self, signal_name, indicator_name, *args, col_name=None, inplace=True, **kwargs):
        signal_func = getattr(self.df[indicator_name], signal_name)
        if signal_func is not None:
            cur_signal = signal_func(*args, **kwargs)

        if col_name is None:
            col_name = '%s_%s'%(indicator_name, signal_name)
        if col_name in self.df.columns:
            cnt = 1
            cur_col_name = '%s_%s'%(col_name, cnt)
            while cur_col_name in self.df.columns:
                cnt += 1
                cur_col_name = '%s_%s'%(col_name, cnt)
            col_name = cur_col_name

        if inplace:
            self.df.df[col_name] = cur_signal
        else:
            return SS(cur_signal, col_name)


    def sig_and(self, in_col_names, col_name=None, inplace=True):
        tdf = self.df[in_col_names]
        tdf = pd.concat(lines, axis=1)
        s = len(col_names)
        cur_signal = tdf.apply(lambda x: 1 if sum(x) == s else 0, axis=1)
        if col_name is None:
            col_name = 'sig_and'
        if col_name in self.df.columns:
            cnt = 1
            cur_col_name = '%s_%s'%(col_name, cnt)
            while cur_col_name in self.df.columns:
                cnt += 1
                cur_col_name = '%s_%s'%(col_name, cnt)
            col_name = cur_col_name

        if inplace:
            self.df.df[col_name] = cur_signal
        else:
            return SS(cur_signal, col_name)


    def sig_or(self, in_col_names, col_name=None, inplace=True):
        tdf = self.df[in_col_names]
        tdf = pd.concat(lines, axis=1)
        cur_signal = tdf.apply(lambda x: 1 if sum(x) > 0 else 0, axis=1)
        if col_name is None:
            col_name = 'sig_or'
        if col_name in self.df.columns:
            cnt = 1
            cur_col_name = '%s_%s'%(col_name, cnt)
            while cur_col_name in self.df.columns:
                cnt += 1
                cur_col_name = '%s_%s'%(col_name, cnt)
            col_name = cur_col_name

        if inplace:
            self.df.df[col_name] = cur_signal
        else:
            return SS(cur_signal, col_name)


    def sig_not(self, in_col_name, col_name=None, inplace=True):
        ts = self.df[in_col_name]
        cur_signal = ts.apply(lambda x: 1 - x)
        if col_name is None:
            col_name = 'sig_not'
        if col_name in self.df.columns:
            cnt = 1
            cur_col_name = '%s_%s'%(col_name, cnt)
            while cur_col_name in self.df.columns:
                cnt += 1
                cur_col_name = '%s_%s'%(col_name, cnt)
            col_name = cur_col_name

        if inplace:
            self.df.df[col_name] = cur_signal
        else:
            return SS(cur_signal, col_name)


    def apply_time_series(self, in_col_names, date_s, inplace, copy_when_padding=True, any_when_merging=True):
        """
        copy_when_padding: only use when time scale of date_s is shorter than self.df. if True, copy signal 
        value when date_s is shorter than self.df; if False, for each time in self.df, only the last time of 
        date_s will be set as signal value, and others will be fill with 0
        
        any_when_merging: only use when time scale of date_s is larger than self.df. if True, the result will
        set as 1 if there is at least one 1 signal during each time scale of self.df; if False, the result will 
        set as 1 if all signals are 1 during each time scale of self.df.

        change signals [in_col_names] of [self.df] with time scale [date_s]
        NOTE: x+timedelta(hours=20) for D, x+timedelta(hours=22) for W
        """
        to_short_interval = True if date_s[1] - date_s[0] <= self.df.date[1] - self.df.date[0] else False
        if to_short_interval:
            df = pd.DataFrame(0, index=date_s, columns=in_col_names)
            df_idx = pd.DataFrame(date_s)
            df_idx['idx'] = 0
            ll, lb = range(len(self.df.date)),  list(self.df.date) + [Timestamp(3999999999999999999), ]
            df_idx.idx = pd.cut(df_idx.date, bins=lb, labels=ll)
            d_idx = df_idx.set_index('date').to_dict()['idx']
            df_data = self.df[['date']+in_col_names].copy(deep=True)[in_col_names] # df_data = self.df[in_col_names]
            df = df.apply(lambda x: df_data.ix[d_idx[Timestamp(x.name)]], axis=1)
            df = df.reset_index(drop=False)
            # df = pd.concat([pd.DataFrame(date_s), df], axis=1)
        else:
            df = self.df[['date']+in_col_names].copy(deep=True)[in_col_names]
            ll, lb = list(date_s), [Timestamp(0), ] + list(date_s)
            df.date = pd.cut(self.df.date, bins=lb, labels=ll)
            #TODO: now only "any" method, should add "all" method
            df = df.groupby(df.date).sum().clip_upper(1).reset_index(drop=True)
            df = pd.concat([pd.DataFrame(date_s), df], axis=1)

        if inplace:
            self.df = df
        else:
            return SDF(df)


    def combine_backward(self, order, period=30, strict=[]):
        return StockDataFrame(tss.combine_backward(self.df, order, period, strict), stat='record')


    def combine_forward(self, order, period=None, strict=[]):
        return StockDataFrame(tss.combine_forward(self.df, order, period, strict), stat='record')


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


if __name__ == '__main__':
    dl = DataLoader()
    df = dl.get_stock_data('002230', start_date='2015-01-01', end_date='2017-12-31')
    df_w = dl.get_stock_data('002230', start_date='2015-01-01', end_date='2017-12-31', autype='W')
    df_w = SDF(df_w)
    df_60= dl.get_stock_data('002230', start_date='2015-01-01', end_date='2017-12-31', autype='60MIN')
    df_60= SDF(df_60)
    fb = FeatureBuilder(df)
    # fb.add('ma', inplace=True)
    # fb.add('macd', inplace=True)
    fb.add('ma', n=5)
    fb.add('macd')
    fb.add('boll')
    fb.add('bbi')
    fb.add_signal('increase', 'MA_5')
    fb.sig_not('MA_5_increase')
    # print(fb.apply_time_series(['MA_5_increase'], df_w['date'], inplace=False))
    print(fb.apply_time_series(['MA_5_increase', 'sig_not'], df_60['date'], inplace=False))
    print(fb.df.tail(30))
