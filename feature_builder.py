#/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import Timestamp
import qntstock.time_series_system as tss
from qntstock.data_loader import DataLoader
from qntstock.StockDataFrame import StockDataFrame as SDF
from qntstock.StockSeries import StockSeries as SS

class FeatureBuilder():
    def __init__(self, df):
        self.df = SDF(df)

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


    def combine(self, order, forward=True, suffix='_combined', period=30, strict=[], inplace=False):
        """
        combine given signals, save first matching time and drop others
        order: the signal sequences, the left signal will happen early than the right signal
        forward: if True, matching is from left to right of "order", the first nearest time in right signal will be saved to match each left signal
                 if False, matching is from right to left, the last nearest time in left signal will be saved to match each right signal
        """
        if forward:
            record = tss.combine_forward(self.df, order, period, strict)
        else:
            record = tss.combine_backward(self.df, order, period, strict)
        combined_df = tss.convert_record_to_signal(record, self.df['date'])
        if suffix is None:
            cols = [col for col in self.df if col == 'date' or col not in combined_df]
            self.df = self.df[cols]
        else:
            rename_dict = {k: '%s_%s'%(k, suffix) for k in combined_df if k != 'date'}
            combined_df = combined_df.rename(columns=rename_dict)
        merged_df = self.df.merge(combined_df, on='date')
        if inplace:
            self.df = SDF(merged_df) 
        else:
            return SDF(merged_df)


    def record_to_date(self):
        if self.stat != 'record':
            raise OSError('Error: Could not convert. The status of StockDataFrame is not record. Use method \
                    "combine_backward" or "combine_forward" to get record StockDataFrame"')
        return SDF(tss.convert_record_to_date(self.df, self.df['date']), stat='date_record')


    def record_to_signal(self):
        if self.stat != 'record':
            raise OSError('Error: Could not convert. The status of StockDataFrame is not record. Use method \
                    "combine_backward" or "combine_forward" to get record StockDataFrame"')
        return SDF(tss.convert_record_to_date(self.df, self.df['date']))


    def date_to_signal(self):
        if self.stat != 'date_record':
            raise OSError('Error: Could not convert. The status of StockDataFrame is not date_record. Use method \
                    "record_to_date" to get date_record StockDataFrame"')
        return SDF(tss.convert_date_to_signal(self.df, self.df['date']))


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
    # print(fb.apply_time_series(['MA_5_increase', 'sig_not'], df_60['date'], inplace=False))
    print(fb.df.tail(30))
    print(fb.combine(order=["MA_5_increase","sig_not"], forward=True, suffix=None))
