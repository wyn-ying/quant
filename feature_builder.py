#/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
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
    df = dl.get_stock_data('002230')
    fb = FeatureBuilder(df)
    # fb.add('ma', inplace=True)
    # fb.add('macd', inplace=True)
    fb.add('ma', n=5)
    fb.add('macd')
    fb.add('boll')
    fb.add('bbi')
    fb.add_signal('increase', 'MA_5')
    fb.sig_not('MA_5_increase')
    print(fb.df.head(30))
