#!/usr/bin/python
# -*- coding: utf-8 -*-

from pandas import Timestamp
import pandas as pd
import numpy as np
import os
from qntstock.time_series_system import *
from qntstock.utils import _sort, PATH, FS_PATH, FS_PATH_OL, ProgressBar
from qntstock.stock_data import get_stock_data
from functools import reduce
import StockDataFrame as SDF
import StockSeries as SS


class TradeSystem:
    def __init__(self, buy=None, sell=None, df=None, gain=None, loss=None, maxreduce=None, gainbias=0.007):
        self._buy           = buy
        self._sell          = sell
        self.gain           = gain
        self.loss           = loss
        self.maxreduce      = maxreduce
        self.gainbias       = gainbias
        self.record         = None
        self.successrate    = None
        self.avggainrate    = None
        self.keepdays       = None


    def backtest(self, testlist='all', start='2014-01-01', end=None, savepath=PATH+'/data/backtest_records.csv'):
        """
        backtest(self, testlist='all', start='20110101', end=None):

        回测系统，统计成功率和平均收益率和持有天数
        # TODO: 对学习系统还应该有准确率和召回率

        Input:
            testlist: ('all' or list of code): 要进行回测的股票列表, 'all'为全部股票

            start: (string of date): 回测数据的起始时间(含start)

            end: (string of date): 回测数据的结束时间(含end)

            savepath: (string): 生成的记录报告的保存路径,为None则不保存
        Return:
            None
        """
        if testlist is 'all':
            testlist = os.listdir(FS_PATH)
            testlist = [filename.split('.')[0] for filename in testlist]
        records = None
        records_tmp = [None for _ in testlist]
        cnt = 0
        bar = ProgressBar(total=len(testlist))
        for i, code in enumerate(testlist):
            bar.log(code)
            df = get_stock_data(code, start, end)
            df['date'] = df['date'].apply(lambda x: Timestamp(x))
            buy_record = self.buy(df, code, start, end)
            buy_and_sell_record = self.sell(df, buy_record)
            if buy_and_sell_record is not None and len(buy_and_sell_record) > 0:
                buy_and_sell_record = buy_and_sell_record.apply(lambda record: self.integrate(df, record), axis=1)
                buy_and_sell_record.insert(0,'code',[code for _ in range(len(buy_and_sell_record))])
            records_tmp[i] = buy_and_sell_record
            bar.move()
        if len(records_tmp) > 1:
            all_None = reduce(lambda x,y: None if x is None and y is None else 1, records_tmp)
        else:
            all_None = None if records_tmp[0] is None else 1
        if all_None is not None:
            records = pd.concat(records_tmp)
        if records is not None and len(records) > 0:
            self.avggainrate = round(records['gainrate'].mean(), 4) - self.gainbias
            self.successrate = round(len(records[records['gainrate']>self.gainbias]) / len(records), 4)
            self.keepdays = round(records['keepdays'].mean(), 2)
            if savepath is not None:
                records.to_csv(savepath, index=False)
                print('records is saved at '+savepath)
        else:
            print('No records')


    def predict(self, date):
        """
        考虑扩展性，不仅按照时间判断，还可能输出的是对某一天的打分。
        """
        #self.buy(df)
        pass


    def buy(self, df, code, start, end):
        """
        生成一个buy_record，至少包括列['buydate', 'buy']
        buydate:买入日期
        buy:买入价
        """
        # NOTE: 涨停不能买入
        buy_record = self._buy(df, code, start, end)
        if buy_record is not None and len(buy_record)>0:
            buy_record['valid'] = buy_record.apply(lambda x: _valid_buy(df, x), axis=1)
            buy_record = buy_record.reset_index(drop=True).drop(np.where(buy_record['valid'] == False)[0])
            buy_record = buy_record.reset_index(drop=True)
        return buy_record


    def sell(self, df, buy_record):
        """
        生成buy_and_sell_record，包括列['selldate', 'sell']
        selldate:按条件卖出日期
        sell:按条件卖出价
        """
        buy_and_sell_record = self._sell(df, buy_record)
        return buy_and_sell_record


    def integrate(self, df, record):
        """
        由于止盈或止损导致最终卖出点有变化（提前），原计划卖出是sell，最终卖出是final
        生成final_record，包括列['finaldate', 'final', 'type', 'gainrate']
        finaldate:最终卖出日期，由于
        final:最终卖出价
        type:卖出方式，包括止盈'gain'，止损'sell'，正常卖出'sell'
        gainrate:收益率
        keepdays:持股天数
        """
        if record['selldate'] is None:
            record['finaldate'] = None
            record['final_price'] = None
            record['final_type'] = None
        else:
            gain_price = record['buy'] * (1 + self.gain) if self.gain is not None else 9999999
            loss_price = record['buy'] * (1 - self.loss) if self.loss is not None else 0
            buyidx =  _get_date(df, record['buydate']).index[0]
            sellidx =  _get_date(df, record['selldate']).index[0]
            final_price, final_type, idx = None, None, None
            for idx in range(buyidx + 1, sellidx + 1):
                data = df.loc[idx]
                final_price, final_type = (data['open'], 'loss') if data['open'] < loss_price \
                                     else (data['open'], 'gain') if data['open'] > gain_price \
                                     else (loss_price,   'loss') if data['low']  < loss_price \
                                     else (gain_price,   'gain') if data['high'] > gain_price \
                                     else (None, None)
                if final_type is not None:
                    break
            record['finaldate'] = df.loc[idx]['date'] if final_price is not None else record['selldate']
            record['finaldate'] = pd.Timestamp(record['finaldate'])
            record['final'] = final_price if final_price is not None else record['sell']
            record['type'] = final_type if final_price is not None else 'sell'
            record['gainrate'] = record['final'] / record['buy'] - 1 if record['final'] is not None else None
            record['keepdays'] = (record['finaldate']-record['buydate']).days
        return record


def _get_date(df, date):
    return df[df['date']==date]


def _offset_date(df, date, offset):
    idx = df[df['date'] == date].index[0]
    return df.loc[idx+offset, 'date'] if idx + offset < len(df) else None


def _valid_buy(df, x):
    last_date = _offset_date(df, x['buydate'], -1)
    close_price = _get_date(df, last_date)['close'].values[0]
    open_price = _get_date(df, x['buydate'])['open'].values[0]
    return True if open_price < close_price * 1.099 else False


def buy(df, code, start, end):
    df_w = SDF(get_stock_data(code, start, end, autype='W'))
    df_w['date'] = df_w['date'].apply(lambda x: Timestamp(x))
    df_w.macd(inplace=True)
    #df_w.


    ma_raise = factor_ma_raise(df)
    ma_long = factor_ma_long_position(df)
    cond_df = pd.DataFrame()
    ma_line=ma(df)
    cond_df[['date','close']] = df[['date','close']]
    cond_df['MA_5'] = ma_line['MA_5']
    cond_df['MA_10'] = ma_line['MA_10']
    cond_df['5r'] = inflection(ma_line['MA_5'], strict=True, signal_type='raise')
    cond_df['5rc10'] = cross(ma_line['MA_5'], ma_line['MA_10'], strict=True, signal_type='raise')
    cond_df['5f'] = inflection(ma_line['MA_5'], strict=True, signal_type='fall')
    order =['5r', '5rc10', '5f']
    rdf = combine_backward(cond_df, order=order, period=20)
    buy_record = convert_record_to_date(rdf, df['date'])
    if buy_record is not None:
        buy_record['buydate'] = buy_record['5f'].apply(lambda x: _offset_date(df, x, 1))
        buy_record = buy_record[pd.notnull(buy_record['buydate'])]
        # NOTE: 'open'也可能是'close'
        buy_record['buy'] = buy_record['buydate'].apply(lambda x: _get_date(df, x)['open'].values[0])
        buy_record = buy_record[['buydate', 'buy']]
    return buy_record


def sell(df, buy_record):
    ##def offset_func(df, date):
    ##    idx = df[df['date'] == date].index[0]
    ##    return df.loc[idx+2, 'date'] if idx + 2 < len(df) else None
    buy_and_sell_record = buy_record
    if buy_and_sell_record is not None:
        buy_and_sell_record['selldate'] = buy_and_sell_record['buydate'].apply(lambda date: _offset_date(df, date, 2))
        buy_and_sell_record = buy_and_sell_record[pd.notnull(buy_and_sell_record['selldate'])]
        buy_and_sell_record['sell'] = buy_and_sell_record['selldate'].apply(lambda date: _get_date(df, date)['close'].values[0])
    return buy_and_sell_record


if __name__ == '__main__':
    df = pd.DataFrame()
    from qntstock.strategy import buy_1, sell_1
    t = TradeSystem(buy=buy, sell=sell, df=df, gain=0.05, loss=0.05)
    #t.backtest(start='2017-01-01')
    #t.backtest(['000001','002230'],start='2017-01-01', savepath='test_trade_sys.csv')
    #t.backtest('all',start='2016-01-01', savepath='test_buy_1_sell_1.csv')
    #t.backtest(['603758'],start='2016-01-01', savepath='test_sell_1.csv')
    t.backtest(['002032'],start='2016-01-01', savepath=None)
    print('gain rate:', t.avggainrate, '\nsuccess rate:', t.successrate, '\nkeep days:', t.keepdays)
