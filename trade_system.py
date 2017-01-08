#!/usr/bin/python
# -*- coding: utf-8 -*-

from pandas import Timestamp
import pandas as pd
import numpy as np
from qntstock.time_series_system import *
from database import get_stock_list, get_connection
from utils import _sort, PATH, ProgressBar

class TradeSystem:
    def __init__(self, buy=None, sell=None, df=None, gain=None, loss=None, maxreduce=None, gainbias=0.007):
        self._buy           = buy
        self._sell          = sell
        self.df             = df
        self.gain           = gain
        self.loss           = loss
        self.maxreduce      = maxreduce
        self.gainbias       = gainbias
        self.record         = None
        self.successrate    = None
        self.avggainrate    = None
        self.keepdays       = None


    def backtest(self, testlist='all', start='20110101', end=None, savepath=PATH+'/data/backtest_records.csv'):
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
            testlist = []
            for pool in ['0', '3', '6']:
                testlist.extend(get_stock_list(stock_pool=pool))
        con0 = get_connection(stock_pool='0')
        con3 = get_connection(stock_pool='3')
        con6 = get_connection(stock_pool='6')
        con = {'0':con0, '3':con3, '6':con6}
        records, records_tmp = pd.DataFrame(), pd.DataFrame()
        cnt = 0
        bar = ProgressBar(total=len(testlist))
        for code in testlist:
            bar.log(code)
            sql = 'select distinct * from ' + code + ' where date>=' + start\
                + ((' and date<='+end) if end is not None else '') + ';'
            df = pd.read_sql(sql, con[code[2]])
            self.df = _sort(df)
            self.df['date'] = self.df['date'].apply(lambda x: Timestamp(x))
            self.buy()
            self.sell()
            if self.record is not None and len(self.record) > 0:
                self.record = self.record.apply(lambda record: self.integrate(record), axis=1)
            if self.record is not None and len(self.record) > 0:
                records_tmp = self.record.append(records_tmp)
                cnt += 1
            if cnt >= 100:
                records = records_tmp.append(records)
                records_tmp = pd.DataFrame()
                cnt = 0
            bar.move()
        records = records_tmp.append(records)
        if len(records) > 0:
            self.avggainrate = round(records['gainrate'].mean(), 4) - self.gainbias
            self.successrate = round(len(records[records['gainrate']>self.gainbias]) / len(records), 4)
            self.keepdays = round(records['keepdays'].mean(), 2)
            if savepath is not None:
                records.to_csv(savepath)
                print('records is saved at '+savepath)
        else:
            print('No records')


    def predict(self, date):
        """
        考虑扩展性，不仅按照时间判断，还可能输出的是对某一天的打分。
        """
        self.buy()
        pass


    def buy(self):
        """
        生成一个self.record，至少包括列['buydate', 'buy']
        buydate:买入日期
        buy:买入价
        """
        # NOTE: 涨停不能买入
        self._buy(self)
        if self.record is not None and len(self.record)>0:
            self.record['valid'] = self.record.apply(lambda x: self._valid_buy(x), axis=1)
            self.record = self.record.drop(np.where(self.record['valid'] == False)[0])
            self.record = self.record.reset_index(drop=True)


    def sell(self):
        """
        扩展self.record，包括列['selldate', 'sell']
        selldate:按条件卖出日期
        sell:按条件卖出价
        """
        self._sell(self)


    def integrate(self, record):
        """
        扩展self.record，包括列['finaldate', 'final', 'type', 'gainrate']
        finaldate:最终卖出日期
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
            gain_price = record['buy'] * (1 + self.gain) if self.gain is not None else None
            loss_price = record['buy'] * (1 - self.loss) if self.loss is not None else None
            buyidx =  self._get_data(record['buydate']).index[0]
            sellidx =  self._get_data(record['selldate']).index[0]
            final_price, final_type, idx = None, None, None
            for idx in range(buyidx + 1, sellidx + 1):
                data = self.df.loc[idx]
                final_price, final_type = (data['open'], 'loss') if data['open'] < loss_price \
                                     else (data['open'], 'gain') if data['open'] > gain_price \
                                     else (loss_price,   'loss') if data['low']  < loss_price \
                                     else (gain_price,   'gain') if data['high'] > gain_price \
                                     else (None, None)
                if final_type is not None:
                    break
            record['finaldate'] = self.df.loc[idx]['date'] if final_price is not None else record['selldate']
            record['finaldate'] = pd.Timestamp(record['finaldate'])
            record['final'] = final_price if final_price is not None else record['sell']
            record['type'] = final_type if final_price is not None else 'sell'
            record['gainrate'] = record['final'] / record['buy'] - 1 if record['final'] is not None else None
            record['keepdays'] = (record['finaldate']-record['buydate']).days
        return record


    def _get_data(self, date):
        return self.df[self.df['date']==date]


    def _offset_date(self, date, offset):
        idx = self.df[self.df['date'] == date].index[0]
        return self.df.loc[idx+offset, 'date'] if idx + offset < len(self.df) else None


    def _valid_buy(self, x):
        last_date = self._offset_date(x['buydate'], -1)
        close_price = self._get_data(last_date)['close'].values[0]
        open_price = self._get_data(x['buydate'])['open'].values[0]
        return True if open_price < close_price * 1.099 else False


def buy(self):
    # import tushare as ts
    # df = ts.get_k_data('002335',autype=None)
    df = self.df
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
    self.record = convert_record_to_date(rdf, df['date'])
    if self.record is not None:
        self.record['buydate'] = self.record['5f'].apply(lambda x: self._offset_date(x, 1))
        self.record = self.record[pd.notnull(self.record['buydate'])]
        # NOTE: 'open'也可能是'close'
        self.record['buy'] = self.record['buydate'].apply(lambda x: self._get_data(x)['open'].values[0])
        self.record = self.record[['buydate', 'buy']]


def sell(self):
    def offset_func(df, date):
        idx = df[df['date'] == date].index[0]
        return df.loc[idx+2, 'date'] if idx + 2 < len(df) else None
    if self.record is not None:
        self.record['selldate'] = self.record['buydate'].apply(lambda date: self._offset_date(date, 2))
        self.record = self.record[pd.notnull(self.record['selldate'])]
        self.record['sell'] = self.record['selldate'].apply(lambda date: self._get_data(date)['close'].values[0])


if __name__ == '__main__':
    t = TradeSystem(buy=buy, sell=sell, df=df, gain=0.05, loss=0.05)
    t.backtest(start='20160101')
    print(t.avggainrate, t.successrate, t.keepdays)
