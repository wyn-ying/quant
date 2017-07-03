#!/usr/bin/python
# -*- coding: utf-8 -*-
from qntstock.database import get_stock_list, get_df
import logging
from random import choice
from qntstock import policys
from math import floor

class Enveriment(object):
    def __init__(self, policy=None, window_width=15, features=None, start=None, end=None, data_df=None, tax=0.7):
        self.width = window_width
        # short is 0, long is 1
        self.action_space = [0, 1]
        self.n_actions = 2
        self.n_features = None
        if policy is not None:
            if hasattr(policys, policy):
                policy = getattr(policys, policy)
            else:
                logging.warning('Can not find policy by name: '+policy+'.\nUse default BasePolicy')
                policy = None
        self.p = policys.BasePolicy() if policy is None else policy()
        self.features = None
        self._set_features(features)
        self.df = data_df
        self.df_has_set = True if data_df is not None else False
        self.start = start if start is not None else '20140101'
        self.end = end
        self.steps = None
        self.timecnt =None
        self.codelist = []
        self.stat = None
        self.code = None
        self.date = None
        self.tax = tax
        self.lstart = None
        self.lend = None
        self.lsteps = None
        for i in ['0','3','6']:
            self.codelist.extend(get_stock_list(stock_pool=i))

    def reset(self, code=None, start=None, end=None, steps=None, must_this=True):
        lcode = choice(self.codelist) if code is None else code
        if not self.df_has_set:
            self.code = lcode
            self.lstart = start if start is not None else self.start
            self.lend = end if end is not None else self.end
            self.lsteps = steps
            self.df = self._get_df(lcode, self.lstart, self.lend, self.lsteps)
            while self.df.shape[0] < self.width + (0 if steps is None else steps) + 1:
                assert (code is None) or (not must_this), 'code %s has not enough length during %s to %s'%(
                        code,
                        self.lstart if self.lstart is not None else 'the begining',
                        self.lend if self.lend is not None else 'the end')
                if code is not None:
                    print('code %s has not enough length during %s to %s. as not set [must_this], changing for another code'%(code, self.lstart if self.lstart is not None else 'the begining', self.lend if self.lend is not None else 'the end'))

                lcode = choice(self.codelist)
                self.code = lcode
                self.df = self._get_df(lcode, self.lstart, self.lend, self.lsteps)
        # TODO: TODO: TODO:
        # 通过time_series_system的函数，结合features，一次性生成需要的特征，注意必须保留'close'，建议保留当日相比昨日的4个涨幅
        # 这个功能放到policy里，在这里调用self.df=p.xxxx(self.df, self.features)
        self.timecnt = self.width - 1
        self.steps = len(self.df)
        self.last_buy_price = 0
        self.money = 100
        self.stat = 0
        observation = self._get_observation()
        return observation

    def _set_features(self, features):
        self.p.set_features(self, features)

    def _get_df(self, code, start, end, steps):
        df = get_df(code, start=start, end=end)
        df = df.tail(self.width+steps) if steps is not None else df
        df = df.reset_index(drop=True)
        return df

    def _set_stat(self, action):
        self.stat = action

    def _get_observation(self):
        return self.p.get_observation(self)

    def _get_reward(self, action):
        # NOTE: consider how to compute reward is write
        if self.stat == 0 and action == 1:      # buy
            wave = self.df.ix[self.timecnt,'close'] / self.df.ix[self.timecnt,'open'] - 1
            wave = floor(wave * 10000) / 100
            reward = wave - self.tax

        elif self.stat == 0 and action == 0:    # wait
            reward = 0

        elif self.stat == 1 and action == 1:    # hold
            wave = self.df.ix[self.timecnt,'close'] / self.df.ix[self.timecnt-1,'close'] - 1
            wave = floor(wave * 10000) / 100
            reward = wave

        elif self.stat == 1 and action == 0:    # sell
            wave = self.df.ix[self.timecnt,'open'] / self.df.ix[self.timecnt-1,'close'] - 1
            wave = floor(wave * 10000) / 100
            reward = wave - self.tax
        return reward

    def _get_if_done(self):
        return True if (self.timecnt==self.steps-1) else False

    def step(self, action):
        # all trades are at the begin of a day in the current version
        # compute now money, last buy price (based on stat and next action), stat, and so on
        if self._get_if_done():
            self.reset(self.code, self.lstart, self.lend, self.lsteps)
        self.timecnt += 1
        if self.stat == 0 and action == 1:      # buy
            self.last_buy_price = self.df.ix[self.timecnt,'open']
            self.money = self.money * (100-self.tax) / 100

        elif self.stat == 0 and action == 0:    # wait
            pass

        elif self.stat == 1 and action == 1:    # hold
            # add last unfinished trade into total money
            if self.timecnt == self.steps-1:
                self.money *= (self.df.ix[self.timecnt,'close'] / self.last_buy_price * (100-self.tax)/100)

        elif self.stat == 1 and action == 0:    # sell
            self.money *= (self.df.ix[self.timecnt,'open'] / self.last_buy_price * (100-self.tax)/100)

        reward = self._get_reward(action)
        observation = self._get_observation()
        done = self._get_if_done()
        # update the state of the enveriment
        self.date = self.df.ix[self.timecnt,'date']
        self._set_stat(action)

        return (observation, reward, done)


if __name__ == '__main__':
    e = Enveriment(policy='FollowPolicy', features=['open','high','low','close'])
    #e = Enveriment(policy='RLPolicy', features=['open','high','low','close','volume'])
    # observation = e.reset(code='600212',start='20170101', steps=3) # 600212-->15+3
    observation = e.reset(start='20170101') # 600212-->15+3
    print(observation, '\n')
    print('Some infermation after reset:', e.code)
    done = False
    print(e.features)
    action = 0
    cnt = 0
    while cnt < 3:
        action = e.p.policy(observation)
        observation, reward, done = e.step(action)
        print(reward, e.money, e.date, e.stat)
        print(observation, '\n')
        if done:
            cnt += 1
            done = False
            print('cnt: ', cnt)
            print(e.code, e.money)
            print(e.last_buy_price, e.df.tail(1)['close'])
            observation = e.reset(code='600212',start='20170101',steps=2)
