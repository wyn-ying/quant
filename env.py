#!/usr/bin/python
# -*- coding: utf-8 -*-
from qntstock.database import get_stock_list, get_df
import logging
from random import choice
import policys

class Enveriment(object):
    def __init__(self, policy=None, features=None, start=None, end=None, steps=None, data_df=None, tax=0.7):
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
        self.steps = steps
        self.timecnt =None
        self.codelist = []
        self.stat = None
        self.next_action = None
        self.last_buy_price = 0
        self.money = 100
        self.code = None
        self.date = None
        self.tax = tax
        for i in ['0','3','6']:
            self.codelist.extend(get_stock_list(stock_pool=i))

    def reset(self, code=None, start=None, end=None, steps=None):
        if code==None:
            code = choice(self.codelist)
        if not self.df_has_set:
            self.code = code
            lstart = start if start is not None else self.start
            lend = end if end is not None else self.end
            lsteps = steps if steps is not None else self.steps
            self.df = self._get_df(code, lstart, lend, lsteps)
        self.timecnt = 0
        self.steps = len(self.df)
        self.stat = 0
        self.next_action = None

    def _set_features(self, features):
        self.p.set_features(self, features)

    def _get_df(self, code, start, end, steps):
        df = get_df(code, start=start, end=end)
        return df.tail(steps) if steps is not None else df

    def _get_stat(self, action):
        self.stat = action

    def _get_observation(self):
        return self.p.get_observation(self)

    def _get_reward(self):
        return self.p.get_reward(self)

    def _get_if_done(self):
        return True if (self.timecnt==self.steps-1) else False

    def step(self, action):
        self.next_action = action
        # all trades are at the end of a day in the current version
        # compute now money, last buy price (based on stat and next action), stat, and so on
        if self.stat == 0 and action == 1:      # buy
            self.last_buy_price = self.df.ix[self.timecnt,'close']
            self.money = self.money * (100-self.tax) / 100

        elif self.stat == 0 and action == 0:    # wait
            pass

        elif self.stat == 1 and action == 1:    # hold
            # add last unfinished trade into total money
            if self.timecnt == self.steps-1:
                self.money *= (self.df.ix[self.timecnt,'close'] / self.last_buy_price * (100-self.tax)/100)

        elif self.stat == 1 and action == 0:    # sell
            self.money *= (self.df.ix[self.timecnt,'close'] / self.last_buy_price * (100-self.tax)/100)

        reward = self._get_reward()
        observation = self._get_observation()
        done = self._get_if_done()
        # update the state of the enveriment
        self.date = self.df.ix[self.timecnt,'date']
        self.timecnt += 1
        self._get_stat(self.next_action)

        return (observation, reward, done)


if __name__ == '__main__':
    e = Enveriment(policy='FollowPolicy', features=['open','high','low','close'])
    e.reset(start='20170101')
    print('Some infermation after reset:', e.code)
    done = False
    print(e.features)
    action = 0
    while not done:
        observation, reward, done = e.step(action)
        if not done:
            action = e.p.policy(observation)
        print(reward, e.money, e.date, e.stat)
        print(observation, '\n')
    print(e.code, e.money)
    print(e.last_buy_price, e.df.tail(1)['close'])
