#!/usr/bin/python
# -*- coding: utf-8 -*-

#from qntstock.database import get_df
from qntstock.stock_data import get_stock_list, get_stock_data
import numpy as np
from qntstock.createimg import *
from math import floor
class BasePolicy(object):
    def __init__(self):
        self.name='BasePolicy'

    def policy(self, observation):
        next_action = 0
        return next_action

    def set_features(self, env, features):
        # NOTE: set features based on given features
        # should be easy to extend features in the future
        if features == None:
            features = ['open', 'high', 'low', 'close']

        if 'close' not in features:
            features.append('close')
        sample_code = env.codelist[0]
        own_features = list(get_stock_data(sample_code).columns)
        env.features = [i for i in features if i in own_features]
        missing_list = [i for i in features if i not in own_features]
        if len(missing_list) >0:
            logging.warning('The following features are missing:'+str(missing_list))

    def get_observation(self, env):
        # NOTE: observation may include many features, the first must be the status of last action
        # compute relative value of every feature
        last_price = env.df.ix[env.timecnt-1,'close']
        lfeatures = env.df.ix[env.timecnt,env.features] / last_price - 1
        lfeatures = lfeatures.apply(lambda x: floor(x * 10000) / 100).values
        observation = np.concatenate([[env.stat], lfeatures])
        return observation
    '''
    def get_reward(self, env):
        # NOTE: consider how to compute reward is write
        if env.timecnt == env.steps-1:
            reward = 0
        else:
            # all trades are at the end of a day in the current version
            if env.stat == 0 and env.next_action == 1:      # buy
                reward = -env.tax

            elif env.stat == 0 and env.next_action == 0:    # wait
                reward = 0

            elif env.stat == 1 and env.next_action == 1:    # hold
                wave = env.df.ix[env.timecnt,'close'] / env.df.ix[env.timecnt-1,'close'] - 1
                wave = floor(wave * 10000) / 100
                reward = wave

            elif env.stat == 1 and env.next_action == 0:    # sell
                wave = env.df.ix[env.timecnt,'close'] / env.df.ix[env.timecnt-1,'close'] - 1
                wave = floor(wave * 10000) / 100
                reward = wave - env.tax
        return reward
    '''
class FollowPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.name='FollowPolicy'

    def policy(self, observation):
        next_action = 1 if observation[-1]>0 else 0
        return next_action

class RLPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.name='RLPolicy'

    def set_features(self, env, features):
        # TODO: set features as what shape should use for RL
        #        maybe need data of dapan
        features = ['open', 'high', 'close', 'low', 'volume']
        super(RLPolicy, self).set_features(env, features)

    def policy(self, observation):
        # TODO: output next_action based on observation
        return 1

    def get_observation(self, env):
        df = env.df.ix[env.timecnt-env.width:env.timecnt,:]
        observation = translate_img(df, env.width, env.image_hight)
        # TODO: output observation(2darray or it's like)
        return observation

