#!/usr/bin/python
# -*- coding: utf-8 -*-
from RL_brain import DoubleDQN
import numpy as np
import tensorflow as tf
from qntstock.env import Enveriment


def train(RL, env, saver, sess, ckpt_path, test_steps=None, test_code=None, save_steps=None):
    total_steps = 0
    observation = env.reset(start='20160101', end='20161231') # 600212-->15+3
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done = env.step(action)

        # reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > RL.memory_size:   # learning
            RL.learn()
        if done:
            try:
                observation = env.reset(start='20160101', end='20161231')
            except Exception as e:
                print(e)
                print(env.code)
                raise e
        else:
            observation = observation_
        if total_steps - RL.memory_size> 1000000:
            break
        total_steps += 1
        if (test_steps is not None) and (total_steps%test_steps == 0):
            print('training steps:',total_steps)
            test(RL, test_code)
        if (save_steps is not None) and (total_steps%save_steps == 0):
            print('model saving')
            saver.save(sess, ckpt_path, global_step=total_steps)
    return RL.q


def test(RL, test_code, withrand=False):
    e = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    total_steps = 0
    # print('---------- test ----------')
    # observation = e.reset(code=test_code,start='20160101',steps=200,must_this=False) # 600212-->15+3
    observation = e.reset(code=test_code,start='20170101',must_this=False)
    print('test code:', e.code)
    # print('start money:', e.money)
    done = False
    cnt_change = 0
    cnt_success = 0
    last_buy_price = None
    last_sell_price = None
    state = 0
    while not done:
        action = RL.choose_action(observation, withrand=withrand)
        observation, reward, done = e.step(action)
        if action != state:
            cnt_change += 1
            if action == 1:
                last_buy_price = e.df.ix[e.timecnt,'open']
            else:
                last_sell_price = e.df.ix[e.timecnt,'open']
                if last_sell_price > last_buy_price:
                    cnt_success += 1
            state = action
    if state == 1:
        last_sell_price = e.df.tail(1)['close'].values[0]
        if last_sell_price > last_buy_price:
            cnt_success += 1
        cnt_change += 1
    success_rate = cnt_success*2.0/cnt_change if cnt_change > 0 else '--'
    print('end money: {money:>9.4f}, change times: {change:>4}, success times: {success:>4}, success rate: {rate:>4}'.format(money=e.money, change=cnt_change, success=cnt_success, rate=success_rate))
    #print('-------- test end --------')
    return e.money, e.code, cnt_change, success_rate


def DQN_train(config):
    env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    # ckpt_path = './checkpoint/last_model.ckpt'
    ckpt_path = config['path'] if 'path' in config.keys() else None
    MEMORY_SIZE = config['MEMORY_SIZE'] if 'MEMORY_SIZE' in config.keys() else 3000
    ACTION_SPACE = 2
    save_steps = config['save_steps'] if 'save_steps' in config.keys() else 2000
    test_steps = config['test_steps'] if 'test_steps' in config.keys() else 2000
    max_to_keep = config['max_to_keep'] if 'max_to_keep' in config.keys() else 5

    with tf.Session() as sess:
        with tf.variable_scope('Double_DQN'):
            double_DQN = DoubleDQN(
                n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
                e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

        saver = tf.train.Saver(max_to_keep=max_to_keep)
        if ckpt_path is None:
            ckpt_path = '/home/wyn/data/stock_RL_model'
        import os
        os.system('mkdir -p '+ckpt_path)

        model_file = tf.train.latest_checkpoint(ckpt_path)
        if model_file is None:
            model_file = ckpt_path+'/model.ckpt'
            sess.run(tf.global_variables_initializer())
            saver.save(sess, model_file, global_step=0)
        else:
            saver.restore(sess, model_file)
        q_double = train(double_DQN, env, saver, sess, model_file, test_steps=test_steps,save_steps=save_steps)
        # print(q_double)


def all_test(config):
    env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    ckpt_path = config['path'] if 'path' in config.keys() else None
    MEMORY_SIZE = config['MEMORY_SIZE'] if 'MEMORY_SIZE' in config.keys() else 3000
    ACTION_SPACE = 2

    with tf.Session() as sess:
        with tf.variable_scope('Double_DQN'):
            double_DQN = DoubleDQN(
                n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
                e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
        saver = tf.train.Saver()
        if ckpt_path is None:
            ckpt_path = '/home/wyn/data/stock_RL_model'
            print('Error marked: checkpoint path is not given, use default path: '+ckpt_path)
        model_file = tf.train.latest_checkpoint(ckpt_path)
        print('restore from %s'%model_file)
        saver.restore(sess, model_file)
        gain_dic={}
        for code in env.codelist:
            try:
                gain, code, cnt_change, rate = test(double_DQN, code, withrand=False)
                gain_dic[code] = (gain, cnt_change, rate)
            except AssertionError:
                continue
        #q_double = train(double_DQN, env, test_steps=2000)
    print('code number:',len(gain_dic.keys()))
    print('avg money:',sum([i[0] for i in gain_dic.values()])/len(gain_dic.keys()))
    print('min money:',min([i[0] for i in gain_dic.values()]))
    print('avg change times:',sum([i[1] for i in gain_dic.values()])/len(gain_dic.keys()))
    print('avg success rate:',sum([i[2] for i in gain_dic.values() if i[2] != '--'])/len(gain_dic.keys()))
    print('max success rate:',max([i[2] for i in gain_dic.values() if i[2] != '--']))
    print('min success rate:',min([i[2] for i in gain_dic.values() if i[2] != '--']))


def DQN_test():
    env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    ckpt_path = config['path'] if 'path' in config.keys() else None
    MEMORY_SIZE = config['MEMORY_SIZE'] if 'MEMORY_SIZE' in config.keys() else 3000
    ACTION_SPACE = 2

    sess = tf.Session()

    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
    saver = tf.train.Saver()
    if ckpt_path is None:
        ckpt_path = '/home/wyn/data/stock_RL_model'
        print('Error marked: checkpoint path is not given, use default path: '+ckpt_path)
    model_file = tf.train.latest_checkpoint(ckpt_path)
    print('restore from %s'%model_file)
    saver.restore(sess, model_file)
    # test(double_DQN, '600212')
    test(double_DQN, '600212')
    # test(double_DQN, '002853')
    #q_double = train(double_DQN, env, test_steps=2000)
    sess.close()



def model_run(is_train, config):
    if is_train:
        DQN_train(config)
    else:
        all_test(config)


if __name__ == '__main__':
    # DQN_train()
    # DQN_test()
    # all_test()
    config = {}
    config['path'] = '/home/wyn/data/stock_RL_model'
    config['MEMORY_SIZE'] = 3000
    config['save_steps'] = 10000
    config['test_steps'] = 2000
    config['max_to_keep'] = 5
    model_run(is_train=True, config=config)
