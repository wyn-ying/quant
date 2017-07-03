#!/usr/bin/python
# -*- coding: utf-8 -*-
from RL_brain import DoubleDQN
import numpy as np
import tensorflow as tf
from qntstock.env import Enveriment

def train(RL, env, saver, sess, test_steps=None, test_code=None, save_steps=None):
    import os
    os.system('mkdir -p ./checkpoint')

    total_steps = 0
    observation = env.reset(start='20160101') # 600212-->15+3
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done = env.step(action)

        # reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > RL.memory_size:   # learning
            RL.learn()
        if done:
            try:
                observation = env.reset(start='20160101')
            except Exception as e:
                print(e)
                print(env.code)
                raise e
        else:
            observation = observation_
        if total_steps - RL.memory_size> 500000:
            break
        total_steps += 1
        if (test_steps is not None) and (total_steps%test_steps == 0):
            print('training steps:',total_steps)
            test(RL, test_code)
        if (save_steps is not None) and (total_steps%save_steps == 0):
            print('model saving')
            saver.save(sess, './checkpoint/last_model.ckpt')
    return RL.q

def test(RL, test_code, withrand=False):
    e = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    total_steps = 0
    #print('---------- test ----------')
    observation = e.reset(code=test_code,start='20160101',must_this=False) # 600212-->15+3
    print('test code:', e.code)
    print('start money:', e.money)
    done = False
    while not done:
        action = RL.choose_action(observation, withrand=withrand)
        observation, reward, done = e.step(action)
    print('end money:', e.money)
    print('-------- test end --------')
    return e.money

def DQN_train():
    env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    restore_from_ckpt = './checkpoint/last_model.ckpt'
    MEMORY_SIZE = 3000
    ACTION_SPACE = 2

    sess = tf.Session()

    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if restore_from_ckpt is not None:
        saver.restore(sess, restore_from_ckpt)
    test(double_DQN,'600212')
    q_double = train(double_DQN, env, saver, sess,test_steps=2000,save_steps=2000)
    # print(q_double)
    test(double_DQN, '600212')
    sess.close()


def all_test():
    env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    restore_from_ckpt = './checkpoint/last_model.ckpt'
    save_steps = 2000
    MEMORY_SIZE = 3000
    ACTION_SPACE = 2

    sess = tf.Session()

    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
    print('------',tf.get_variable_scope().original_name_scope)
    saver = tf.train.Saver()
    if restore_from_ckpt is not None:
        saver.restore(sess, restore_from_ckpt)
    for code in env.codelist:
        test(double_DQN, code)
    #q_double = train(double_DQN, env, test_steps=2000)
    sess.close()

def DQN_test():
    env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    restore_from_ckpt = './checkpoint/last_model.ckpt'
    save_steps = 2000
    MEMORY_SIZE = 3000
    ACTION_SPACE = 2

    sess = tf.Session()

    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
    print('------',tf.get_variable_scope().original_name_scope)
    saver = tf.train.Saver()
    if restore_from_ckpt is not None:
        saver.restore(sess, restore_from_ckpt)
    # test(double_DQN, '600212')
    # test(double_DQN, '600212')
    test(double_DQN, '002853')
    #q_double = train(double_DQN, env, test_steps=2000)
    sess.close()
if __name__ == '__main__':
    # DQN_train()
    DQN_test()
    #all_test()
