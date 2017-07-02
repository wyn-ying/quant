"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


# import gym
from RL_brain import DoubleDQN
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from qntstock.env import Enveriment

env = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
MEMORY_SIZE = 3000
ACTION_SPACE = 2

sess = tf.Session()
# with tf.variable_scope('Natural_DQN'):
    # natural_DQN = DoubleDQN(
        # n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        # e_greedy_increment=0.001, double_q=False, sess=sess
    # )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=env.width*env.width*3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset(start='20160101') # 600212-->15+3
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done = env.step(action)

        # reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()
        if done:
            observation = env.reset(start='20160101')
        else:
            observation = observation_
        if total_steps - MEMORY_SIZE > 2000000:
            break
        total_steps += 1
        if total_steps%1000 == 0:
            print(total_steps)
            test(RL)
    return RL.q

def test(RL):
    e = Enveriment(policy='RLPolicy', features=['open','high','close','low','volume'])
    total_steps = 0
    print('----------test----------')
    observation = e.reset(code='600212',start='20160101') # 600212-->15+3
    print('test code:', e.code)
    print('start money:', e.money)
    done = False
    while not done:
        action = RL.choose_action(observation)
        observation, reward, done = e.step(action)
    print('end money:', e.money)
    print('--------test end --------')
    return env.money

q_double = train(double_DQN)
# print(q_double)

test_money = test(double_DQN)
print(test_money)
