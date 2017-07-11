#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

#np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
    def __init__(
            self,
            feature_config,
            n_actions=2,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):
        self.n_features = feature_config['n_features']
        self.x_pixels = feature_config['x_pixels']
        self.y_pixels = feature_config['y_pixels']
        self.n_channels= feature_config['n_channels']
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self._build_net()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.histogram('loss', self.loss)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(
                    0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(
                self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(
                self.s_, c_names, n_l1, w_initializer, b_initializer)

    def reshape(self, observation):
        return observation.reshape(-1)

    def restore(self, observation):
        if len(observation.shape) == 1:
            observation = observation[np.newaxis, :]
        return observation

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        stored_s = self.reshape(s)  # TODO:notify shape
        stored_s_ = self.reshape(s_)  # TODO:notify shape
        transition = np.hstack((stored_s, [a, r], stored_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, withrand=True):
        observation = self.reshape(observation)  # TODO:notify shape
        observation = self.restore(observation)
        actions_value = self.sess.run(
            self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if withrand and (np.random.uniform() > self.epsilon):  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # batch_memory = self.restore(batch_memory)

        data_s = self.restore(batch_memory[:, -self.n_features:])
        data_s_ = self.restore(batch_memory[:, -self.n_features:])
        q_next, q_eval4next = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: data_s_, self.s: data_s})    # next observation

        data_s = self.restore(batch_memory[:, :self.n_features])
        q_eval = self.sess.run(self.q_eval, {self.s: data_s})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            # the action that brings the highest value is evaluated by q_eval
            max_act4next = np.argmax(q_eval4next, axis=1)
            # Double DQN, select q_next depending on above actions
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * selected_q_next

        data_s = self.restore(batch_memory[:, :self.n_features])
        _, self.cost, result = self.sess.run([self._train_op, self.loss, self.merged],
                                     feed_dict={self.s: data_s,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            self.writer.add_summary(result,self.learn_step_counter)
            # print('\ntarget_params_replaced\n')


        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


class DoubleDQRCNN(DoubleDQN):
    def __init__(
            self,
            feature_config,
            n_actions=2,
            learning_rate=0.0005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):
        self.n_features = feature_config['n_features']
        self.x_pixels = feature_config['x_pixels']
        self.y_pixels = feature_config['y_pixels']
        self.n_channels = feature_config['n_channels']
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self._build_net()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.histogram('loss', self.loss)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_layers(self, s, c_names, normrand_initializer, const_initializer):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

        with tf.variable_scope('conv1'): #s: [batchsize, xlen, ylen, n_channels]
            fea_map_num1 = 16
            filter_width=3
            w1 = tf.get_variable('w', [filter_width, self.y_pixels, self.n_channels, fea_map_num1],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b1 = tf.get_variable('b', [1, fea_map_num1],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            conv1 = tf.nn.relu(conv2d(s, w1) + b1)  #conv1: [batchsize ,xlen-filter_width+1, ylen=1, fea_map_num1]

        with tf.variable_scope('LSTM'):
            rnn_input = tf.squeeze(conv1, [2,]) # rnn_input: [batch_size, xlen-filter_width, fea_map_num1]
            batch_size = tf.shape(rnn_input)[0]
            cell = tf.contrib.rnn.BasicLSTMCell(fea_map_num1)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=init_state, dtype=tf.float32)
            #output_rnn: [batch_size, xlen-filter_width, fea_map_num1]
            reshape_width = (self.x_pixels-filter_width+1) * fea_map_num1
            output = tf.reshape(output_rnn,[-1, reshape_width])
            w_out = tf.get_variable('w', [reshape_width, self.n_actions],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b_out = tf.get_variable('b', [1, self.n_actions],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            out = tf.matmul(output, w_out) + b_out
        return out #, final_states

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.x_pixels, self.y_pixels, self.n_channels], name='s')  # input
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, normrand_initializer, const_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = self._build_layers(
                self.s, c_names, normrand_initializer, const_initializer)

        with tf.variable_scope('loss'):
            # TODO:see loss again
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.x_pixels, self.y_pixels, self.n_channels], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = self._build_layers(
                self.s_, c_names, normrand_initializer, const_initializer)

    def reshape(self, observation):
        return observation.reshape(-1)

    def restore(self, observation):
        assert len(observation.shape) in [1,2], 'check the shape of observation'
        if len(observation.shape) == 1:
            observation = observation.reshape([1, self.x_pixels, self.y_pixels, self.n_channels])
        elif len(observation.shape) == 2:
            observation = observation.reshape([-1, self.x_pixels, self.y_pixels, self.n_channels])
        observation = observation.astype(np.float32)/2
        return observation


class DoubleWDQRCNN(DoubleDQRCNN):
    def _build_layers(self, s, c_names, normrand_initializer, const_initializer):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

        with tf.variable_scope('conv1_1'): #s: [batchsize, xlen, ylen, n_channels]
            fea_map_num1_1 = 16
            filter_width1=3
            w1_1 = tf.get_variable('w1_1', [filter_width1, self.y_pixels, self.n_channels, fea_map_num1_1],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b1_1 = tf.get_variable('b1_1', [1, fea_map_num1_1],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            conv1_1 = tf.nn.relu(conv2d(s, w1_1) + b1_1)  #conv1_1: [batchsize ,xlen-filter_width1+1, ylen=1, fea_map_num1_1]
            # if self.is_train:
                # conv1_1 = tf.nn.dropout(conv1_1, keep_prob=0.8)

        with tf.variable_scope('conv1_2'):
            fea_map_num1_2 = 32
            w1_2 = tf.get_variable('w1_2', [filter_width1, 1, fea_map_num1_1, fea_map_num1_2],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b1_2 = tf.get_variable('b1_2', [1, fea_map_num1_2],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            conv1_2 = tf.nn.relu(conv2d(conv1_1, w1_2) + b1_2)  #conv1_2: [batchsize ,xlen-2*(filter_width1+1), ylen=1, fea_map_num1_2]
            # if self.is_train:
                # conv1_2 = tf.nn.dropout(conv1_2, keep_prob=0.8)

        with tf.variable_scope('conv2'): #s: [batchsize, xlen, ylen, n_channels]
            fea_map_num2 = 32
            filter_width2 = 5
            w2 = tf.get_variable('w2', [filter_width2, self.y_pixels, self.n_channels, fea_map_num2],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, fea_map_num2],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            conv2 = tf.nn.relu(conv2d(s, w2) + b2)  #conv2: [batchsize ,xlen-filter_width2+1, ylen=1, fea_map_num2]
            # if self.is_train:
                # conv2 = tf.nn.dropout(conv2, keep_prob=0.8)

        with tf.variable_scope('conv3'):
            input3 = tf.concat([conv1_2, conv2], 3) #input3: [batchsize, xlen-filter_width2+1, ylen=1, fea_map_num1_2+fea_map_num2]
            feature_num3 = fea_map_num1_2 + fea_map_num2
            rnn_cell_num = 128
            w3 = tf.get_variable('w3', [1, 1, feature_num3, rnn_cell_num],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b3 = tf.get_variable('b3', [1, rnn_cell_num],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            conv3 = tf.nn.relu(conv2d(input3, w3) + b3)    #conv3: [batchsize, xlen-filter_width2+1, feature_num3]

        with tf.variable_scope('LSTM'):
            input_rnn = tf.squeeze(conv3, [2,]) # input_rnn: [batch_size, xlen-filter_width2+1, feature_num3]
            batch_size = tf.shape(input_rnn)[0]
            n_steps = input_rnn.shape[1].value
            # input_rnn = tf.transpose(input_rnn, [1,0,2]) # this and next two lines for input of static_bidirectional_rnn
            # input_rnn = tf.reshape(input_rnn, [-1, rnn_cell_num])
            # input_rnn = tf.split(input_rnn, n_steps)
            cell_fw = tf.contrib.rnn.BasicLSTMCell(rnn_cell_num)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(rnn_cell_num)
            init_fw = cell_fw.zero_state(batch_size,dtype=tf.float32)
            # if self.is_train:
                # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=0.8)
                # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=0.8)
            ##output_rnn, out_state_fw, out_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, input_rnn, dtype=tf.float32)
            # output_rnn, out_state_fw = tf.contrib.rnn.static_rnn(cell_fw, input_rnn, initial_state=init_fw, dtype=tf.float32)
            output_rnn,final_states = tf.nn.dynamic_rnn(cell_fw, input_rnn,initial_state=init_fw, dtype=tf.float32)
            #TODO: try to use only out_states
            reshape_width = n_steps * rnn_cell_num
            output = tf.reshape(output_rnn,[-1, reshape_width])
            w_out = tf.get_variable('w', [reshape_width, self.n_actions],
                        dtype=tf.float32, initializer=normrand_initializer, collections=c_names)
            b_out = tf.get_variable('b', [1, self.n_actions],
                        dtype=tf.float32, initializer=const_initializer, collections=c_names)
            out = tf.matmul(output, w_out) + b_out
        return out #, final_states


