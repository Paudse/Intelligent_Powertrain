"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import re

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            vehicle_name,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.vehicle_name = vehicle_name
        self.lr = learning_rate
        # print(learning_rate)
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.all_keep_prob = 1 # for dropout

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # print(e_params)
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        # if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []    

        # saver = tf.train.Saver()
        self.saver=tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())


        try:
            model_iteration_cycle_file = tf.train.latest_checkpoint('check_point/'+self.vehicle_name+'/')
            vehicle_name_num = re.findall(r"\d+\.?\d*",self.vehicle_name)
            iteration_cycle = re.findall(r"\d+\.?\d*",model_iteration_cycle_file)
            # print(iteration_cycle)
            self.last_iteration = int(iteration_cycle[len(vehicle_name_num)])
            # self.last_max_cycle_reward = iteration_cycle[len(vehicle_name_num)+2]
            print('last_learning_iteration:', self.last_iteration)
            # self.cycle_c = float(iteration_cycle[1])
            self.saver.restore(self.sess,model_iteration_cycle_file)
            print('restore previous NN data')
        except:
            self.last_iteration = 0
            print('no previous NN data, start new')

    def return_last_iteration(self):
        return self.last_iteration

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, 50,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                w1 = tf.nn.dropout(w1, keep_prob=self.all_keep_prob)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                b1 = tf.nn.dropout(b1, keep_prob=self.all_keep_prob)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                w2 = tf.nn.dropout(w2, keep_prob=self.all_keep_prob)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                b2 = tf.nn.dropout(b2, keep_prob=self.all_keep_prob)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
                # print(b2.eval)

            # # 3rd layer. collections is used later when assign to target net
            # with tf.variable_scope('l3'):
            #     w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
            #     w3 = tf.nn.dropout(w3, keep_prob=self.all_keep_prob)
            #     b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
            #     b3 = tf.nn.dropout(b3, keep_prob=self.all_keep_prob)
            #     l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
            #     # print(b2.eval)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                w3 = tf.nn.dropout(w3, keep_prob=self.all_keep_prob)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                b3 = tf.nn.dropout(b3, keep_prob=self.all_keep_prob)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                w1 = tf.nn.dropout(w1, keep_prob=self.all_keep_prob)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                b1 = tf.nn.dropout(b1, keep_prob=self.all_keep_prob)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                w2 = tf.nn.dropout(w2, keep_prob=self.all_keep_prob)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                b2 = tf.nn.dropout(b2, keep_prob=self.all_keep_prob)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
                # print(b2.eval)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                w3 = tf.nn.dropout(w3, keep_prob=self.all_keep_prob)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                b3 = tf.nn.dropout(b3, keep_prob=self.all_keep_prob)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            # print(self.s)
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self,save_step,step):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # print(self.learn_step_counter)
        # if step % save_step == 0:
        #     self.saver.save(self.sess,self.vehicle_name+'_ckpt/mnist.ckpt',global_step=step)
        #     print('save parameters')

    def save_learning(self,cycle_round,max_cycle_reward,cycle_name):
        # self.saver.save(self.sess,self.vehicle_name+'_ckpt/'+str(cycle_name)+'/reward-'+str(round(max_cycle_reward,2))+'.ckpt')
        self.saver.save(self.sess,'check_point/'+self.vehicle_name+'/iteration_'+str(self.last_iteration+cycle_round)+'_'+str(cycle_name)+'_reward-'+str(round(max_cycle_reward,2))+'.ckpt')
        # self.saver.save(self.sess,self.vehicle_name+'_ckpt_cycle_iteration'+str(self.cycle_iteration)+'.ckpt')
        # self.saver.save(self.sess,self.vehicle_name+'_ckpt/'+self.tj_name+'_reward-'+str(round(max_cycle_reward,2))+'.ckpt')
        # self.saver.save(self.sess,self.vehicle_name+'_ckpt/mnist_reward-'+str(round(max_cycle_reward,2))+'.ckpt',global_step=tj_round)
        print('save parameters')

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Score')
        plt.xlabel('training steps')
        plt.show()
