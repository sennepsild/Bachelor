import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
import tensorflow as tf

import gym

env = gym.make('CartPole-v0')
env.reset()


n_inputs = 4
n_hidden = 50

n_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
hidden2 = tf.layers.dense(hidden, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
hidden3 = tf.layers.dense(hidden2, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)


logits = tf.layers.dense(hidden3, n_outputs, kernel_initializer=initializer)

outputs = tf.nn.sigmoid(logits)
#outputs = tf.nn.softmax(logits)

p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
#print(action)
#action =tf.nn.softmax(outputs)

y = 1. - tf.to_float(action)

learning_rate = 0.01

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    total_rewards = 0
    for step in reversed(range(len(rewards))):
        total_rewards = rewards[step] + total_rewards * discount_rate
        discounted_rewards[step] = total_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


n_iterations = 250
n_max_steps = 200
n_games_per_update = 10
save_interations = 10
discount_rate = 0.95
plot_reward = []

iteration_mean = 10
converge = False
converge_TimeStep = 0
numberOfIterationsAbove190 = 0

with tf.Session() as sess:
    init.run()

    for iteration in range(n_iterations):
        rewards_per_tengame = 0
        all_rewards = []
        all_gradients = []

        for game in range(n_games_per_update):
            total_reward = 0
            current_rewards = []
            current_gradients = []

            obs = env.reset()
            for step in range(n_max_steps):
                if (converge):
                    env.render()
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                #print(action_val)
                #if action_val[0][0]
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                total_reward += reward
                if done: break

            rewards_per_tengame += total_reward
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}

        plot_reward.append(rewards_per_tengame / n_games_per_update)

        print(iteration)
        if (iteration > iteration_mean):
            currentMean = 0
            for i in range(iteration - iteration_mean, iteration):
                currentMean += plot_reward[i]
            currentMean = currentMean / iteration_mean

            print(currentMean)
            if (currentMean > 190):
                print("we Have Convergence suckers!")
                converge = True
                converge_TimeStep = iteration

        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[grad_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_interations == 0:
            saver.save(sess, "./mypolicy_net_pg.ckpt")


fig = plt.figure()

plt.plot(plot_reward)

fig.suptitle('16 neurons 1 hidden')
plt.xlabel('Iteration')
plt.ylabel('Score')