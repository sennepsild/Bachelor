import tensorflow as tf
import numpy as np



import time
#import gym
import PongEnviroment

from tensorflow.contrib.layers import fully_connected
# 1. Specify the neural network architecture
n_inputs = 6
n_hidden = 52
n_outputs = 3
#initializer = tf.contrib.layers.variance_scaling_initializer()
initializer = tf.zeros_initializer();
# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
weights_initializer=initializer)

hidden2 = fully_connected(hidden, n_hidden, activation_fn=tf.nn.elu,
weights_initializer=initializer)



logits = fully_connected(hidden2, n_outputs, activation_fn=None,
weights_initializer=initializer)


outputs = tf.nn.softmax(logits)


outputs2 = tf.add( outputs,[[0.1,0.1,0.1]])

outputsnorm = tf.norm(outputs2)

#outputs = tf.exp(logits)/tf.reduce_sum(tf.exp(logits))


# 3. Select a random action based on the estimated probabilities
#p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(outputs), num_samples=1)
init = tf.global_variables_initializer()

#y = tf.placeholder(tf.float32, shape=[None, 2])
#y = np.array([1,1]).reshape(1,2)

bjarne = tf.to_int32(action)
gee = tf.reshape(bjarne,shape=[])

hugo =[gee]
#hugo = tf.Variable([1],dtype=tf.float32)
#y = tf.constant([[1,1]],dtype=tf.float32)
y =hugo

learning_rate = 0.01
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(dtype= tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)  #ADDED DISCOUNT_RATE
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
            for discounted_rewards in all_discounted_rewards]


n_iterations = 10000
#
n_max_steps = 5000
#
n_games_per_update = 3 #
save_iterations = 10
#
discount_rate = 0.99
#number of training iterations
#max steps per episode
#train the policy every 10 episodes
#save the model every 10 training iterations

env = PongEnviroment.Pong()

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        #obs = env.getInit(True)
        all_rewards = []
# all sequences of raw rewards for each episode
        all_gradients = [] # gradients saved at each step of each episode
        for game in range(n_games_per_update):
            current_rewards = []
# all raw rewards from the current episode
            current_gradients = [] # all gradients from the current episode
            obs = env.getInit(False)
            for step in range(n_max_steps):
                #time.sleep(0.01)
                action_val, gradients_val, outs, saction = sess.run(
                        [action, gradients,outputs,action],
                        feed_dict={X: obs.reshape(1, n_inputs)}) # one obs
                #feed_dict ={y:}
                #print("probabilities")
                #print(outs)
                #print("action")
                #print(saction)
                #print(outputerino)
                obs, reward = env.Game(False,action_val[0][0],0)



                current_rewards.append(reward)
                #env.render()
                current_gradients.append(gradients_val)

            print(np.sum( current_rewards))

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
    # At this point we have run the policy for 10 episodes, and we are
    # ready for a policy update using the algorithm described earlier.
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate) #ADDED DISCOUNT_RATE
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
    # multiply the gradients by the action scores, and compute the mean
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)],
                axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")

