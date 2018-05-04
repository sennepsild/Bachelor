import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import PongEnviroment as env
import matplotlib.pyplot as plt

env = env.Pong()

gamma = 0.99 #discount rate

def discount_rewards(r):
    discountedRewards = np.zeros_like(r)
    currentReward = 0
    for t in reversed(range(0,r.size)):
            currentReward = currentReward*gamma+r[t]
            discountedRewards[t] = currentReward
    return discountedRewards

class agent():
    def __init__(self,learningRate,s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hiddenLayer = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hiddenLayer,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)


        self.theReward = tf.placeholder(shape=[None],dtype= tf.float32)
        self.theAction = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0,tf.shape(self.output)[0])* tf.shape(self.output)[1]+self.theAction
        self.corresponding_outputs = tf.gather(tf.reshape(self.output,[-1]),self.indexes)

        self.loss = tf.reduce_mean(tf.log(self.corresponding_outputs)*self.theReward)

        trainVars = tf.trainable_variables()
        self.gradients = []
        for idx, var in enumerate(trainVars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradients.append(placeholder)

        self.gradients = tf.gradients(self.loss, trainVars)

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradients, trainVars))
#Training of the agent
tf.reset_default_graph()  # Clear the Tensorflow graph.

myAgent = agent(learningRate=1e-2, s_size=4, a_size=2, h_size=8)  # Load the agent.

total_episodes = 5000  # Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d == True:
                # Update the network.
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.theReward: ep_history[:, 2],
                             myAgent.theAction: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradients, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_lenght.append(j)
                break


                # Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1