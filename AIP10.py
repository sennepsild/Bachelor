#import gym
import gym

import numpy as np
import matplotlib.pyplot as plt

from gym.envs.registration import register
register(
    id='MyFrozenLake-v0',
    #entry_point='gym.envs.toy_text:FrozenLakeEnv',
    entry_point='My Machine Learning:MyFrozenLake',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},



)


env = gym.make('MyFrozenLake-v0')



#env = gym.make("")

Q = np.ones([env.observation_space.n, env.action_space.n]) \
    # *-.1
# Q = np.random.rand(env.observation_space.n,env.action_space.n)

lr = 0.8
y = 0.99
num_episodes = 1000

rewardList = []

for i in range(num_episodes):
    s = env.reset()
    rewardEp = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        print(Q)
        env.render()
        # Add some noise to all actions (promote exploration) and choose best action
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Observation (state) , reward , done , info
        s1, r, d, _ = env.step(a)
        # Q(s,a) = Q(s,a) + lr*(reward + discount * Q(s1) - Q(s,a)) -> Update Q-value of state-action pair towards
        # reward + value of next state
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        # Add the reward
        print(r)
        rewardEp += r
        # Update observation (state)
        s = s1
        if d == True:
            break
    rewardList.append(rewardEp)

print("Score over time: " + str(sum(rewardList) / num_episodes))
print(Q)

plt.plot(rewardList)
plt.show()

s = env.reset()
for _ in range(10):
    env.render()
    a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
    env.step(a)  # take a random action
    s1, r, d, _ = env.step(a)