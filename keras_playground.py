from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import gym

env = gym.make('CartPole-v0')

# create memory pool
memory_state = np.zeros((2000,4))
memory_action = np.zeros((2000,1))
memory = deque(maxlen=2000)
for i in range(0, 2000):
    if i == 0:
        cur_state = env.reset().reshape(1, 4)
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    next_state = next_state.reshape(1, 4)
    memory.append((cur_state, reward, action, next_state))
    memory_state[i] = cur_state
    memory_action[i] = action
    if done:
        cur_state = env.reset().reshape(1,4)

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss = 'mean_squared_error', optimizer = Adam(lr=0.001))

model.fit(memory_state, memory_action, epochs = 10, batch_size=32)