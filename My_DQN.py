import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

lr = 0.95
max_epoch = 1000
max_step_epoch = 200
param_reset_frequency = 20
epsilon = 0.95
memory_buffer_size = 2000
minibatch_size =100


class DQN:
    def __init__(self, state_space_size, action_space_size, lr, gamma_decay, epsilon, env):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = lr
        self.gamma_decay = gamma_decay
        self.epsilon = epsilon
        self.env = env
        self.eval_model = self.eval_model()
        self.policy_model = self.policy_model()
        self.memory_state = np.zeros((memory_buffer_size, self.state_space_size))
        self.memory_action = np.zeros((memory_buffer_size, self.action_space_size))
        self.memory_target = np.zeros((memory_buffer_size, 1))
        self.memory_counter = 0

    def eval_model(self):
        model = models.Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space_size, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model

    def policy_model(self):
        model = models.Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space_size, activation='relu'))
        return model

    def act(self, cur_state):
        if random.random() > self.epsilon:
            action = self.env.action_space.sample()
        else:
            pred = self.eval_model.predict(cur_state)  # rephrase
            action = np.argmax(pred)
        return action

    def learn(self):
        if self.memory_counter < memory_buffer_size:
            return
        index = np.random.choice(self.max_memo_size, minibatch_size)
        training_sample = self.memory_state[index]
        training_target = self.memory_target[index]
        self.eval_model().fit(training_sample, training_target, epochs=1, batch_size=32)

    def restore_memeory(self, state, reward, action_index, next_state, done):
        index = self.memory_counter % memory_buffer_size
        new_action = np.zeros((1, self.action_space_size))
        new_action[0,action_index] = 1
        self.memory_state[index] = state
        self.memory_action[index] = new_action
        if done:
            target = reward
        else:
            Q = np.amax(self.policy_model.predict(next_state))
            target = reward + self.gamma_decay * Q
        self.memory_target[index] = target


def main():
    env = gym.make('CartPole-v0')
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]
    DQN_Network = DQN(state_space_size= state_space_size, action_space_size= action_space_size, lr=0.01, gamma_decay=0.95, epsilon=0.95, env=env)
    for episode in range(0, max_epoch):
        cur_state = env.reset().reshape(1, DQN_Network.state_space_size)
        if episode % 2 == 1:
            weights = DQN_Network.eval_model.get_weights()
            DQN_Network.policy_model.set_weights(weights)

        for step in range(0, max_step_epoch):
            action = DQN_Network.act(cur_state)
            new_state, reward, done, _ = DQN_Network.env.step(action)
            new_state = new_state.reshape(1, DQN_Network.state_space_size)
            DQN_Network.restore_memeory(state=cur_state, reward=reward, action_index=action, done=done, next_state=new_state)
            if done:
                cur_state = env.reset().reshape(1, DQN_Network.state_space_size)
            else:
                cur_state = new_state
        print("Episode ",episode," is completed.")


if __name__ == "__main__":
    main()
