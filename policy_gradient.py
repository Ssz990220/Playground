import gym
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.optimizers import Adam
from collections import deque


class PGAgent:
    def __init__(self, state_size, action_space_size, lr=0.01, r_decay=0.95):
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.lr = lr
        self.r_decay = r_decay

    def _model(self):
        model = tf.keras.models.Sequential()
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_shape=(self.state_size)))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space_size, activation='softmax'))
        opt = Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def acting(self):
        if
    def store_transition(self):

    def learn(self):
