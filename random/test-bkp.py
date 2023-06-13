import gymnasium as gym
from gymnasium import spaces
from highway_env.envs import HighwayEnv
import numpy as np
import json
import sys
import pprint


def main():

    config = json.load(open('../config.json', 'r'))
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure(config)

    while True:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action = env.action_type.actions_indexes["IDLE"]
            obs, reward, done, truncated, info = env.step(action)
            env.render()


class ValueIterationAgent:
    def __init__(self, env):
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.value_function = np.zeros(np.exp(2, 25))
        self.policy = np.zeros(np.exp(2, 25))

        self.gamma = 0.9
        self.epsilon = 1e-6

    def value_iteration(self, gamma=0.9, epsilon=1e-6):
            delta = 0
            for state in range(self.num_states):
                max_value = float('-inf')
                best_action = None
                for action in range(self.num_actions):
                    value = self.calculate_value(state, action, gamma)
                    if value > max_value:
                        max_value = value
                        best_action = action
                delta = max(delta, abs(self.value_function[state] - max_value))
                self.value_function[state] = max_value
                self.policy[state] = best_action
            if delta < epsilon:
                break

    def calculate_value(self, state, action, gamma):
        value = 0
        for prob, next_state, reward, _ in self.env.P[state][action]:
            value += prob * (reward + gamma * self.value_function[next_state])
        return value

    def run(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy[state]
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                self.env.render()


if __name__ == '__main__':

    # Create the HighwayEnv environment
    env = HighwayEnv()

    # Create the ValueIterationAgent
    agent = ValueIterationAgent(env)

    # Run value iteration
    # agent.value_iteration()

    print(f"num_states {agent.env.observation_space.shape[0]}")
    print(f"num_actions {agent.env.action_space.n}")
    print(f"num_actions {agent.env.observation_space}")

    # Run the agent for a specified number of episodes
    agent.run(num_episodes=20)
