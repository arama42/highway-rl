import gymnasium as gym
import numpy as np
import json
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
        # config = json.load(open('config.json', 'r'))
        # env.configure(config)

        # convert environment to finite MDP
        self.mdp = env.unwrapped.to_finite_mdp()
        self.env = env

        self.num_states = self.mdp.transition.shape[0]
        self.num_actions = self.env.action_space.n

        self.values = np.zeros(self.mdp.transition.shape[0])     # initialize v(s) arbitrarily for each state
        self.policy = np.zeros(self.mdp.transition.shape[0])
        self.EPISODES = 10
        self.gamma = 0.9
        self.epsilon = 1e-6     # a small number
        self.delta = 0

    def value_iteration(self):

        for i in range(self.EPISODES):
            print(f"delta {self.delta} epsilon {self.epsilon}")
            prev_values = self.values.copy()

            for state in range(self.num_states):
                q_sa = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    next_state = self.mdp.transition[state, action]
                    r = self.mdp.reward[state, action]
                    print(f"next_state {next_state} reward {r}")
                    q_sa[action] += (r + self.gamma * prev_values[next_state])

                self.values[state] = max(q_sa)
                self.policy[state] = np.argmax(q_sa)
                print(f"updated value {max(q_sa)} updated policy {np.argmax(q_sa)}")

                self.delta = max(self.delta, abs(prev_values[state] - self.values[state]))
                if self.delta >= self.epsilon:
                    print('Problem converged at iteration %d.' % (i + 1))
                    break

    @staticmethod
    def is_finite_mdp(env):
        try:
            finite_mdp = __import__("finite_mdp.envs.finite_mdp_env")
            if isinstance(env.unwrapped, finite_mdp.envs.finite_mdp_env.FiniteMDPEnv):
                return True
        except (ModuleNotFoundError, TypeError):
            return False

    def policy_evaluation(self):

        for episode in range(self.EPISODES):
            s, _ = self.env.reset()
            done = False
            steps_survived = 0
            total_reward = 0
            state = np.random.choice(self.num_states)

            while not done:
                print(f"state: {state}")
                action = int(self.policy[state])
                print(f"action: {action}")
                next_state, reward, done, _ = self.mdp.step(action)
                state = next_state      # .reshape([1, self.num_states[0], self.num_states[1]])  #remove reshape

                steps_survived += 1
                total_reward += reward

                if done:
                    print('episode: {}/{}, steps survived: {}, total reward: {}'.format(episode + 1,
                                                                                        self.EPISODES,
                                                                                        steps_survived,
                                                                                        total_reward))


if __name__ == '__main__':

    env = gym.make("highway-v0", render_mode="rgb_array")
    agent = ValueIterationAgent(env)

    print(f"env {agent.env}")
    print(f"mdp {agent.mdp}")
    print(f"mdp mode {agent.mdp.mode}")
    print(f"transition shape {agent.mdp.transition.shape}")
    print(f"reward shape {agent.mdp.reward.shape}")

    print(f"num_states {agent.num_states}")
    print(f"num_actions {agent.num_actions}")
    print(f"states {agent.mdp.transition}")
    print(f"actions {agent.env.action_space}")

    # create value iteration agent
    agent = ValueIterationAgent(env)
    agent.value_iteration()
    agent.policy_evaluation()
