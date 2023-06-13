import gymnasium as gym
import numpy as np
import json
import pprint
import matplotlib.pyplot as plt


class ValueIterationAgent:
    def __init__(self, env):

        # convert environment to finite MDP
        self.config = json.load(open('config.json', 'r'))
        self.env = env
        #self.env.config["observation"]["type"] = "OccupancyGrid"
        env.configure(config)

        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        self.values = np.zeros(np.power(2, 25))     # initialize v(s) arbitrarily for each state
        self.policy = np.zeros(np.power(2, 25), dtype=int)    # initialize policy
        self.EPISODES = 1000
        self.gamma = 0.9
        self.epsilon = 1e-18    # a small number


    def value_iteration(self):
        delta = 0
        for i in range(self.EPISODES):
            # copy previous values
            prev_values = self.values.copy()

            state, _ = self.env.reset()

            print(state)

            # # for state in range(self.num_states):
            #     # calculate Q for each action in the state
            #     q_sa = np.zeros(self.num_actions)
            #     for action in range(self.num_actions):
            #         next_state = self.mdp.next_state(state, action)     # transition[state, action]
            #         r = self.mdp.reward[state, action]
            #         # print(f"state {state} action {action} next_state {next_state} reward {r}")
            #         q_sa[action] += (r + self.gamma * prev_values[next_state])
            #
            #     # print(f"q_sa {q_sa} updated value {max(q_sa)} updated policy {np.argmax(q_sa)}")
            #     self.values[state] = max(q_sa)
            #     self.policy[state] = np.argmax(q_sa)
            #     # delta = max(delta, abs(prev_values[state] - self.values[state]))
            #
            # #if delta <= self.epsilon:
            # if np.max(np.fabs(prev_values - self.values)) <= self.epsilon:
            #     print('Problem converged at iteration %d.' % (i + 1))
            #     break

    def policy_evaluation(self, episodes, initial_state=0):

        for episode in range(episodes):
            done = False
            steps_survived = 0
            total_reward = 0
            state = initial_state  # np.random.choice(self.num_states)

            while not done:
                action = self.policy[state]
                print(f"state: {state} action: {action}")
                next_state, reward, done, _ = self.mdp.step(action)
                state = next_state

                steps_survived += 1
                total_reward += reward

                if done:
                    print('episode: {}/{}, steps survived: {}, total reward: {}'.format(episode + 1,
                                                                                        episodes,
                                                                                        steps_survived,
                                                                                        total_reward))
    def plot(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(self.values)
        ax1.set_title('Optimal Value Function')

        ax2.plot(self.policy)
        ax2.set_title('Optimal Policy')
        plt.show()


if __name__ == '__main__':

    config = json.load(open('config.json', 'r'))
    env = gym.make("highway-v0", render_mode="rgb_array")
    # env.configure(config)

    # create value iteration agent
    agent = ValueIterationAgent(env)
    agent.value_iteration()

    #print(f"num states: {agent.state_size}")

    # print(f"mdp mode: {agent.mdp.mode}")
    # print(f"rewards : {agent.mdp.reward}")
    # print(f"transition shape: {agent.mdp.transition.shape}")



    #agent.value_iteration()

    #agent.plot()
    # agent.policy_evaluation(40, 3)
