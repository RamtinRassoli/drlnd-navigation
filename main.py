#!/usr/bin/env python3
from unityagents import UnityEnvironment
import numpy as np
from model.dqn_agent import Agent
from model.utils import get_config
from collections import deque
import torch
import matplotlib.pyplot as plt
import argparse


def load_environment():
    env = UnityEnvironment(file_name=ENV_FILE)
    brain_name = env.brain_names[0]

    return env, brain_name


def get_env_metadata(env, brain_name):
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    return state_size, action_size


def dqn(env, brain_name, agent, config, train_mode=True, weights_path=None):
    """Deep Q-Learning.

    Params
    ======
        env: Unity Environment
        brain_name: Environment brain that is responsible for deciding the actions of their associated agents
        config: the loaded yaml file that contains the needed hyperparmeters for running and training the agent
    """
    print("start the learning process...")
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = config['train']['eps_start']  # initialize epsilon
    score_threshold = 13.0

    if weights_path:
        weights = torch.load(weights_path)
        agent.qnetwork_local.load_state_dict(weights)
        agent.qnetwork_target.load_state_dict(weights)
        print("Weights loaded from {}".format(weights_path))

    for i_episode in range(1, config['train']['n_episodes'] + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(config['train']['max_t']):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]
            if train_mode:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            # print('\r', 'Iteration', t, 'Score:', score, end='')
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(config['train']['eps_end'], config['train']['eps_decay'] * eps)  # decrease epsilon
        # print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if train_mode:
            if i_episode % 100 == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= score_threshold:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                           np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_{}.pth'.format(i_episode))
                score_threshold += 1
            if i_episode % 500 == 0:
                plot_scores(scores)

    plot_scores(scores)
    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def main(config_path, train_mode=True, weights_path=None):
    """Load the environment, create an agent, and train it.
    """
    config = get_config(config_path)
    env, brain_name = load_environment()
    state_size, action_size = get_env_metadata(env, brain_name)
    agent = Agent(state_size=state_size, action_size=action_size, config=config, seed=0)

    scores = dqn(env, brain_name, agent, config, train_mode, weights_path)

    return scores


if __name__ == "__main__":
    ENV_FILE = "Banana.app"
    parser = argparse.ArgumentParser(description='Banana Navigation Agent')
    parser.add_argument('--train', action="store_true", default=False,
                        help='Train the DQN agent if used, else load the trained weights and play the game')
    parser.add_argument('--weights', action="store", dest="path", type=str,
                        help='path of .pth file with the trained weights')
    args = parser.parse_args()
    print("Train_mode: {}".format(args.train))

    main("model/config.yaml", train_mode=args.train, weights_path=args.path)
