import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import gym
import csv
import pandas as pd
from pathlib import Path

from pole.models import Agent
from pole.data import Memory    # I need this to open memory file with pickle

seed = 123
random.seed(seed)
np.random.seed(seed)

lr = 1e-3   # 1e-4

i_start = 1000

model_type = 'dueling'  # '10_10'

b_save = True  # boolean to save weights and network
name_save = '{model_name}_{i}'
b_load = True
if b_load:
    model_name = f'{model_type}_{i_start-1}'
    # model_name = 'try{}'.format(i_start - 1)  # previous/last one

b_train = True

b_pretrain = False  # Doesn't seem useful ATM
if b_pretrain:
    epoch_start = 25000 #5000
    n_pretrain = 5000

MINI_BATCH_SIZE = 16    # 4
BATCH_SIZE = 32  # 32
GAMMA = 0.99

n_tries = 1000


def get_reward(reward, done, b_end, state_next):

    if done and not b_end:
        # print('previous reward: ', reward)
        reward = -1.
    else:
        reward = 1.   # reward

    # Probably bad idea
    # # Depends on application! but we want to propose a state that is around 0
    # # Should be lower than the q values of next!
    # mu = 1e-5
    # distr = np.array([0.6911251, 0.45466556, 0.02490752, 0.20573978])  # from agent.distr...
    # l1 = -mu*np.sum(np.abs(state_next-0)/distr)
    # print(l1)
    # reward += l1

    return reward

def get_data():
    # probably should save (previous) data somewhere
    return -1 # TODO


def main():
    # TODO switch to v1???
    env = gym.make('CartPole-v1')   # wondering what difference is between v0 and v1
    env.seed(seed)
    env.reset()

    env._max_episode_steps

    # env.observation_space is a Box of size 4
    observation_space = env.observation_space.shape[0]
    # env.observation_space is a Discrite of size 2 (can be 0 or 1)
    action_space = env.action_space.n

    # TODO, our agent with input size and amount of outputs.
    if b_load:
        with open('memory.pkl', 'rb') as f:
            memory = pickle.load(f)
    else:
        memory = None


    agent = Agent(observation_space, action_space, memory=memory,
                  batch_size=BATCH_SIZE, mini_batch_size=MINI_BATCH_SIZE, gamma=GAMMA, lr=lr,
                  model_type=model_type)

    # loading pretrained model. but only if we aren't pretraining
    if b_load and not b_pretrain:
        agent.load_model(model_name)

    # pretrain
    if b_pretrain:   # TODO
        if epoch_start != 0:    # TODO for continue pretraining
            agent.model.load_weights('weights/' + model_type + '_pretrain_e{}.h5'.format(epoch_start))

        agent.pretrain(model_type, n_pretrain=n_pretrain, epoch_start=epoch_start)

    # open and plot the previous run
    if 0:
        df = pd.read_csv('logGood01.csv')
        df.plot(x='iteration', y='n alive')
        df.plot(x='iteration', y='loss')
        print('Close to continue.')
        plt.show(True)

    plt.figure()

    h1, = plt.plot([], [])
    plt.xlabel('game i')
    plt.ylabel('score (time alive with max {})'.format(env._max_episode_steps))

    plt.figure()

    h2, = plt.plot([], [])
    plt.xlabel('game i')
    plt.ylabel('average loss')

    # plt.show(True)  # TODO ion? false/true...

    def update_line(h1, new_data_x, new_data_y):
        h1.set_xdata(np.append(h1.get_xdata(), new_data_x))
        h1.set_ydata(np.append(h1.get_ydata(), new_data_y))
        h1.axes.relim()   # TODO uncomment or not?
        h1.axes.autoscale_view()
        plt.pause(0.05)

    # Generate the file
    if not b_load:
        with open('log.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["iteration", "n alive", "loss"])
        csvFile.close()

    if b_train:
        # amount of restarting
        for i_try in range(i_start, i_start+n_tries):
            state = env.reset()

            n_live = 0
            loss = []
            while True:
                env.render()

                # random
                action = env.action_space.sample()

                # non-random    # single batch
                action = agent.act(np.array([state]))

                def get_b_end():
                    b_end = env._max_episode_steps <= env._elapsed_steps+1
                    return b_end
                b_end = get_b_end()

                """
                state (array of values)
                reward: value that represents "reward"
                done: boolean
                info: extra info
                """
                state_next, reward, done, info = env.step(action)  # take a random action

                # update reward (wrong for "dying")
                reward = get_reward(reward, done, b_end, state_next)
                if b_end:
                    done = False    # at least for pole as end reward is not "defined"/can't know when you're going to win, unless state of how many epochs you live are given.

                # save
                agent.remember(state, action, reward, state_next, done)
                loss.append(agent.experience_replay(i_try))   # learn

                # update for next epoch
                state = state_next  # update state with the new state
                n_live += 1

                if done or b_end:
                    break

            print('try {}: n={}'.format(i_try, n_live))

            update_line(h1, i_try, n_live)

            update_line(h2, i_try, np.mean(loss))

            csvData = [i_try, n_live, np.mean(loss)]

            with open(r'log.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(csvData)
            csvFile.close()

            if b_save:
                agent.save_model(name_save.format(**{'model_name': model_type, 'i': i_try}))
                agent.save_memory()

    else:
        # # amount of restarting
        for i_try in range(i_start, i_start + n_tries):
            state = env.reset()

            n_live = 0
            # loss = []
            while True:
                env.render()

                # non-random    # single batch
                action = agent.act(np.array([state]))

                def get_b_end():
                    b_end = env._max_episode_steps <= env._elapsed_steps + 1
                    return b_end

                b_end = get_b_end()

                """
                state (array of values)
                reward: value that represents "reward"
                done: boolean
                info: extra info
                """
                state_next, reward, done, info = env.step(action)  # take a random action

                # update reward (wrong for "dying")
                reward = get_reward(reward, done, b_end, state_next)
                if b_end:
                    done = False  # at least for pole as end reward is not "defined"/can't know when you're going to win, unless state of how many epochs you live are given.

                # # save
                # agent.remember(state, action, reward, state_next, done)
                # loss.append(agent.experience_replay())  # learn

                # update for next epoch
                state = state_next  # update state with the new state
                n_live += 1

                if done or b_end:
                    break

            print('try {}: n={}'.format(i_try, n_live))

            update_line(h1, i_try, n_live)

            # update_line(h2, i_try, np.mean(loss))

            # csvData = [i_try, n_live, np.mean(loss)]

            # with open(r'log.csv', 'a') as csvFile:
            #     writer = csv.writer(csvFile)
            #     writer.writerow(csvData)
            # csvFile.close()

            # if b_save:
            #     agent.save_model('try{}'.format(i_try))
            #     agent.save_memory()


    env.close()

    plt.show(True)

    return 1


if __name__ == '__main__':
    main()
