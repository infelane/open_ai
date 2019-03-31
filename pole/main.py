import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

import gym
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import adam
from keras import backend as K
from keras.regularizers import l2

seed = 123
random.seed(seed)
np.random.seed(seed)

lr = 1e-3

b_save = True  # boolean to save weights and network

MINI_BATCH_SIZE = 1
BATCH_SIZE = 32  # 32
GAMMA = 0.9

n_tries = 500

class Agent(object):
    def __init__(self, observation_space, action_space, memory=None):
        """

        :param observation_space: length/shape of input
        :param action_space: length of outputs
        """
        l2_val = 0.     # 1e-4

        x = Input(shape=(observation_space,))
        l = Dense(10, activation='elu', kernel_regularizer=l2(l2_val))(x)
        y = Dense(action_space)(l)

        self.model = Model(x, y)

        def masked_mse(y_true, y_pred):

            mask_true = K.cast(K.not_equal(y_true, y_pred), K.floatx())
            masked_squared_error = K.square(mask_true * (y_true - y_pred))
            m_mse = K.sum(masked_squared_error, axis=-1)  # we expect only one value different
            return m_mse

        # TODO right cost function (the sparse mse thingy)
        self.model.compile(adam(lr=lr), loss=masked_mse)

        if memory is not None:
            self.memory = memory
        else:
            self.memory = []

    def act(self, state):

        y = self.model.predict(state)

        print('q values:', y)

        # TODO argmax or according to probability?
        return np.argmax(y)

    def experience_replay(self):

        # fit a batch at once

        # TODO probs get batch per minibatch! in for loop.
        batch = random.sample(self.memory, BATCH_SIZE)
        # batch_vali = random.sample(self.memory, BATCH_SIZE) # TODO

        loss = self.train(batch, BATCH_SIZE//MINI_BATCH_SIZE,
                   MINI_BATCH_SIZE
                   )

        return np.mean(loss)

    def remember(self, state, action, reward, state_next, done):

        save = [state, action, reward, state_next, done]

        N_MAX = 1000 # should be way larger
        if len(self.memory) == N_MAX:
            self.memory[np.random.randint(N_MAX)] = save
        elif len(self.memory) > N_MAX:
            raise ValueError(len(self.memory), 'should not be larger than', N_MAX)
        else:
            self.memory.append(save)

        return 1

    def pretrain(self):

        n_data = len(self.memory)
        train_split = 0.8
        n_train = int(n_data*train_split)

        data_train = self.memory[:n_train]
        data_val = self.memory[:n_train]

        for i in range(10):

            batch = random.sample(data_train, BATCH_SIZE)
            batch_val = random.sample(data_val, BATCH_SIZE)

            loss = self.train(batch, epochs=1, batch_val=batch_val)

            print(loss)

        return loss

    def train(self, batch, epochs=0, mini_batch_size=1, batch_val=None):
        """
        Trains the network
        :param epochs: can also be seen as BATCH_SIZE
        :param mini_batch_size:
        :return:
        """
        # TODO: train network on all the memory etc! with test set (probs split before training)
        # TODO: we have all the experience so it's probably okay to train the net and than just try to run the thing without needing to train at the moment.
        # TODO add validation

        len(batch)

        if len(self.memory) < epochs:
            return 0    # no learning while memory does not have enough experience

        loss = []
        loss_val = []

        # # Outdated
        # for state, action, reward, state_next, terminal in batch:
        #     q_update = reward
        #     if not terminal:
        #         q_predict_next = self.model.predict(np.array([state_next]))[0]
        #         q_update = (reward + GAMMA * np.max(q_predict_next, axis=-1))
        #     q_values = self.model.predict(np.array([
        #                                                state]))  # Is probably already calculated before, however with double model, this one is different (only updated sometimes)
        #     q_values[0][action] = q_update
        #
        #     # Would prefer to not do SGD
        #     hist = self.model.fit(np.array([state]), q_values, verbose=0)
        #     loss.append(hist.history['loss'])

        def get_q():
            lst_q_update = np.copy(lst_reward)

            # only when not terminal
            b_not_terminal = np.logical_not(lst_terminal)
            lst_q_predict_next = self.model.predict(lst_state_next)
            lst_q_update[b_not_terminal] = (lst_reward + GAMMA * np.max(lst_q_predict_next, axis=-1))[b_not_terminal]

            lst_q_values = self.model.predict(lst_state)
            lst_q_values[:, lst_action] = lst_q_update

            return lst_q_values

        for i in range(epochs):
            minibatch = batch[i*MINI_BATCH_SIZE:(i+1)*MINI_BATCH_SIZE]

            lst_state, lst_action, lst_reward, lst_state_next, lst_terminal = [np.array(a) for a in zip(*minibatch)]

            lst_q_values = get_q()

            if batch_val is not None:
                lst_state_val =     # TODO
                validation_data = [lst_state_val]
            else:
                validation_data = None

            hist = self.model.fit(lst_state, lst_q_values, validation_data=validation_data,
                                  verbose=0)
            loss.append(hist.history['loss'])

            if batch_val is not None:
                loss_val.append(hist.history['loss_val'])

        if batch_val is None:
            return loss
        else:
            return loss, loss_val

    def save_model(self, name):
        self.model.save_weights(name + '.h5')

    def load_model(self, name):
        self.model.load_weights(name + '.h5')

    def save_memory(self):

        with open('memory.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.memory, f)

        return 1   # TODO


def get_reward(reward, done, b_end):

    if done and not b_end:
        # print('previous reward: ', reward)
        return -1.
    else:
        return 0.   # reward

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
    if 1:
        with open('memory.pkl', 'rb') as f:
            memory = pickle.load(f)
    else:
        memory = None
    agent = Agent(observation_space, action_space, memory=memory)

    # loading pretrained model.
    if 1:
        agent.load_model('try150')

    # pretrain
    if 1:
        agent.pretrain()

    fig = plt.figure()

    h1, = plt.plot([], [])
    plt.xlabel('game i')
    plt.ylabel('score (time alive with max 200)')

    fig = plt.figure()

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

    # amount of restarting
    for i_try in range(n_tries):
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
            reward = get_reward(reward, done, b_end)
            if b_end:
                done = False    # at least for pole as end reward is not "defined"/can't know when you're going to win, unless state of how many epochs you live are given.

            # save
            agent.remember(state, action, reward, state_next, done)
            loss.append(agent.experience_replay())   # learn

            # update for next epoch
            state = state_next  # update state with the new state
            n_live += 1

            if done or b_end:
                break

        print('try {}: n={}'.format(i_try, n_live))

        update_line(h1, i_try, n_live)

        update_line(h2, i_try, np.mean(loss))

        if b_save:
            agent.save_model('try{}'.format(i_try))
            agent.save_memory()

        # plt.show()

    env.close()

    return 1


if __name__ == '__main__':
    main()
