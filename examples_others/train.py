from __future__ import division
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import sgd
import os
import random
from os.path import isfile
from collections import deque
from link_to_keras_contrib_lameeus.keras_contrib.callbacks import dead_relu_detector

NUM_ACTIONS = 2
NUM_STATES = 4
MAX_REPLAY_STATES = 100
BATCH_SIZE = 50 #20
NUM_GAMES_TRAIN = 500   # amount of games you will simulate
JUMP_FPS = 2
WEIGHT_FILE = 'weights.h5'
MAX_EPS = 1000  # maximum epochs to simulate

def create_model(n_inputs, n_outputs):
    model = Sequential([
        Dense(8, batch_input_shape = (None, n_inputs)),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(n_outputs),
        Activation('linear')
    ])
    model.compile('adam', loss = 'mse')
    if isfile(WEIGHT_FILE):
        print("[+] Loaded weights from file")
        model.load_weights(WEIGHT_FILE)
    return model


def main():
    env = gym.make('CartPole-v0')
    # TODO commented
    # env.monitor.start('training', force = True)
    model = create_model(NUM_STATES, NUM_ACTIONS)
    
    replay = []
    
    gamma = 0.99
    # epsilon = 1     # is updated over time
    epsilon = 0.1  # 1 for new training is updated over time
    
    
    def next_action():
        if random.random() < epsilon:
            action = np.random.randint(NUM_ACTIONS)
        else:
            q = model.predict(new_state.reshape(1, NUM_STATES))[0]
            action = np.argmax(q)
        return action
    
    env._max_episode_steps = MAX_EPS+5
    
    for number_game in range(NUM_GAMES_TRAIN):
        new_state = env.reset()
        reward_game = 0
        done = False
        loss = 0
        index_train_per_game = 0
        print('[+] Starting Game ' + str(number_game))
        while not done:
            env.render()
            index_train_per_game += 1
            
            action = next_action()
            
            old_state = new_state
            new_state, reward, done, info = env.step(action)
            reward_game += reward
            replay.append([new_state, reward, action, done, old_state])
            if len(replay) > MAX_REPLAY_STATES: replay.pop(
                np.random.randint(MAX_REPLAY_STATES))  # pop random data point.   (not last one)
            if JUMP_FPS != 1 and index_train_per_game % JUMP_FPS == 0:  # We skip this train, but already add data. Every JUMP_FPS is trained...
                continue  # goes to next loop iteration
            len_mini_batch = min(len(replay), BATCH_SIZE)
            mini_batch = random.sample(replay, len_mini_batch)
            X_train = np.zeros((len_mini_batch, NUM_STATES))
            Y_train = np.zeros((len_mini_batch, NUM_ACTIONS))
            for index_rep in range(len_mini_batch):
                new_rep_state, reward_rep, action_rep, done_rep, old_rep_state = mini_batch[index_rep]
                old_q = model.predict(old_rep_state.reshape(1, NUM_STATES))[
                    0]  # predicted reward of first action when giving the old state
                new_q = model.predict(new_rep_state.reshape(1, NUM_STATES))[
                    0]  # predicted reward of first action when giving the new state
                update_target = np.copy(old_q)
                
                # update_target = old_q*gamma # reduce on its own.
                if done_rep:
                    # update_target[action_rep] = -1  # it died by doing 'action_rep', the target should thus be lowered!
                    
                    # update_target[action_rep] = -1 - gamma * np.sum(np.square(new_q)) / (np.sum(new_q) + 0.001)

                    update_target[action_rep] = 0
                    
                    # update_target[action_rep] = (np.min(new_q)/gamma) - 10
                    # update_target[action_rep] = new_q[action_rep]/gamma - 1
                    
                    # cost = 1
                    #
                    # tot = np.sum(old_q)
                    #
                    # update_target = new_q+cost/NUM_ACTIONS
                    # update_target[action_rep] = gamma*new_q[action_rep] - cost
                    
                    # update_target /= 2  # propose all rest
                    # update_target[action_rep] = -1  # the one we chose was bad!
                    # update_target[action_rep] = -1  # dead
                    # update_target[action_rep] = -1  # dead
                else:
                    """ it did not die by doing this action, reward is increased of doing this action.
                    reward_rep is the reward. 1 + expected cost """
                    # update_target[action_rep] = reward_rep + (gamma * np.max(new_q))
                    # update_target[action_rep] = reward_rep + (gamma * np.max(old_q))    # old q instead of new updating
                    # update_target[action_rep] = 1
                    # update_target[action_rep] = 1
                    # update_target[action_rep] = reward_rep + (gamma * new_q[action_rep])
                    # update_target[action_rep] = reward_rep + gamma*new_q[action_rep]
                    # update_target[action_rep] = new_q[action_rep] + reward_rep
                    # update_target[action_rep] = 1
                    
                    # update_target[action_rep] = reward_rep + np.mean(new_q)*gamma    # has to decrease, otherwise explodes
                    
                    # weighted sum
                    # update_target[action_rep] = reward_rep + gamma * np.sum(np.square(new_q)) / (np.sum(new_q) + 0.001)
                    # TODO include previous weight to. (should converge to a value that it had already!!

                    # update_target[action_rep] = 1

                    update_target[action_rep] = gamma * np.max(old_q) + 1
                    
                    # update_target[action_rep] = reward_rep + np.mean(new_q)*gamma    # has to decrease, otherwise explodes
                    
                # TODO This is not normalized. # DO!
                
                X_train[index_rep] = old_rep_state
                Y_train[index_rep] = update_target
            
            if 0:
                loss += model.train_on_batch(X_train, Y_train)
            else:
                cb = [lambda : dead_relu_detector(X_train)]
                model.fit(X_train, Y_train, epochs=1, verbose=0, callbacks=cb)
                loss += model.test_on_batch(X_train, Y_train)
            if reward_game > MAX_EPS:
                break
        print("[+] End Game {} | Reward {} | Epsilon {:.4f} | TrainPerGame {} | Loss {:.4f} ".format(number_game,
                                                                                                     reward_game,
                                                                                                     epsilon,
                                                                                                     index_train_per_game,
                                                                                                     loss / index_train_per_game * JUMP_FPS))
        if epsilon >= 0.1:
            epsilon -= (1 / (NUM_GAMES_TRAIN))
        
        # epsilon = np.exp(-reward_game)   # chance on random input
        
        if isfile(WEIGHT_FILE):
            os.remove(WEIGHT_FILE)
        model.save_weights(WEIGHT_FILE)
    # gym.upload('training', api_key='')

if __name__ == '__main__':
    main()
