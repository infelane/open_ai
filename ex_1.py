""" from open_ai gym example"""

import gym
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam
from keras.losses import mse
import keras.backend as K
import matplotlib.pyplot as plt
from link_to_keras_contrib_lameeus.keras_contrib.callbacks.dead_relu_detector import DeadReluDetector


class Foo():
    def __init__(self, dead_or_alive):
        self.dead_or_alive = dead_or_alive
        
        self.folder_name = '/home/lameeus/data/personal/cartpole/'
        in_obs_prev = Input((4,), name='observations')
        layer1 = Dense(10, activation='relu')(in_obs_prev)
        out_act = Dense(2, activation='softmax')(layer1)
        self.model_player = Model(in_obs_prev, out_act, name = 'act_model')

        in_act = Input((2,), name='activation')
        layer2 = Concatenate()([in_obs_prev, in_act])
        layer2 = Dense(10, activation='relu')(layer2)
        out_obs_next = Dense(4, activation='linear', name='next')(layer2)
        self.model_next = Model([in_obs_prev, in_act], out_obs_next, name = 'next_model')
        self.model_next.compile(optimizer=Adam(lr=1e-3), loss='mse')
  
        in_obs_next = Input((4,), name='next_observation')
        layer3 = Dense(10, activation='relu')(in_obs_next)
        if self.dead_or_alive:
            out_epochs = Dense(1, activation='sigmoid', name='when_dead')(layer3)
        else:
            out_epochs = Dense(1, activation='linear', name='when_dead')(layer3)
        
        self.model_dead = Model(in_obs_next, out_epochs, name='when_dead_model')
        self.model_dead.compile(optimizer=Adam(lr=1e-3), loss='mse')

        self.model_dead.trainable = False
        self.model_next.trainable = False
        self.model_player.trainable = True
        
        if 1:
            out_epochs_tot = self.model_dead(self.model_next([in_obs_prev, self.model_player(in_obs_prev)]))
        else:
            out_epochs_tot = out_epochs(out_obs_next([in_obs_prev, out_act(in_obs_prev)]))
        
        self.model_total = Model(in_obs_prev, out_epochs_tot)

        def loss(y_true, y_pred):
            # # Does not work!
            # return -K.log(y_pred)  # you want y_pred as big as possible
            return mse(y_true, y_pred)
        
        self.model_total.compile(optimizer=Adam(lr=1e-3), loss=loss)

        # self.model_dead.trainable = True
        # self.model_next.trainable = True

        # for a in self.model_total.layers:
        #     name = a.name
        #     print(a.name)
        #     print(a.trainable)
        #     if 'when_dead' in name or 'next' in name:
        #         print('idk')
        #         a.trainable = False
        
        out_alls = [self.model_next([in_obs_next, in_act]), self.model_dead(in_obs_next), self.model_total(in_obs_prev)]
        self.modal_all = Model([in_obs_prev, in_act, in_obs_next], out_alls)
        self.modal_all.compile(optimizer=Adam(lr=1e-4), loss=['mse', 'mse', loss])

        print('act:')
        self.model_player.summary()
        
        print('next:')
        self.model_next.summary()
        
        print('dead:')
        self.model_dead.summary()
        
        print('total:')
        self.model_total.summary()
        
        print('all:')
        self.modal_all.summary()
        
        self.load()
        
    def _path(self, name = 'v0'):
        return self.folder_name + name + 'h5'
        
    def load(self):
        path = self._path()
        self.model_player.load_weights(path)
        self.model_next.load_weights(self._path('next'))
        self.model_dead.load_weights(self._path('dead'))
    
    def save(self):
        path = self._path()
        self.model_player.save_weights(path, overwrite = True)
        self.model_next.save_weights(self._path('next'), overwrite = True)
        self.model_dead.save_weights(self._path('dead'), overwrite=True)
    
    def predict(self, x):
        x = np.reshape(x, newshape=(-1, 4))
        
        y = self.model_player.predict(x)
        action_prop = np.argmax(y)
        # action_prop = 1 if y > 0 else 0
        return action_prop, y
        
    def train_predict(self, obs_prev, act, obs, y_epochs, verbose=1, epochs=1):
        # TODO order of execution!? might be fixed when all trained at same time!!

        if self.dead_or_alive:
            t_goal = y_epochs*0 # the goal is to stay alive
        else:
            # t_goal is now incread by 10%
            t_goal = y_epochs * 1.1
        if 1:
            self.model_dead.fit(obs, y_epochs, epochs=epochs, verbose=verbose)  # although it says untrainable, but it does!
            self.model_next.fit([obs_prev, act], obs, epochs=epochs, verbose=verbose)    # although it says untrainable, but it does!
            self.model_total.fit(obs_prev, t_goal, epochs=epochs, verbose=verbose)      # Trains the act
        
        else:   # all trained at the same time
            self.modal_all.fit([obs_prev, act, obs], [obs, y_epochs, t_goal], epochs=epochs, verbose=verbose)
        
        self.save()


class Main():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.dead_or_alive = True
        self.foo = Foo(self.dead_or_alive)
        
    def run_multiple(self):
        
        # settings
        n_runs = 5  # 20

        # variables to save
        n_over = 0
        t_loop = []
        y_epochs = np.array([])

        self.obs_prev = []
        self.act = []
        self.obs = []
        self.alive = []
        
        for i_episode in range(n_runs):
            max_timesteps, bool_over = self.loop()
        
            t_loop.append(max_timesteps)
            n_over += bool_over
            
            # TODO instead of count_down: alive or dead
            if self.dead_or_alive:
                count_down = np.zeros((max_timesteps+1))
                if not bool_over:   # it died
                    count_down[-1] = 1  # done==dead
            else:
                count_down = np.flip(np.arange(0, max_timesteps + 1), axis=0)
            
            y_epochs = np.append(y_epochs, count_down)

        t_ave = np.mean(t_loop)
        print('t_ave = {}'.format(t_ave))
        print('frac over = {}'.format(n_over / float(n_runs)))
        
        return y_epochs
        
    def loop(self):
        
        # settings
        t_max = 1000  # 100
        self.env._max_episode_steps = t_max+5   # to be sure ;)

        
        observation = self.env.reset()
        t = 0
        for t in range(t_max):
            if 1:
                self.env.render()
            action, act_list = self.foo.predict(observation)
            self.obs_prev.append(observation)
            self.act.append(act_list)
        
            observation, reward, done, info = self.env.step(action)
            """ reward always 1, info always {}"""
            assert reward == 1
            assert info == {}

            self.obs.append(observation)
        
            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                self.alive.append(0)
                return t, 0
            else:
                self.alive.append(1)
    
        # TODO what to do with it
        # loop finished before done
        print('{} is not enough'.format(t_max))
        # TODO still alive
        self.alive.append(1)
        return t, 1


def main():
    m = Main()
    
    for _ in range(10):
        y_epochs = m.run_multiple()
        
        obs = np.stack(m.obs, axis = 0)
        obs_prev = np.stack(m.obs_prev, axis = 0)
        act = np.concatenate(m.act, axis = 0) # act has shape (1, n)
        
        m.foo.train_predict(obs_prev, act, obs, y_epochs, epochs=100, verbose=0)
        m.foo.train_predict(obs_prev, act, obs, y_epochs, verbose=2)
        
    plt.plot([0, 1], [2, 3])
    plt.show()


if __name__ == '__main__':
    main()
