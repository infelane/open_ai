import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

from keras.models import Model, clone_model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import adam
from keras import backend as K
from keras.regularizers import l2
from keras.losses import mse
from keras.callbacks import TensorBoard
import tensorflow as tf

from pole.data import Memory


class Agent(object):
    i_train = 0
    def __init__(self, observation_space, action_space, memory=None,
                 batch_size=32,
                 mini_batch_size=4,
                 gamma=0.99,
                 lr=1e-4,
                 model_type='10',
                 b_double = True,    # double deep q learning
                 C_double = 10,   # time between updating target model
                 ):
        """
        :param observation_space: length/shape of input
        :param action_space: length of outputs
        """
        l2_val = 1e-4     # 1e-4

        # TODO get as extra input. and action space = len(names)
        self.names_action = ['left', 'right']
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.b_double = b_double
        self.C_double = C_double

        x = Input(shape=(observation_space,))

        if model_type == '10':
            l = Dense(10, activation='elu', kernel_regularizer=l2(l2_val))(x)
            y = Dense(action_space, name='q')(l)
        elif model_type == '10_10':
            k = 10
            l = Dense(k, activation='elu', kernel_regularizer=l2(l2_val))(x)
            l = Dense(k, activation='elu', kernel_regularizer=l2(l2_val))(l)
            y = Dense(action_space, name='q')(l)
        elif model_type == 'dueling':
            k = 10
            l = Dense(k, activation='elu', kernel_regularizer=l2(l2_val))(x)
            l = Dense(k, activation='elu', kernel_regularizer=l2(l2_val))(l)
            V = Dense(1, name='V', kernel_regularizer=l2(l2_val))(l)
            A = Dense(action_space, name='A_pre', kernel_regularizer=l2(l2_val))(l)
            A = Lambda(lambda i:  i - K.mean(i, keepdims=True), name='A')(A)
            Q = Lambda(lambda av: av[0] + K.expand_dims(av[1][..., 0], -1),
                       output_shape=(action_space,),
                       name='Q'
                       )([A, V])
            y = Q
        else:
            raise ValueError(model_type, 'unknown model type')

        self.model = Model(x, y)
        if b_double:
            self.model_target = clone_model(self.model)

        # # # does not work (always perfect prediction)
        # x_next = Input(shape=(observation_space,))
        # y_next = self.model(x_next)
        # self.model_dual = Model([x, x_next], [y, y_next])

        def masked_mse(y_true, y_pred):

            mask_true = K.cast(K.not_equal(y_true, y_pred), K.floatx())
            masked_squared_error = K.square(mask_true * (y_true - y_pred))
            m_mse = K.sum(masked_squared_error, axis=-1)  # we expect only one value different
            return m_mse

        # TODO right cost function (the sparse mse thingy)
        loss = masked_mse if 1 else mse
        self.model.compile(adam(lr=lr), loss=loss)
        # self.model_dual.compile(adam(lr=lr),
        #                         loss=[masked_mse, mse],
        #                         loss_weights=[1, 1e-1]   # while we want consistency/smoothing
        #                         )

        if memory is not None:
            self.memory = memory
        else:
            self.memory = Memory(n_classes=2)
            names = ['state', 'action', 'reward', 'state_next', 'done']
            self.memory.set_names(names)

    def act(self, state, b_stoch=False):

        y = self.model.predict(state)
        assert y.shape[0] == 1
        y = y[0]

        # TODO argmax or according to probability?
        # Too stochastic
        if b_stoch:
            # e = np.exp(y-np.max(y))
            # s = np.sum(e)
            # p = e/s
            #
            # return np.random.choice(np.arange(2), p=p)

            p_base = 0.1
            p = [p_base for _ in range(2)]
            p[np.argmax(y)] = 1-p_base

            action = np.random.choice(np.arange(2), p=p)

        else:
            action = np.argmax(y)

        print('q values:', y, self.names_action[action])

        return action

    def update_target(self):
        if self.b_double:
            self.model_target.set_weights(self.model.get_weights())
        else:
            raise AttributeError('double Rl nor selected')

    def experience_replay(self, i_try=0):

        # fit a batch at once

        if self.memory.b_enough_expirience(self.batch_size):
            return 0

        # TODO probs get batch per minibatch! in for loop.
        batch = self.memory.get_train_batch(self.batch_size)

        # batch_vali = random.sample(self.memory, self.batch_size) # TODO

        if 1:
            loss = self.train(batch, self.batch_size//self.mini_batch_size,
                              self.mini_batch_size,
                              initial_epoch=i_try
                              )
        else:
            loss = self.train_dual(batch, self.batch_size//self.mini_batch_size,
                                   self.mini_batch_size
                                   )

        return np.mean(loss)

    def remember(self, state, action, reward, state_next, done):

        save = [state, action, reward, state_next, done]

        return self.memory.add(save)

    def pretrain(self, model_type, n_pretrain=1000, epoch_start=0):
        # TODO, don't know if it makes sense/if it does something good...

        # TODO remove commented pieces of code once we know the new memory works
        # n_data = len(self.memory)
        # train_split = 0.8
        # n_train = int(n_data*train_split)

        # data_train = self.memory[:n_train]
        # data_val = self.memory[n_train:]

        lst_loss = []
        lst_val_loss = []

        logdir = 'logs/' + model_type + '_e{}'.format(epoch_start)    # TODO make it logs?

        from keras.callbacks import TensorBoard
        TensorBoard()

        tb = TrainValTensorBoard(Path(logdir), write_graph=False, batch_size=1)

        callbacks = [tb]

        # TODO would be nice to have a generator that generates x and y on the spot (would make training and callbacks way nicer)

        for i in range(epoch_start, epoch_start+n_pretrain):
            print('epoch:', i)

            batch = self.memory.get_train_batch(self.batch_size*self.mini_batch_size)
            batch_val = self.memory.get_valid_batch(self.batch_size*self.mini_batch_size)

            loss, val_loss = self.train(batch, epochs=1,
                                        mini_batch_size=self.mini_batch_size,
                                        batch_val=batch_val,
                                        callbacks=callbacks,
                                        initial_epoch=i
                                        )

            lst_loss.append(np.mean(loss))
            lst_val_loss.append(np.mean(val_loss))

        filepath = 'weights/' + model_type + '_pretrain_e{}.h5'.format(epoch_start+n_pretrain)
        self.model.save_weights(filepath)

        plt.figure()
        plt.plot(np.arange(n_pretrain), lst_loss, label='train')
        plt.plot(np.arange(n_pretrain), lst_val_loss, label='validation')
        plt.legend()
        plt.show()

        return np.mean(lst_loss)

    def _get_q(self, batch, b_next=False):
        lst_state, lst_action, lst_reward, lst_state_next, lst_terminal = [np.array(a) for a in zip(*batch)]

        lst_q_update = np.copy(lst_reward)

        # only when not terminal
        b_not_terminal = np.logical_not(lst_terminal)
        if self.b_double:
            lst_q_predict_next = self.model_target.predict(lst_state_next)
        else:
            lst_q_predict_next = self.model.predict(lst_state_next)

        lst_q_update[b_not_terminal] = (lst_reward + self.gamma * np.max(lst_q_predict_next, axis=-1))[b_not_terminal]

        lst_q_values = self.model.predict(lst_state)
        for i in range(len(lst_action)):
            lst_q_values[i, lst_action[i]] = lst_q_update[i]

        if b_next:
            return lst_state, lst_q_values, lst_q_predict_next
        else:
            return lst_state, lst_q_values

    def train(self, batch, epochs=0, mini_batch_size=1, batch_val=None, callbacks=None,
              initial_epoch=None
              ):
        """
        Trains the network
        :param epochs: can also be seen as BATCH_SIZE
        :param mini_batch_size:
        :return:
        """
        # TODO: train network on all the memory etc! with test set (probs split before training)
        # TODO: we have all the experience so it's probably okay to train the net and than just try to run the thing without needing to train at the moment.
        # TODO add validation

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

        for i in range(epochs):
            if self.b_double:
                if self.i_train % self.C_double == 0:
                    self.update_target()

            minibatch = batch[i*mini_batch_size:(i+1)*mini_batch_size]

            lst_state, lst_q_values = self._get_q(minibatch)

            if batch_val is not None:
                minibatch_val = random.sample(batch_val, mini_batch_size)
                lst_state_val, lst_q_values_val = self._get_q(minibatch_val)
                # lst_state_val =     # TODO
                validation_data = [lst_state_val, lst_q_values_val]
            else:
                validation_data = None

            if initial_epoch is None:
                epochs = 1
                initial_epoch = 0
            else:
                epochs = initial_epoch+1
            hist = self.model.fit(lst_state, lst_q_values, validation_data=validation_data,
                                  epochs=epochs,
                                  verbose=0, callbacks=callbacks,
                                  initial_epoch=initial_epoch)
            loss.append(hist.history['loss'])

            if batch_val is not None:
                loss_val.append(hist.history['val_loss'])

            self.i_train += 1

        if batch_val is None:
            return loss
        else:
            return loss, loss_val

    def train_dual(self,
                   batch,
                   epochs=0,
                   mini_batch_size=1,
                   batch_val=None
                   ):
        """
        Trains the network
        :param epochs: can also be seen as BATCH_SIZE
        :param mini_batch_size:
        :return:
        """

        """
        Doesn't make any sense to do it this way as the deritive will also be 0, so no learning!!
        I'll have to do it sequentially.
        """

        # TODO same as before, is it even needed to check this??
        # if len(self.memory) < epochs:
        #     return 0    # no learning while memory does not have enough experience

        loss = []
        # loss_val = []
        #
        # # # Outdated
        # # for state, action, reward, state_next, terminal in batch:
        # #     q_update = reward
        # #     if not terminal:
        # #         q_predict_next = self.model.predict(np.array([state_next]))[0]
        # #         q_update = (reward + GAMMA * np.max(q_predict_next, axis=-1))
        # #     q_values = self.model.predict(np.array([
        # #                                                state]))  # Is probably already calculated before, however with double model, this one is different (only updated sometimes)
        # #     q_values[0][action] = q_update
        # #
        # #     # Would prefer to not do SGD
        # #     hist = self.model.fit(np.array([state]), q_values, verbose=0)
        # #     loss.append(hist.history['loss'])
        #

        lst_state_next_prev = None
        lst_q_next_prev = None

        for i in range(epochs):
            minibatch = batch[i*mini_batch_size:(i+1)*mini_batch_size]

            _, _, _, lst_state_next, _ = [np.array(a) for a in zip(*minibatch)]

            lst_state, lst_q_values, lst_q_next = self._get_q(minibatch, b_next=True)

        #     if batch_val is not None:
        #         minibatch_val = random.sample(batch_val, mini_batch_size)
        #         lst_state_val, lst_q_values_val = self._get_q(minibatch_val)
        #         # lst_state_val =     # TODO
        #         validation_data = [lst_state_val, lst_q_values_val]
        #     else:
        #         validation_data = None

            # Does not work, no derivative

            if i == 0:
                hist = self.model.fit(lst_state, lst_q_values,
                                      # validation_data=validation_data,
                                      verbose=0
                                      )

                loss.append(hist.history['loss'])

            else:
                hist = self.model_dual.fit([lst_state, lst_state_next_prev], [lst_q_values, lst_q_next_prev],
                                           # validation_data=validation_data,
                                           verbose=0
                                           )

                loss.append(hist.history['q_loss'])

            # fit on previous. To make transition more smooth.
            lst_state_next_prev = lst_state_next
            lst_q_next_prev = lst_q_next


            # self.model.fit(lst_state_next, lst_q_next,
            #                # validation_data=validation_data,
            #                verbose=0
            #                )

        #     if batch_val is not None:
        #         loss_val.append(hist.history['val_loss'])

        # if batch_val is None:
        return loss
        # else:
        #     return loss, loss_val

    def save_model(self, name):
        self.model.save_weights(name + '.h5')

    def load_model(self, name):
        self.model.load_weights(name + '.h5')
        if self.b_double:
            self.update_target()

    def save_memory(self):

        with open('memory.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.memory, f)
        return 1

    def get_dist_state(self):
        # at some point: array([0.6911251 , 0.45466556, 0.02490752, 0.20573978])
        return np.std(np.array([a for a in zip(*self.memory.get_all_data())][0]), axis=0)


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = str(Path(log_dir) / 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = str(Path(log_dir) / 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
