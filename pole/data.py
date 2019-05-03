import numpy as np
import random
from itertools import chain


class Memory(object):
    def __init__(self, n_classes=1):
        pass    # TODO

        distr = [.5, .5]
        assert sum(distr) == 1
        self.n_total = 1000
        self.n_list = [int(p*self.n_total) for p in distr]

        self.training = [[] for _ in range(n_classes)]
        self.validation = [[] for _ in range(n_classes)]

    def set_names(self, names):
        self.names = names

    def add(self, save):

        i_class = 1 if save[self.names.index('done')] else 0  # Should be more general. Perhaps with a function that finds i_class out of the save

        # 0.8 chance if adding to training
        # TODO set as parameter at init
        # TODO n_total should probably also scale with this probability
        p_traing_val = [0.8, 0.2]
        i_train_val = np.random.choice(np.arange(2), p=p_traing_val)

        m = self.training if i_train_val == 0 else self.validation

        n = self.n_list[i_class]
        if len(m[i_class]) == n:
            m[i_class][np.random.randint(n)] = save
        elif len(m[i_class]) > n:
            raise ValueError(len(m[i_class]), 'should never be larger than', n)
        else:
            m[i_class].append(save)

        return 1

        # TODO remove once we know the above code is implemented correctly
        # if len(self.memory) == N_MAX:
        #     self.memory[np.random.randint(N_MAX)] = save
        # elif len(self.memory) > N_MAX:
        #     raise ValueError(len(self.memory), 'should not be larger than', N_MAX)
        # else:
        #     self.memory.append(save)
        #
        # return 1

    def _len_memory_train(self):
        return sum(len(x) for x in self.training)

    def b_enough_expirience(self, n):
        return n >= self._len_memory_train(
        )

    def get_train_batch(self, n):
        batch = random.sample(self._total(self.training), n)
        return batch

    def get_valid_batch(self, n):
        batch = random.sample(self._total(self.validation), n)
        return batch

    def get_all_data(self):
        return self._total(self._total(s) for s in [self.training, self.validation])

    def _total(self, l):
        # return [item for sublist in set for item in sublist]
        return list(chain.from_iterable(l))
