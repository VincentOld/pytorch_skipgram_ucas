import numpy as np


class NegativeSampler(object):
    def __init__(
            self, frequency: np.ndarray, negative_alpha=0., table_length=int(1e8), is_neg_loss=True,
    ):
        if negative_alpha == 0:
            self.table_length = len(frequency)
            self.negative_table = np.arange(self.table_length, dtype=np.int32)

        else:
            self.table_length = table_length
            z = np.sum(np.power(frequency, negative_alpha))
            negative_table = np.zeros(table_length, dtype=np.int32)
            begin_index = 0
            for index, freq in enumerate(frequency):
                c = np.power(freq, negative_alpha)
                end_index = begin_index + int(c * table_length / z) + 1
                negative_table[begin_index:end_index] = index
                begin_index = end_index

            self.negative_table = negative_table
        if not is_neg_loss:
            self.noise_dist = np.power(frequency, negative_alpha)

    def sample(self, k, rnd: np.random.RandomState, exclude_words=None):
        """
        :param k: number of negative samplings
        :param rnd: np.random.RandomState
        :param exclude_words: numpy array contains context words
        :return: negative words (#context_words, #negatives)
        """
        size = (len(exclude_words), k)
        if exclude_words is None:
            return self.negative_table[rnd.randint(low=0, high=self.table_length, size=size)]
        else:
            negs = np.zeros(size, np.int32)
            for i, word in enumerate(exclude_words):
                negs_for_word = self.negative_table[rnd.randint(low=0, high=self.table_length, size=k)]
                while word in negs_for_word:
                    negs_for_word = self.negative_table[rnd.randint(low=0, high=self.table_length, size=k)]
                negs[i] = negs_for_word
            return negs
