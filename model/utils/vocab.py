import numpy as np


class Dictionary(object):
    def __init__(
            self,
            replace_lower_freq_word=False,
            replace_word='<unk>'
    ):
        self.word2id = {}
        self.id2word = []
        self.word2freq = {}
        self.id2freq = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.replace_word = replace_word

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word.append(word)
            self.word2id[word] = len(self.id2word) - 1
            self.word2freq[word] = 1
        else:
            self.word2freq[word] += 1

    def rebuild(self, min_count=5):
        self.id2word = sorted(self.word2freq, key=self.word2freq.get, reverse=True)

        for new_word_id, word in enumerate(self.id2word):
            freq = self.word2freq[word]
            if freq >= min_count:
                self.word2id[word] = new_word_id
            else:
                if self.replace_lower_freq_word:
                    self.word2id[self.replace_word] = new_word_id
                    sum_unk_freq = 0
                    for word in self.id2word[new_word_id:]:
                        sum_unk_freq += self.word2freq[word]
                        del self.word2id[word]
                    self.word2freq[self.replace_word] = sum_unk_freq
                    self.id2word = self.id2word[:new_word_id]
                    self.id2word.append(self.replace_word)

                else:
                    for word in self.id2word[new_word_id:]:
                        del self.word2id[word]
                    self.id2word = self.id2word[:new_word_id]

                break
        self.id2freq = np.array([self.word2freq[word] for word in self.id2word])
        del self.word2freq

    def __len__(self):
        return len(self.id2word)

class Corpus(object):
    def __init__(
            self,
            min_count=5,
            replace_lower_freq_word=False,
            replace_word='<unk>',
            bos_word='<bos>',
            eos_word='<eos>'
    ):
        self.dictionary = Dictionary(replace_lower_freq_word, replace_word)
        self.min_count = min_count
        self.num_words = 0
        self.num_vocab = 0
        self.num_docs = 0
        self.discard_table = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.replace_word = replace_word
        self.bos_word = bos_word
        self.eos_word = eos_word

    def tokenize_from_file(self, path):
        def _add_special_word(sentence):
            return self.bos_word + ' ' + sentence + ' ' + self.eos_word

        self.num_words = 0
        self.num_docs = 0
        with open(path, encoding='utf-8') as f:
            for l in f:
                for word in _add_special_word(l.strip()).split():
                    self.dictionary.add_word(word=word)
        self.dictionary.rebuild(min_count=self.min_count)
        self.num_vocab = len(self.dictionary)

        with open(path,encoding='utf-8') as f:
            docs = []
            for l in f:
                doc = []
                for word in _add_special_word(l.strip()).split():
                    if word in self.dictionary.word2id:
                        doc.append(self.dictionary.word2id.get(word))
                    elif self.replace_lower_freq_word:
                        doc.append(self.dictionary.word2id.get(self.replace_word))
                if len(doc) > 1:
                    docs.append(np.array(doc))
                    self.num_words += len(doc)
                    self.num_docs += 1

        return np.array(docs)

    def build_discard_table(self, t=1e-4):
        tf = t / (self.dictionary.id2freq / self.num_words)
        self.discard_table = np.sqrt(tf) + tf

    def discard(self, word_id, rnd):
        return rnd.rand() > self.discard_table[word_id]
