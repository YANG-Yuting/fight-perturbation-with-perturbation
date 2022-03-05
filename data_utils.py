import os
# import nltk
import re
from collections import Counter
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

import pickle as pickle
import random

class myDataset(object):
    def __init__(self, task, path='data/adversary_training_corpora/fake', max_vocab_size=None):
        self.task = task
        self.path = path
        # self.train_path = path + '/train'
        # self.test_path = path + '/test'
        self.vocab_path = path + '/' + task + '.vocab'
        self.max_vocab_size = max_vocab_size
        self._read_vocab()
        train_text, self.train_y, test_text, self.test_y = self.read_text(self.path)
        self.train_text = train_text
        self.test_text = test_text
        print('tokenizing...')

        # Tokenized text of training data
        self.tokenizer = Tokenizer()

        # nlp = spacy.load('en')
        # train_text = [nltk.word_tokenize(doc) for doc in train_text]
        # test_text = [nltk.word_tokenize(doc) for doc in test_text]
        # train_text = [[w.string.strip() for w in nlp(doc)] for doc in train_text]
        # test_text = [[w.string.strip() for w in nlp(doc)] for doc in test_text]
        self.tokenizer.fit_on_texts(self.train_text)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1
        # sorted_words = sorted([x for x in self.tokenizer.word_counts])
        # self.top_words = sorted_words[:max_vocab_size-1]
        # self.other_words = sorted_words[max_vocab_size-1:]
        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]

        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word
        print('Dataset built !')

    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y), f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y), f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)

    def _read_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab_words = f.read().split('\n')
            self.vocab = dict([(w, i) for i, w in enumerate(vocab_words)])
            self.reverse_vocab = dict([(i, w) for w, i in self.vocab.items()])

    def read_text(self, path):
        if self.task == 'imdb':
            """ Returns a list of text documents and a list of their labels
            (pos = +1, neg = 0) """
            # train
            train_path = path + '/train'
            train_pos_path = train_path + '/pos'
            train_neg_path = train_path + '/neg'
            train_pos_files = [train_pos_path + '/' + x for x in os.listdir(train_pos_path) if x.endswith('.txt')]
            train_neg_files = [train_neg_path + '/' + x for x in os.listdir(train_neg_path) if x.endswith('.txt')]

            train_pos_list = [open(x, 'r').read().lower() for x in train_pos_files]
            train_neg_list = [open(x, 'r').read().lower() for x in train_neg_files]
            train_data_list = train_pos_list + train_neg_list
            train_label_list = [1] * len(train_pos_list) + [0] * len(train_neg_list)

            # test
            test_path = path + '/test'
            test_pos_path = test_path + '/pos'
            test_neg_path = test_path + '/neg'
            test_pos_files = [test_pos_path + '/' + x for x in os.listdir(test_pos_path) if x.endswith('.txt')]
            test_neg_files = [test_neg_path + '/' + x for x in os.listdir(test_neg_path) if x.endswith('.txt')]

            test_pos_list = [open(x, 'r').read().lower() for x in test_pos_files]
            test_neg_list = [open(x, 'r').read().lower() for x in test_neg_files]
            test_data_list = test_pos_list + test_neg_list
            test_label_list = [1] * len(test_pos_list) + [0] * len(test_neg_list)
        elif self.task == 'mr':
            train_data_list = []
            train_label_list = []
            train_lines = open(path + '/train.txt', 'r').read().lower().splitlines()
            for line in train_lines:
                train_data_list.append(line.split(' ')[1:])
                train_label_list.append(int(line.split(' ')[0]))

            test_data_list = []
            test_label_list = []
            test_lines = open(path + '/test.txt', 'r').read().lower().splitlines()
            for line in test_lines:
                test_data_list.append(line.split(' ')[1:])
                test_label_list.append(int(line.split(' ')[0]))
        elif self.task == 'fake':
            train_data_list = []
            train_label_list = []
            train_lines = open(path + '/train_tok.csv', 'r').read().lower().splitlines()
            for line in train_lines:
                train_data_list.append(line[:-2])
                train_label_list.append(int(line[-1]))

            test_data_list = []
            test_label_list = []
            test_lines = open(path + '/test_tok.csv', 'r').read().lower().splitlines()
            for line in test_lines:
                test_data_list.append(line[:-2])
                test_label_list.append(int(line[-1]))

            # test_data_list = []
            # test_label_list = []
            # test_lines = open('data/fake', 'r').read().lower().splitlines()
            # for line in test_lines:
            #     test_data_list.append(line.split(' ')[1:])
            #     test_label_list.append(int(line.split(' ')[0]))

        a = list(zip(train_data_list, train_label_list))
        random.shuffle(a)
        train_data_list, train_label_list = zip(*a)

        b = list(zip(test_data_list, test_label_list))
        random.shuffle(b)
        test_data_list, test_label_list = zip(*b)

        return train_data_list, train_label_list, test_data_list, test_label_list


    def build_text(self, text_seq):
        text_words = [self.inv_full_dict[x] for x in text_seq]
        return ' '.join(text_words)


class IMDBDataset(object):
    def __init__(self, path='dataset', max_vocab_size=None):
        self.path = path
        self.train_path = path + '/train'
        self.test_path = path + '/test'
        self.vocab_path = path + '/imdb.vocab'
        self.max_vocab_size = max_vocab_size
        self._read_vocab()
        train_text, self.train_y = self.read_text(self.train_path)
        test_text, self.test_y = self.read_text(self.test_path)
        self.train_text = train_text
        self.test_text = test_text
        print('tokenizing...')

        # Tokenized text of training data
        self.tokenizer = Tokenizer()

        # nlp = spacy.load('en')
        # train_text = [nltk.word_tokenize(doc) for doc in train_text]
        # test_text = [nltk.word_tokenize(doc) for doc in test_text]
        # train_text = [[w.string.strip() for w in nlp(doc)] for doc in train_text]
        # test_text = [[w.string.strip() for w in nlp(doc)] for doc in test_text]
        self.tokenizer.fit_on_texts(self.train_text)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1
        # sorted_words = sorted([x for x in self.tokenizer.word_counts])
        # self.top_words = sorted_words[:max_vocab_size-1]
        # self.other_words = sorted_words[max_vocab_size-1:]
        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]

        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word
        print('Dataset built !')

    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y), f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y), f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)

    def _read_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab_words = f.read().split('\n')
            self.vocab = dict([(w, i) for i, w in enumerate(vocab_words)])
            self.reverse_vocab = dict([(i, w) for w, i in self.vocab.items()])

    def read_text(self, path):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        pos_list = []
        neg_list = []

        pos_path = path + '/pos'
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_list = [open(x, 'r').read().lower() for x in pos_files]
        neg_list = [open(x, 'r').read().lower() for x in neg_files]
        data_list = pos_list + neg_list
        labels_list = [1] * len(pos_list) + [0] * len(neg_list)
        return data_list, labels_list

    def build_text(self, text_seq):
        text_words = [self.inv_full_dict[x] for x in text_seq]
        return ' '.join(text_words)