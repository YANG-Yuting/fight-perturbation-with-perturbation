# /usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import pickle
from tqdm import tqdm
import dataloader
from sklearn.model_selection import train_test_split
import os
from nltk.tag import StanfordPOSTagger

if __name__ == '__main__':
    jar = 'dataset/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
    model = 'dataset/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

    task = 'imdb'
    train = True

    # load data
    data_path = '/pub/data/huangpei/TextFooler/data/adversary_training_corpora/'
    if task == 'mr':
        # train_x, train_y = dataloader.read_corpus('data/adversary_training_corpora/mr/train.txt', clean=False,FAKE=False,shuffle=False)
        # test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test.txt', clean=False, FAKE=False,shuffle=False)

        with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/mr/dataset_20000.pkl', 'rb') as f:
            datasets = pickle.load(f)
        train_x = datasets.train_seqs2
        train_x = [[datasets.inv_full_dict[word] for word in text] for text in train_x]
        train_y = datasets.train_y
        test_x = datasets.test_seqs2
        test_x = [[datasets.inv_full_dict[word] for word in text] for text in test_x]
        test_y = datasets.test_y
    elif task == 'imdb':
        # train_x, train_y = dataloader.read_corpus(os.path.join(data_path + 'imdb', 'train_tok.csv'), clean=False, FAKE=False, shuffle=True)
        # test_x, test_y = dataloader.read_corpusus(os.path.join(data_path + 'imdb', 'test_tok.csv'), clean=False, FAKE=False, shuffle=True)
        with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/imdb/dataset_50000.pkl', 'rb') as f:
            datasets = pickle.load(f)
        train_x = datasets.train_seqs2
        train_x = [[datasets.inv_full_dict[word] for word in text] for text in train_x]
        train_y = datasets.train_y
        test_x = datasets.test_seqs2
        test_x = [[datasets.inv_full_dict[word] for word in text] for text in test_x]
        test_y = datasets.test_y
    elif task == 'fake':
        train_x, train_y = dataloader.read_corpus(data_path + '{}/train_tok.csv'.format(task), clean=False, FAKE=True, shuffle=True)
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=1)

    if train:
        out_file = data_path + task + '/pos_tags.pkl'
        data = train_x
    else:
        out_file = data_path + task + '/pos_tags_test.pkl'
        data = test_x

    # 删除数据中的奇怪符号
    clean_data = []
    for text in data:
        new_text = []
        for word in text:
            if word == '\x85':
                continue
            else:
                new_text.append(word.replace('\x85',''))
        clean_data.append(new_text)

    # POS
    all_pos_tags = {}
    a = pos_tagger.tag_sents(clean_data)
    for i in range(len(clean_data)):
        text = ' '.join(clean_data[i])
        all_pos_tags[text] = a[i]
        if text != ' '.join([b[0] for b in a[i]]):
            print(i)
            exit(0)
    # for k in a:
    #     text = ' '.join([b[0] for b in k])
    #     all_pos_tags[text] = k

    # write to file
    f = open(out_file, 'wb')
    pickle.dump(all_pos_tags, f)