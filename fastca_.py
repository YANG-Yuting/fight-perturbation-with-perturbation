# /usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import os
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
from attack_classification import *
from itertools import islice
from nltk.stem import WordNetLemmatizer
import time
import random
import json
from itertools import chain
from nltk.corpus import wordnet


"""给定一条文本，获得每个位置同义词个数等信息，写入文件./data/test_text.txt，作为fastCA的输入；
获得该文本各个位置同义词列表，用于结合CA映射回词语，进一步获得替换后文本"""
def pre_for_fastCA(text):
    # 1. 获取待测试文本各个位置的同义词个数
    syn_words_num = []  # 保存该文本各个位置同义词个数
    text_syns = []  # 保存该文本各个位置同义词（获取CA后，用其映射回词语）
    for word in text:
        print(word)
        synonyms = wordnet.synsets(word)
        lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        lemmas.sort()
        print(lemmas)
        if len(lemmas) == 0:
            num = 1
            text_syns.append([word])
        else:
            num = len(lemmas)
            text_syns.append(lemmas)
        print(num)
        syn_words_num.append(num)
    # 2. 写入文件
    file = './data/test_text_.txt'
    t = '2'  # 2-路覆盖
    with open(file, 'w') as w:
        w.write(t + '\n') # 第一行：写入t
        # print('The number of syn words in each position:')
        # print(len(syn_words_num),np.sum(np.array(syn_words_num)), np.array(syn_words_num))
        w.write(str(len(text)) + '\n') # 第二行：写入文本长度（词语个数）
        syn_words_num = [str(sw) for sw in syn_words_num]
        w.write(' '.join(syn_words_num) + '\n') # 第三行：写入所有位置同义词个数
        w.close()
    return text_syns


"""运行fastCA，将获得的覆盖数组转化为文本，并写入文件"""
def get_replace_text(idx, text):
    print(idx, text)
    # 1. 准备fastCA输入文件，并获得同义词列表
    text_syn = pre_for_fastCA(text)
    text_syn_flatten = [w for s in text_syn for w in s]  # 将该文本所有位置的同义词（包含自己）按顺序拉伸成一维列表
    # 2. 运行fastCA
    print('Run fastCA...') # 输入：data/test_text.txt 输出：data/CA_array.txt
    os.system('/home/yangyuting/projects/fastca/fastCA/FastCA /home/yangyuting/projects/TextFooler/data/test_text_.txt 100 1')
    # 3. 读取覆盖数组，转化为文本，并写入文件
    file = 'data/CA_array.txt'
    out_file = 'adv_results/imdb_fastca/' + str(idx) +'.txt'
    w = open(out_file, 'w')
    lines = open(file, 'r').readlines()
    all_replace_text = []
    for i in range(len(lines)):
        replace_text = []
        line = [int(n) for n in lines[i].strip().split(' ')]
        for j in line:
            replace_word = text_syn_flatten[j]
            replace_text.append(replace_word)
        all_replace_text.append(replace_text)
        w.write(' '.join(replace_text)+'\n') # 写入文件
        w.flush()
    w.close()
    print('%d data wrote finished' % idx)
    return all_replace_text


"""获得所有替换文本"""
def get_replace_text_all():
    dataset_path = 'data/imdb'
    data_size = 1000
    # get data
    texts, labels = dataloader.read_corpus(dataset_path)
    data = list(zip(texts, labels))
    data = data[:data_size] # choose how many samples for adversary
    # 获得替换文本，并写入文件
    for idx, (text, true_label) in enumerate(data):
        _ = get_replace_text(idx, text)


"""对于一条文本，获取它周围的一组随机数据（随机选择一个位置，随机选择一个同义词进行替换），暂定10个"""
def get_text_around(text, syn_dict, sample_num =10):
    # 1. 标志哪些位置有同义词
    has_syn = []
    keys = syn_dict.keys()
    for i in range(len(text)):
        if text[i] in keys:
            has_syn.append(i)
    # 2. 从中随机选取sample_num个位置
    all_new_text = []
    if len(has_syn) >= sample_num:  # 若超过sample_num个位置有同义词
        psts = random.sample(has_syn, sample_num)
        for pst in psts:  # 每个位置，随机选取一个同义词
            new_text = text.copy()
            new_text[pst] = random.choice(syn_dict[text[pst]])
        all_new_text.append(new_text)
    else:
        
        
        for i in has_syn:
            new_text = text.copy()
            new_text[i] = random.choice(syn_dict[text[i]])
            all_new_text.append(new_text)
        left_num = 0
        # while left_num < sample_num-len(has_syn):
        #     new_text = text.copy()
        #     pst = random.sample(ha)
        #     syns = syn_dict[]
            
        pass
    return all_new_text


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        default='data/imdb',
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        default='wordCNN',
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        default='./pretrained_model/imdb',
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='./glove.6B/glove.6B.200d.txt',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        default='./data/counter-fitted-vectors.txt',
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='./cos_sim_counter_fitting.npy',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    # Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    parser.add_argument("--data_size", default=10, type=int, help="Data size to create adversaries")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data
    texts, labels = dataloader.read_corpus(args.dataset_path)
    data = list(zip(texts, labels))
    data = data[:args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # build dictionary via the embedding file
    # idx2word = {}
    # word2idx = {}
    #
    # print("Building vocab...")
    # with open(args.counter_fitting_embeddings_path, 'r') as ifile:
    #     for line in ifile:
    #         word = line.split()[0]
    #         if word not in idx2word:
    #             idx2word[len(idx2word)] = word
    #             word2idx[word] = len(idx2word) - 1

    # Replace words and attack
    
    """Attack"""
    adv_w = open('adv_results/imdb_adv_results.txt', 'w')
    # wordnet_lemmatizer = WordNetLemmatizer()
    ori_predict_true = 0
    # label_change_lemm = 0
    change_rates = []  # 记录对于各个测试数据，覆盖数组中改变label的数据比例
    for idx, (text, true_label) in enumerate(data):
        print(idx)
        print(idx, file = adv_w)
        # starttime = time.time()
        # first check the prediction of the original text
        orig_probs = predictor([text]).squeeze()
        orig_label = torch.argmax(orig_probs)
        if orig_label == true_label:  # 对每一个原始预测正确的样本，计算替换后的准确率变化比例
            ori_predict_true += 1
            label_change_replace = 0  # 保存对于该测试样本，覆盖数组中label变化的个数
            replace_text_num = [] # 保存覆盖数组元素个数
            with open('adv_results/imdb_fastca/' + str(idx) + '.txt', 'r') as r:
                changed_w = open('adv_results/imdb_adv_results/' + str(idx) + '.txt' , 'w')
                all_replace_text = r.readlines()
                replace_text_num.append(len(all_replace_text))
                for replace_text in all_replace_text:
                    replace_text = replace_text.split(' ')
                    # print('Original text:')
                    # print(len(text), text)
                    # print('Replace to:')
                    # print(len(replace_text), replace_text)
                    replace_probs = predictor([replace_text]).squeeze()
                    replace_label = torch.argmax(replace_probs)
                    if orig_label != replace_label:
                        label_change_replace += 1
                        # print('Label changed at this time.')
                        changed_w.write('1 ') # 标志当前替换文本会改变label
                    else:
                        changed_w.write('0 ')
                changed_w.write('\n')
            print('Num of replace text: %d' % len(all_replace_text))
            print('Num of replace text: %d' % len(all_replace_text), file = adv_w)
            change_rate = float(label_change_replace) / float(len(all_replace_text))  # 覆盖数组中，会改变label的所占比例
            print('Rates of label changed for all replace text: %f' % change_rate)
            print('Rates of label changed for all replace text: %f' % change_rate, file = adv_w)
            change_rates.append(change_rate)

            # 词替换
            # replace_text = text.copy()
            # text_lemm = text.copy()
            # for i in range(len(replace_text)):
            #     replace_text[i] = wordnet_lemmatizer.lemmatize(replace_text[i])
            #     text_lemm[i] = wordnet_lemmatizer.lemmatize(text_lemm[i])
            #     try:
            #         syns = syn_dict[replace_text[i]]
            #     except:
            #         continue
            #     replace_text[i] = syns[0]  # 选第一个候选词
            # replace_probs = predictor([replace_text]).squeeze()
            # replace_label = torch.argmax(replace_probs)
            # lemm_probs = predictor([text_lemm]).squeeze()
            # lemm_label = torch.argmax(lemm_probs)

            # if orig_label != lemm_label:
            #     label_change_lemm += 1
            # if orig_label != replace_label:
            #     label_change_replace += 1
        else:
            print('Predict false!')
            change_rates.append(-1)
            replace_text_num.append(-1)

    print('Origin data size: %d, Origin predict true: %d' % (args.data_size, ori_predict_true))
    print('Origin data size: %d, Origin predict true: %d' % (args.data_size, ori_predict_true), file = adv_w)
    print('Change rates of t-cover-array:')
    print(change_rates)
    adv_w.close()

    
if __name__ == '__main__':
    text = ['i', 'really', 'like', 'traci', 'lords', 'she', 'may', 'not', 'be', 'the', 'greatest', 'actress', 'in', 'the', 'world',
                 'but', 'she', "'s", 'rather', 'good', 'in', 'this', 'she', 'play', 'the', 'dowdy', ',', 'conservative', ',', 'reporter',
                 'to', 'a', "'", 't', "'", 'it', "'s", 'a', 'great', 'little', 'thriller', 'which', 'keeps', 'you', 'guessing', 'for', 'a',
                 'good', 'while', 'jeff', 'fahey', 'is', 'also', 'good', 'as', 'traci', "'s", 'boss', 'i', 'think', 'given', 'a', 'decent',
                 'break', 'traci', 'could', 'be', 'a', 'top', 'actress', 'she', "'s", 'certainly', 'no', 'worse', 'than', 'some', 'of',
                 'today', "'s", 'leading', 'ladies'] #True, Prediction, Replace, Lemm Label: 1, 1, 0, 1
    
    # get_replace_text_all()
    # main()
    pre_for_fastCA(text)

