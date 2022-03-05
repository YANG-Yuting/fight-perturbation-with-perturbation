import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from sru import *
import dataloader_ as dataloader
import modules
import math
from gen_pos_tag import pos_tagger
from numpy.random import choice

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import datetime

# 样本带权重的自定义损失函数
class myloss(nn.Module):
    def _init_(self):
        super().__init__()
    def forward(self, predict,y,weight):
        loss = Variable(torch.FloatTensor([0.0]), requires_grad=True).cuda()
        for i in range(predict.shape[0]):
            target = predict[i][y[i].data]
            loss = loss + (-target + torch.log(torch.sum(torch.exp(predict[i]))))*weight[i]
        loss = loss/float(y.shape[0])
        return loss

# 获得候选集（对于给定词语，获得它5个同义词列表的同义词的交集，均包含自己）
def get_candidates(word, pos, word_candidate, tf_vocabulary, text_words, position, if_bound): # word是词语id,text_words是字符串
    syns_l1 = word_candidate[word][pos] + [word] # 包含自己；一级同义词
    # 1.直接获得原位置的一级同义词，返回
    syns_inter = syns_l1
    # 1. 对所有同义词的同义词获取交集 acc:mr-lstm, 69.6%
    # for syn in syns_l1:
    #     syns_l2 =  word_candidate[syn][pos] + [syn, word]# 默认同义词和原词一定有一致的词性；将原词也加上（大多具有对称性，有的因不在top5而被过滤了）
    #     syns_inter = list(set(syns_inter).intersection(set(syns_l2)))
    # 2. 从同义词里随机选一个/第一个，获得其同义词，再得交集 69.6%
    # syns_l2 = word_candidate[syns_l1[0]][pos] + [syns_l1[0], word]  # 默认同义词和原词一定有一致的词性；将原词也加上（大多具有对称性，有的因不在top5而被过滤了）
    # syns_inter = list(set(syns_inter).intersection(set(syns_l2)))
    # 3. 从同义词中获得词频最高的那个，获得其同义词，再得交集
    # # syn_max, ti_max = word, 0 # 默认词频最大的同义词是自己
    # tf_1gram_all, tf_2gram_former_all,tf_2gram_latter_all, tf_2gram_all  = {},{},{},{}
    # for syn in syns_l1:
    #     tf_1gram, tf_2gram_former, tf_2gram_latter = 0,0,0
    #     # 1-gram
    #     if inv_full_dict[syn] in tf_vocabulary.keys():
    #         tf_1gram = tf_vocabulary[inv_full_dict[syn]]
    #     tf_1gram_all[syn] = tf_1gram
    #
    #     # 2-gram
    #     if if_bound == 'not_bound':
    #         # 左2-gram
    #         two_gram_former = text_words[position-1] + ' ' + inv_full_dict[syn]  # 获得文本中该位置前面的词语和该位置同义词的拼接（2-gram）
    #         if two_gram_former in tf_vocabulary.keys():
    #             tf_2gram_former = tf_vocabulary[two_gram_former]
    #         tf_2gram_former_all[two_gram_former] = tf_2gram_former
    #         tf_2gram_all[two_gram_former] = tf_2gram_former
    #         # 右2-gram
    #         two_gram_latter = inv_full_dict[syn] + ' ' + text_words[position + 1]
    #         if two_gram_latter in tf_vocabulary.keys():
    #             tf_2gram_latter = tf_vocabulary[two_gram_latter]
    #         tf_2gram_latter_all[two_gram_latter] = tf_2gram_latter
    #         tf_2gram_all[two_gram_latter] = tf_2gram_latter
    #     elif if_bound == 'left_bound':
    #         two_gram_latter = inv_full_dict[syn] + ' ' + text_words[position + 1]
    #         if two_gram_latter in tf_vocabulary.keys():
    #             tf_2gram_latter = tf_vocabulary[two_gram_latter]
    #         tf_2gram_latter_all[two_gram_latter] = tf_2gram_latter
    #         tf_2gram_all[two_gram_latter] = tf_2gram_latter
    #     elif if_bound == 'right_bound':
    #         two_gram_former = text_words[position - 1] + ' ' + inv_full_dict[syn]  # 获得文本中该位置前面的词语和该位置同义词的拼接（2-gram）
    #         if two_gram_former in tf_vocabulary.keys():
    #             tf_2gram_former = tf_vocabulary[two_gram_former]
    #         tf_2gram_former_all[two_gram_former] = tf_2gram_former
    #         tf_2gram_all[two_gram_former] = tf_2gram_former
    #
    # if tf_2gram_all.values() == 0: # 若所有2-gram都未在训练集中出现过，则在1-gram中寻找最佳
    #     syn_best_1gram = max(tf_1gram_all, key=tf_1gram_all.get)  # id
    #     syn_best = syn_best_1gram
    # else:
    #     if if_bound == 'left_bound':
    #         syn_best_2gram = max(tf_2gram_latter_all, key=tf_2gram_latter_all.get).split(' ')[0]
    #     elif if_bound == 'right_bound':
    #         syn_best_2gram = max(tf_2gram_former_all, key=tf_2gram_former_all.get).split(' ')[1]
    #     elif if_bound == 'not_bound':
    #         if max(tf_2gram_latter_all) > max(tf_2gram_former_all):
    #             syn_best_2gram = max(tf_2gram_latter_all, key=tf_2gram_latter_all.get).split(' ')[0]
    #         else:
    #             syn_best_2gram = max(tf_2gram_former_all, key=tf_2gram_former_all.get).split(' ')[1]
    #     syn_best_2gram = full_dict[syn_best_2gram]  # str转id
    #     syn_best = syn_best_2gram
    #
    # # 就用1-gram
    # # syn_best_1gram = max(tf_1gram_all, key=tf_1gram_all.get)  # id
    # # syn_best = syn_best_1gram
    #
    # syns_l2 = word_candidate[syn_best][pos] + [syn_best] # 默认同义词和原词一定有一致的词性；加上原词word，adv acc:68.8，不加，73.1
    # syns_inter = syns_l2 # 不求交集了，直接返回最高词频同义词的同义词
    #
    # # syns_inter = list(set(syns_inter).intersection(set(syns_l2)))


    return syns_inter


class Model(nn.Module):
    def __init__(self, args, max_seq_length, embedding, hidden_size=150, depth=1, dropout=0.3, cnn=False, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(embs = dataloader.load_embedding(embedding))
        self.word2id = self.emb_layer.word2id
        self.max_seq_length = max_seq_length
        self.tf_vocabulary=pickle.load(open('data/adversary_training_corpora/mr/tf_vocabulary.pkl', "rb"))
        if cnn:
            self.encoder = modules.CNN_Text(self.emb_layer.n_d, widths = [3,4,5],filters=hidden_size)
            d_out = 3*hidden_size
        else:
            self.encoder = nn.LSTM(self.emb_layer.n_d,hidden_size//2,depth,dropout = dropout,bidirectional=True)
            d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)

    #input: tensor, id 统一长度（通常为一个batch，batch_size*max_len）
    def forward(self, input):
        if self.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            # output = output[-1]
            output = torch.max(output, dim=0)[0].squeeze()

        output = self.drop(output)
        return self.out(output)

    # def forward(self, text):  # text: 划分batch后的 ; return hidden state
    #     # batches_x = dataloader.create_batches_x(text, self.max_seq_length, batch_size, self.word2id)
    #     outs = []
    #     for x in text:
    #         x = Variable(x)
    #         if self.cnn:
    #             x = x.t()
    #         emb = self.emb_layer(x)
    #
    #         if self.cnn:
    #             output = self.encoder(emb)
    #         else:
    #             output, hidden = self.encoder(emb)
    #             # output = output[-1]
    #             output = torch.max(output, dim=0)[0].squeeze()
    #         output = self.drop(output)
    #         output = self.out(output)
    #         outs.append(output)
    #     return torch.cat(outs, dim=0)

    # 输入：生数据，字符串

    # text: list of list, str
    def text_pred_org(self, text, batch_size=128):
        # batches_x = dataloader.create_batches_x(text, self.max_seq_length, batch_size, self.word2id)
        batches_x = dataloader.create_batches_x(text, 128, self.word2id)
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                output = self.forward(x)
                outs.append(F.softmax(output, dim=-1))

        return torch.cat(outs, dim=0)

    # 生成多个样本附近的随机采样样本
    # text: list of list, str

    def gen_sample_aText(self, text, sample_num, batch_size):  # text: list of str id
        # 获得同义词空间随机采样样本
        text = [x for x in text if x != '\x85']  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
        # if args.task == 'mr':
        #     text = [x for x in text if x != 50000]  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
        # elif args.task == 'imdb':
        #     text = [x for x in text if x != 5169]  # 注：文本里有5169(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        # 加载词频信息
        tf_vocabulary = self.tf_vocabulary

        # full_dict['<oov>'] = len(full_dict.keys())
        # inv_full_dict[len(full_dict.keys())] = '<oov>'
        # text_words = [inv_full_dict[id] for id in text]  # id转word
        pos_tag = pos_tagger.tag(text)  # 改为直接做词性标注，分词需要词语
        sample_texts = []

        # 按照对f分类的重要性，对句子中的词语排序（参考TF攻击attack_classification_hownet_top5.py中的做法，get importance）
        # get importance score
        orig_probs = self.text_pred_org([text], batch_size)
        orig_label = torch.argmax(orig_probs)
        orig_prob = orig_probs.max()
        leave_1_texts = [text[:ii] + ['<oov>'] + text[min(ii + 1, len(text)):] for ii in range(len(text))]
        leave_1_probs = self.text_pred_org(leave_1_texts, batch_size)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        a = (leave_1_probs_argmax != orig_label).float()
        b = leave_1_probs.max(dim=-1)[0]
        c = torch.index_select(orig_probs[0], 0, leave_1_probs_argmax)
        d = b - c
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()
        # 保存按重要性排序的下标（大->小）
        text_idx_sorted = sorted(range(len(import_scores)), key=lambda k: import_scores[k], reverse=True)

        # Sample
        # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
        text_ids = []
        for word_str in text:
            if word_str in full_dict.keys():
                text_ids.append(full_dict[word_str]) # id
            else:
                text_ids.append(word_str) # str

        for s in range(sample_num):
            sample_text_strs = text_ids.copy() # 保存的是采样后的，str
            replace_num = 0.0
            for j in text_idx_sorted:  # 对于每个词语（按重要性排序）
                word = sample_text_strs[j]
                if isinstance(word, str):  # 若当前词语不在词汇表中，一定没有同义词，直接返回
                    continue

                # 若是id：
                pos = pos_tag[j][1]  # 当前词语词性
                if pos not in pos_list:
                    continue
                if pos.startswith('JJ'):
                    pos = 'adj'
                elif pos.startswith('NN'):
                    pos = 'noun'
                elif pos.startswith('RB'):
                    pos = 'adv'
                elif pos.startswith('VB'):
                    pos = 'verb'
                # 获得候选集
                # 1.
                # neigbhours = word_candidate[word][pos]  # 候选集
                # neigbhours.append(word)  # 候选集包含自己
                # 2.
                # 标记当前词语是否在句子的边界上
                if_bound = 'not_bound'
                if j == 0:
                    if_bound = 'left_bound'
                if j == len(sample_text_strs) -1:
                    if_bound = 'right_bound'
                neigbhours = get_candidates(word, pos, word_candidate, tf_vocabulary, text, j, if_bound)
                sample_text_strs[j] = inv_full_dict[choice(neigbhours)]  # 候选集中随机选择一个，并转成str

                replace_num += 1 # 记录替换位置  （可能会替换到原词，此时算不算替换了呢？）
                change_ratio = replace_num / float(len(sample_text_strs))
                if  change_ratio > 0.2: # 若替换比例大于15%，则停止替换 0.1 0.15 0.2 0.25 0.3 不控制，测试集上准确率分别为 _ _ 77.13 75.82 75.84 77.04
                    break

            # id转为str
            for i in range(len(sample_text_strs)):
                if isinstance(sample_text_strs[i], int):
                    sample_text_strs[i] = inv_full_dict[sample_text_strs[i]]

            sample_texts.append(sample_text_strs)
        return sample_texts # list of list of str

    #add by HP
    def gen_sample_aText1(self, text, sample_num, batch_size):  # text: list of str id
        # 获得同义词空间随机采样样本
        text = [x for x in text if x != '\x85']  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
        # if args.task == 'mr':
        #     text = [x for x in text if x != 50000]  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
        # elif args.task == 'imdb':
        #     text = [x for x in text if x != 5169]  # 注：文本里有5169(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        # 加载词频信息
        #tf_vocabulary = pickle.load(open('data/adversary_training_corpora/mr/tf_vocabulary.pkl', "rb"))
        tf_vocabulary = self.tf_vocabulary
        # full_dict['<oov>'] = len(full_dict.keys())
        # inv_full_dict[len(full_dict.keys())] = '<oov>'
        # text_words = [inv_full_dict[id] for id in text]  # id转word
        pos_tag = pos_tagger.tag(text)  # 改为直接做词性标注，分词需要词语
        sample_texts = []

        # 按照对f分类的重要性，对句子中的词语排序（参考TF攻击attack_classification_hownet_top5.py中的做法，get importance）
        # get importance score
        # orig_probs = self.text_pred_org([text], batch_size)
        # orig_label = torch.argmax(orig_probs)
        # orig_prob = orig_probs.max()
        # leave_1_texts = [text[:ii] + ['<oov>'] + text[min(ii + 1, len(text)):] for ii in range(len(text))]
        # leave_1_probs = self.text_pred_org(leave_1_texts, batch_size)
        # leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        # a = (leave_1_probs_argmax != orig_label).float()
        # b = leave_1_probs.max(dim=-1)[0]
        # c = torch.index_select(orig_probs[0], 0, leave_1_probs_argmax)
        # d = b - c
        # import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()
        # # 保存按重要性排序的下标（大->小）
        # text_idx_sorted = sorted(range(len(import_scores)), key=lambda k: import_scores[k], reverse=True)

        # Sample
        # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
        text_ids = []
        for word_str in text:
            if word_str in full_dict.keys():
                text_ids.append(full_dict[word_str]) # id
            else:
                text_ids.append(word_str) # str

        text_idx_sorted = list(range(len(text_ids)))###hp添加
        random.shuffle(text_idx_sorted) ##     hp添加

        for s in range(sample_num):
            sample_text_strs = text_ids.copy() # 保存的是采样后的，str
            replace_num = 0.0
            for j in text_idx_sorted:  # 对于每个词语（按重要性排序）
                word = sample_text_strs[j]
                if isinstance(word, str):  # 若当前词语不在词汇表中，一定没有同义词，直接返回
                    continue

                # 若是id：
                pos = pos_tag[j][1]  # 当前词语词性
                if pos not in pos_list:
                    continue
                if pos.startswith('JJ'):
                    pos = 'adj'
                elif pos.startswith('NN'):
                    pos = 'noun'
                elif pos.startswith('RB'):
                    pos = 'adv'
                elif pos.startswith('VB'):
                    pos = 'verb'
                # 获得候选集
                # 1.
                # neigbhours = word_candidate[word][pos]  # 候选集
                # neigbhours.append(word)  # 候选集包含自己
                # 2.
                # 标记当前词语是否在句子的边界上
                if_bound = 'not_bound'
                if j == 0:
                    if_bound = 'left_bound'
                if j == len(sample_text_strs) -1:
                    if_bound = 'right_bound'
                neigbhours = get_candidates(word, pos, word_candidate, tf_vocabulary, text, j, if_bound)
                sample_text_strs[j] = inv_full_dict[choice(neigbhours)]  # 候选集中随机选择一个，并转成str

                replace_num += 1 # 记录替换位置  （可能会替换到原词，此时算不算替换了呢？）
                change_ratio = replace_num / float(len(sample_text_strs))
                if  change_ratio > 0.25: # 若替换比例大于15%，则停止替换 0.1 0.15 0.2 0.25 0.3 不控制，测试集上准确率分别为 _ _ 77.13 75.82 75.84 77.04
                    break

            # id转为str
            for i in range(len(sample_text_strs)):
                if isinstance(sample_text_strs[i], int):
                    sample_text_strs[i] = inv_full_dict[sample_text_strs[i]]

            sample_texts.append(sample_text_strs)
        return sample_texts # list of list of str

    # tiz: 加强的分类器
    # text：[[]]有多条数据，是word
    def text_pred(self, text, sample_num=64,y=-1): # text是str

        probs_boost_all = []
        print('gen_sample_ngram')
        start_time = time.clock()
        perturbed_texts = perturb_texts(self.args, text, self.tf_vocabulary, change_ratio=0.1)
        # perturbed_texts=text
        Samples_x = gen_sample_multiTexts(self.args, perturbed_texts, sample_num, change_ratio=0.25)
        use_time = (time.clock() - start_time)
        print("gen_sample_ngram time used:", use_time)
        Sample_probs = self.text_pred_org(Samples_x)
        lable_mum=Sample_probs.size()[-1]
        Sample_probs=Sample_probs.view(len(text),sample_num,lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l),dim=1)  # 获得预测值的比例作为对应标签的概率
            prob = num / float(sample_num)
            probs_boost.append(prob.view(len(text),1))

        probs_boost_all=torch.cat(probs_boost,dim=1)

        return probs_boost_all


"""对于训练集/测试集中出现的文本，进行采样"""
def gen_sample_multiTexts(args, texts, sample_num, change_ratio = 1):
    Finded_num=0
    all_sample_texts = []  # 返回值。list of list of str，包含输入每个数据的所有周围样本

    for text_str in texts:
        if ' '.join(text_str) in args.candidate_bags.keys():  # seen data (from train or test dataset)
            # 获得候选集
            Finded_num+=1
            candidate_bag = args.candidate_bags[' '.join(text_str)]
            # 产生随机替换
            # sample_texts = np.array([choice(neigbhours, size=sample_num, replace=True) for neigbhours in candidate_bag.values()]).T
            # sample_texts = sample_texts.tolist()
            sample_texts = []
            for ii in range(len(text_str)):
                word_str = text_str[ii]
                r_seed = np.random.rand(sample_num)
                n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                n[np.where(r_seed>change_ratio)]=word_str
                # if r_seed <= change_ratio:
                #     n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                # else:
                #     n = [word_str] * sample_num
                sample_texts.append(n)
            sample_texts = np.array(sample_texts).T.tolist()

        else:  # unseen data
            # 词性标注，耗时
            pos_tag = pos_tagger.tag(text_str)

            # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            # 获得候选集
            candidate_bag = {}
            for j in range(len(text_ids)):  # 对于每个位置
                word = text_ids[j]
                pos = pos_tag[j][1]  # 当前词语词性
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list:
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])  # 候选集
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours] # id转为str
                # 可能一句话中一个词语出现多次

            # 开始采样
            # sample_texts = np.array([choice(neigbhours, size=sample_num, replace=True) for neigbhours in candidate_bag]).T
            # sample_texts = sample_texts.tolist()
            sample_texts = []
            for ii in range(len(text_str)):
                word_str = text_str[ii]
                r_seed = np.random.rand(sample_num)
                n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                n[np.where(r_seed>change_ratio)]=word_str
                sample_texts.append(n)
            sample_texts = np.array(sample_texts).T.tolist()

        all_sample_texts.extend(sample_texts)
    print("{:d}/{:d} texts are finded".format(Finded_num,len(texts)))
    return all_sample_texts

"""利用n gram扰动位置"""
def perturb_texts(args, texts, tf_vocabulary, change_ratio = 1):
    select_sents = []
    for text_str in texts:
        candidate_bag = args.candidate_bags[' '.join(text_str)]
        replace_text = text_str.copy()
        for i in range(len(text_str) - 1):  # 对于每个位置
            # 按概率替换
            r_seed = random.uniform(0, 1)
            if r_seed > change_ratio:
                continue

            candi = candidate_bag[text_str[i]]
            # 若候选集只有自己
            if len(candi) == 1:
                continue
            else:
                max_freq = 0
                best_replace = replace_text[i] # 默认最好的是自己
                two_gram_flag = False # 标记是否有存在于词汇表中的2-gram
                for c in candi:
                    two_gram = c + ' ' + text_str[i + 1]
                    if two_gram in tf_vocabulary.keys():
                        two_gram_flag = True # 存在2gram
                        freq = tf_vocabulary[two_gram]
                        if freq > max_freq:
                            max_freq = freq
                            best_replace = c
                if not two_gram_flag: # 没有见过的2 gram
                    for c in candi:
                        if c in tf_vocabulary.keys():
                            freq = tf_vocabulary[c]
                            if freq > max_freq:
                                max_freq = freq
                                best_replace = c

                replace_text[i] = best_replace

        select_sents.append(replace_text)

    return select_sents

"""根据词语的n-gram频率筛选候选集合，获得替换"""
# def gen_sample_ngram(texts, sample_num, tf_vocabulary, change_ratio = 1):
#     # change_ratio = 0.25
#     # tf_vocabulary = pickle.load(open('data/adversary_training_corpora/%s/tf_vocabulary.pkl' % args.task, "rb"))
#     # test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/%s/test.txt' % args.task, clean=False, FAKE=False, shuffle=False)
#     select_sents = []
#     for text_str in texts:
#         candidate_bag = candidate_bags[' '.join(text_str)]
#         replace_text = text_str.copy()
#         for i in range(len(text_str) -1 ):  # 对于每个位置
#             candi = candidate_bag[text_str[i]]
#             # 若候选集只有自己
#             if len(candi) == 1:
#                 best_replace = replace_text[i]
#             else:
#                 # 获得该位置所有候选集和下一位置组成的所有2 gram的频率
#                 freqs_two_gram = {}
#                 for c in candi:
#                     two_gram = c + ' ' + text_str[i+1]
#                     if two_gram in tf_vocabulary.keys():
#                         freqs_two_gram[two_gram] = tf_vocabulary[two_gram]
#                     else:
#                         freqs_two_gram[two_gram] = 0
#                 if freqs_two_gram:
#                     best_two_gram = sorted(freqs_two_gram.items(), key=lambda item:item[1], reverse=True)[0]
#                     best_replace = best_two_gram[0].split(' ')[0]
#
#                 # 若所有2 gram均未在数据集中出现过，则看1 gram
#                 else:
#                     freqs_one_gram = {}
#                     for c in candi:
#                         if c in tf_vocabulary.keys():
#                             freqs_one_gram[c] = tf_vocabulary[c]
#                         else:
#                             freqs_one_gram[c] = 0
#                     if freqs_one_gram:
#                         best_one_gram = sorted(freqs_one_gram.items(), key=lambda item: item[1], reverse=True)[0]
#                         best_replace = best_one_gram[0]
#                     else:
#                         best_replace = replace_text[i]
#
#             # 按概率替换
#             r_seed = random.uniform(0, 1)
#             if r_seed <= change_ratio:
#                 replace_text[i] = best_replace
#
#         # select_sents[' '.join(text_str)] = replace_text
#         select_sents.append(replace_text)

    # 对于每个位置的词语，将其替换成它的同义词，获得词频，筛选词频最高的那个，保存下来，作为替换文本
    # select_sents = {}
    # for text_str in test_x:
    #     if ' '.join(text_str) in candidate_bags.keys():  # seen data (from train or test dataset)
    #         # 获得候选集
    #         # candidate_bag = candidate_bags[' '.join(text_str)]
    #         # 候选集只保留change_ratio，其余全为原词
    #         candidate_bag = {}
    #         for i in range(len(text_str)):
    #             word_str = text_str[i]
    #             r_seed = random.uniform(0, 1)
    #             if r_seed <= change_ratio:
    #                 candidate_bag[word_str] = candidate_bags[' '.join(text_str)][word_str]
    #             else:
    #                 candidate_bag[word_str] = [word_str]
    #
    #         # 获得所有可能的替换
    #         syn_space_size = 1
    #         for v in candidate_bag.values():
    #             syn_space_size = syn_space_size *len(v)
    #         syn_space = []
    #         for ii in range(len(candidate_bag)):
    #             word_str = text_str[ii]
    #             syn_space.append(candidate_bag[word_str] * int((syn_space_size/len(candidate_bag[word_str]))))
    #         syn_space = np.array(syn_space).T
    #
    #         # 选择2-gram乘积最大的替换
    #         spc_feq = []
    #         for sent in syn_space:
    #             sent_freq = 1.0
    #             for i in range(len(sent)-1):
    #                 ngram = sent[i] + ' ' + sent[i+1]
    #                 if ngram in tf_vocabulary.keys():
    #                     # sent_freq = sent_freq * tf_vocabulary[ngram] # 频率相乘容易溢出，改为相加
    #                     sent_freq = sent_freq  + tf_vocabulary[ngram]
    #             spc_feq.append(sent_freq)
    #         select_sent = syn_space[np.argmax(spc_feq), :].tolist()
    #         select_sents[' '.join(text_str)] = select_sent

    # out_file = 'data/adversary_training_corpora/%s/replace_ngram_test.pkl' % args.task
    # f = open(out_file, 'wb')
    # pickle.dump(select_sents, f)

    # 在替换后的样本附近采样
    #sample_texts = gen_sample_multiTexts(select_sents, sample_num, change_ratio)

    #return sample_texts





def eval_model0(args, model, input_x, input_y): # 输入为未划分batch的str list
    model.eval()
    with torch.no_grad():
        output = model.text_pred_org(input_x, batch_size=128)
        acc = float(torch.sum(torch.eq(torch.argmax(output, dim=1), torch.Tensor(input_y).cuda())).data) / float(len(input_x))
    #model.train()
    return acc

# textfooler的
def eval_model(niter, model, input_x, input_y):
    # 我加的
    # input_x, input_y = dataloader.create_batches(input_x, input_y, args.max_seq_length, args.batch_size, model.word2id, )
    input_x, input_y = dataloader.create_batches(input_x, input_y, args.batch_size, model.word2id, )

    model.eval()
    # N = len(valid_x)
    # criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.
    # total_loss = 0.0
    with torch.no_grad():
        for x, y in zip(input_x, input_y):
            x, y = Variable(x, volatile=True), Variable(y)
            output = model(x)
            # loss = criterion(output, y)
            # total_loss += loss.item()*x.size(1)
            pred = output.data.max(1)[1]
            correct += pred.eq(y.data).cpu().sum()
            cnt += y.numel()
    model.train()
    return correct.item()/cnt

# def eval_model(args, model, input_x, input_y): # input_x:未划分batch的二维list，str
#     model.eval()
#     # N = len(valid_x)
#     # criterion = nn.CrossEntropyLoss()
#     batch_size=64
#     correct = 0.0
#     acc=0.0
#     correct0=0.0
#     cnt = 0.
#     data_size = len(input_y)
#     # total_loss = 0.0
#     with torch.no_grad():
#         for step in range(0, data_size, batch_size):
#             input_x1 = input_x[step:min(step + batch_size, data_size)]  # 取一个batch
#             input_y1 = input_y[step:min(step + batch_size, data_size)]
#
#             output = model.text_pred(input_x1)
#             pred = torch.argmax(output, dim=1)
#             correct+=torch.sum(torch.eq(pred,torch.tensor(input_y1).cuda()))
#         acc=(correct.cpu().numpy())/float(data_size)
#     return acc

# train_x, train_y：划分batch
# test_x，test_y：原始数据
def train_model(args, epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):
    model.train()
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y): # 对于每个batch
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    test_acc = eval_model0(model, test_x, test_y)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(epoch, niter,optimizer.param_groups[0]['lr'],loss.item(),test_acc))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
            print('save model when test acc=', best_test)

    return best_test

# HP adds: train_x is str
def train_model1(args, epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):
    print('train:{:d}'.format(len(train_x)))
    criterion = nn.CrossEntropyLoss()
    test_acc = eval_model0(args, model, test_x, test_y)
    print('before train: acc={:.6f}'.format(test_acc))
    cnt = 0
    batch_size=128
    sample_size=20
    niter = epoch * len(train_x)//batch_size

    for step in range(0,len(train_x),batch_size):
        #print('step={:d}'.format(step//batch_size))
        train_x1=train_x[step:min(step+batch_size,len(train_x))] #取一个batch
        train_y1=train_y[step:min(step+batch_size,len(train_x))]

        niter += 1
        cnt += 1
        #print('gen_sample_multiTexts')
        start_time = time.clock()
        Samples_x=gen_sample_multiTexts(args, train_x1, sample_size)   #sample_size for each point
        use_time = (time.clock() - start_time)
        # print("gen_sample_multiTexts time used:", use_time)
        Samples_y=[l for l in train_y1 for i in range(sample_size)] #每个lable复制sample_size 次
        #print('text_pred_org')
        Sample_probs=model.text_pred_org(Samples_x)
        S_result=torch.eq(torch.argmax(Sample_probs, dim=1), torch.Tensor(Samples_y).cuda()).view(len(train_y1),sample_size)
        R_score=torch.sum(S_result,dim=1).view(-1)/float(sample_size) #每个训练点的鲁棒打分
        #print(R_score)


        adv_batch_x=[]
        adv_batch_y=[]
        for i in range(R_score.size()[0]):
            if R_score[i]<2.0/3:
                adv_count=1
                for j in range(sample_size):
                    if S_result[i][j].data!=train_y1[i]:
                        adv_batch_x.append(Samples_x[i*sample_size+j])
                        adv_batch_y.append(train_y1[i])
                        adv_count=adv_count-1
                        if adv_count==0:
                            break
        #print('filt_adv')

        adv_x, adv_y = dataloader.create_batches(adv_batch_x, adv_batch_y, args.max_seq_length, batch_size,
                                                     model.word2id, )
        adv_x = torch.cat(adv_x, dim=0)
        adv_y= torch.cat(adv_y, dim=0)
        model.train()
        model.zero_grad()
        #adv_loss=0
        #计算对抗样本产生的loss
        adv_x, adv_y = Variable(adv_x), Variable(adv_y)
        output = model(adv_x)
        output = torch.reshape(output, (-1, args.num_class))  # 当输入只有一个元素时，output只有一维。需要将其拉成两维

        adv_loss =criterion(output, adv_y)

        #print('finishid adv_loss={:.6f}'.format(adv_loss.item()))

        train_x1, train_y1 = dataloader.create_batches(train_x1, train_y1, args.max_seq_length, batch_size,
                                                     model.word2id, )

        train_x1 = torch.cat(train_x1, dim=0)
        train_y1= torch.cat(train_y1, dim=0)

        train_x1, train_y1 = Variable(train_x1), Variable(train_y1)
        output = model(train_x1)
        Norm_loss =criterion(output, train_y1)

        #print('finishid Norm_loss={:.6f}'.format(Norm_loss.item()))

        loss=0.5*Norm_loss+0.5*adv_loss
        #loss=Norm_loss
        loss.backward()
        optimizer.step()

    # for x, y in zip(train_x, train_y): # 对于每个batch
    #
    #
    #     model.train()
    #     model.zero_grad()
    #     x, y = Variable(x), Variable(y)
    #     output = model(x)
    #     loss = criterion(output, y)
    #     loss.backward()
    #     optimizer.step()

    test_acc = eval_model0(args, model, test_x, test_y)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(epoch, niter,optimizer.param_groups[0]['lr'],loss.item(),test_acc))

    # if test_acc > best_test:
    #     best_test = test_acc
    #     if save_path:
    #         torch.save(model.state_dict(), save_path)
    #         print('save model when test acc=', best_test)
    if save_path:
        torch.save(model.state_dict(), save_path)
        print('save model when test acc=', best_test)
    return best_test

def split_weight(train_y):
    lable=[]
    weight=[]
    for batch_y in train_y:
        for i in range(len(batch_y)):
            if batch_y[i].data>0 and batch_y[i].data<1:
                weight.append(batch_y[i].data)
                lable.append(1)
            elif batch_y<0:
                weight.append(-1*batch_y[i].data)
                lable.append(0)
            else:
                weight.append(1)
                lable.append(batch_y[i].data)
    lable=torch.tensor(lable)
    weight=torch.tensor(weight)
    return lable,weight

def save_data(data, labels, path, type='train'):
    with open(os.path.join(path, type+'.txt'), 'w') as ofile:
        for text, label in zip(data, labels):
            ofile.write('{} {}\n'.format(label, ' '.join(text)))

def main(args):
    print('Load data...')
    if args.dataset == 'mr':
        train_x, train_y = dataloader.read_corpus('data/adversary_training_corpora/mr/train.txt')
        test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test.txt')  # 为了观察，暂时不shuffle
    elif args.dataset == 'imdb':
        train_x, train_y = dataloader.read_corpus(os.path.join(args.data_path + 'imdb', 'train_tok.csv'), clean=False, FAKE=False, shuffle=True)
        test_x, test_y = dataloader.read_corpus(os.path.join(args.data_path + 'imdb', 'test_tok.csv'), clean=False, FAKE=False, shuffle=True)
    elif args.dataset == 'fake':
        train_x, train_y = dataloader.read_corpus(args.data_path + '{}/train_tok.csv'.format(args.dataset), clean=False, FAKE=True, shuffle=True)
        # 关于fake的测试集
        # test_x, test_y = dataloader.read_corpus(args.data_path + '{}/test_tok.csv'.format(args.dataset), clean=False, FAKE=True, shuffle=True) # 原本是从文件中读取
        # test_y = [1-y for y in test_y] # tiz: 反转测试集标签试试 --> 更不对了
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=1) # tiz: 从训练集中获得测试集试试 --> 正常了
    elif args.dataset == 'mr_adv':
       test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test_adv.txt', clean = False, FAKE = False, shuffle = True)

    # nclasses = max(test_y) + 1

    print('Num of testset: %d' % (len(test_y)))

    print('Build model...')
    model = Model(args, args.max_seq_length, args.embedding, args.d, args.depth, args.dropout, args.cnn, args.nclasses).cuda() # tiz: 512 -->args.max_seq_length
    if args.target_model_path is not None:  # tiz
        print('Load pretrained model...')
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)

    if args.mode == 'eval':  # tiz
        print('Eval...')
        # test_x, test_y = dataloader.create_batches(test_x, test_y, args.max_seq_length,args.batch_size, model.word2id, ) # tiz: 512 -->args.max_seq_length
        test_acc0=eval_model0(args, model, test_x, test_y)
        print('Base classifier f acc:{:.6f}'.format(test_acc0))
        test_acc = eval_model(args, model, test_x, test_y)
        print('Test acc: ', test_acc)
    else:
        print('Train...')
        train_acc = eval_model(args, model, train_x, train_y)
        print('Original train acc: ', train_acc)

        #train_x, train_y = dataloader.create_batches(train_x, train_y, args.max_seq_length, args.batch_size, model.word2id, ) # tiz: 512 -->args.max_seq_length

        need_grad = lambda x: x.requires_grad
        optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)

        best_test = 0
        # test_err = 1e+8
        for epoch in range(args.max_epoch):
            best_test = train_model1(args, epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)
            if args.lr_decay > 0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay
        sys.stdout.write("test_err: {:.6f}\n".format(best_test))

# tiz：将从官方下载的fake数据集中的test.csv和submit合并，获得测试集标签，写入test_tok.csv中
def get_test_label():
    test = pd.read_csv('data/adversary_training_corpora/fake/test.csv')['text'].tolist()
    print(pd.read_csv('data/adversary_training_corpora/fake/test.csv').isna().sum())
    labels = pd.read_csv('data/adversary_training_corpora/fake/submit.csv')['label'].tolist()
    with open('data/adversary_training_corpora/fake/test_tok.csv','w') as w:
        for i in range(len(test)):
            try:
                test[i] = test[i].replace('\n','')
                w.write(test[i] + ','+str(labels[i]+1) + '\n')
            except:
                print(test[i])


if __name__ == "__main__":
    # gen_texts_candidates()

    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--data_path", type=str, default="data/adversary_training_corpora/",
                           help="where to load dataset, parent dir")
    argparser.add_argument("--dataset", type=str, default="mr",
                           help="which dataset")  # when ends with "_adv" means loading adversarial examples.
    argparser.add_argument("--embedding", type=str, default='glove.6B/glove.6B.200d.txt', help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=128)
    argparser.add_argument("--max_epoch", type=int, default=70)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.0001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='models/wordLSTM/mr_new')
    argparser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument("--target_model_path", type=str, default=None, help='where to load pretrained model')  # tiz
    argparser.add_argument("--mode", type=str, default='eval', help='train, eval')  # tiz,
    argparser.add_argument("--max_seq_length", default=128, type=int,
                           help="max sequence length for target model")  # tiz
    argparser.add_argument("--task", type=str, default='mr', help="task name: mr/imdb/fake")  # tiz
    argparser.add_argument("--nclasses", type=int, default='2')
    args = argparser.parse_args()

    # tiz
    num_class = {'imdb': 2, 'mr': 2, 'fake': 2}
    args.num_class = num_class[args.task]
    seq_len_list = {'imdb': 256, 'mr': 128, 'fake': 512}
    args.max_seq_length = seq_len_list[args.task]
    sizes = {'imdb': 50000, 'mr': 20000, 'fake': 50000}
    args.max_vocab_size = sizes[args.task]
    # args.target_model_path = ('models/wordLSTM/%s' % args.task)  # 注：该代码只针对wordLSTM的训练等
    with open('data/adversary_training_corpora/%s/dataset_%d.pkl' % (args.task, args.max_vocab_size), 'rb') as f:
        args.datasets = pickle.load(f)
    with open('data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
        args.word_candidate = pickle.load(fp)
    with open('data/adversary_training_corpora/%s/pos_tags_test.pkl' % args.task, 'rb') as fp:  # 针对测试集获得对抗样本
        args.pos_tags_test = pickle.load(fp)
    # load candidate
    with open('data/adversary_training_corpora/%s/candidates_train.pkl' % args.task, 'rb') as fp:
        candidate_bags_train = pickle.load(fp)
    with open('data/adversary_training_corpora/%s/candidates_test.pkl' % args.task, 'rb') as fp:
        candidate_bags_test = pickle.load(fp)
    args.candidate_bags = {**candidate_bags_train, **candidate_bags_test}

    args.pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    args.inv_full_dict = args.datasets.inv_full_dict
    args.full_dict = args.datasets.full_dict

    args.full_dict['<oov>'] = len(args.full_dict.keys())
    args.inv_full_dict[len(args.full_dict.keys())] = '<oov>'

    torch.cuda.set_device(args.gpu_id)
    main(args)


