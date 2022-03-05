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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import combinations, permutations
from tqdm import tqdm
import pickle
import nltk
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *

from nltk.tag import StanfordPOSTagger
jar = '/pub/data/huangpei/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
model = '/pub/data/huangpei/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# args
parser = argparse.ArgumentParser()
parser.add_argument("--sample_num",
                    type=int,
                    default=3000)
parser.add_argument("--train",
                    type=bool,
                    default=False)
parser.add_argument("--nclasses",
                    type=int,
                    default=2,
                    help="How many classes for classification.")
parser.add_argument("--task",
                    type=str,
                    default='mr')
parser.add_argument("--target_model",
                    type=str,
                    default='wordCNN',
                    help="Target models for text classification: fasttext, charcnn, word level lstm "
                         "For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--target_model_path",
                    type=str,
                    default='./pretrained_model/imdb',
                    help="pre-trained target model path")
parser.add_argument("--word_embeddings_path",
                    type=str,
                    default='./glove.6B/glove.6B.300d.txt',
                    help="path to the word embeddings for the target model")
parser.add_argument("--counter_fitting_embeddings_path",
                    type=str,
                    default='./data/counter-fitted-vectors.txt',
                    help="path to the counter-fitting embeddings we used to find synonyms")
parser.add_argument("--counter_fitting_cos_sim_path",
                    type=str,
                    default='./cos_sim_counter_fitting.npy',
                    help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
parser.add_argument("--USE_cache_path",
                    type=str,
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
parser.add_argument("--data_size", default=1000, type=int, help="Data size to create adversaries")
args = parser.parse_args()


args.dataset_path = 'data/adversary_training_corpora/' + args.task
if args.train:
    args.ca_dir = args.dataset_path + '/fastca_hownet/%s/train' % args.target_model
else:
    args.ca_dir = args.dataset_path + '/fastca_hownet/%s/test' % args.target_model
if not os.path.exists(args.ca_dir):
    os.makedirs(args.ca_dir)

sizes = {'imdb': 50000, 'mr': 20000}
max_vocab_size = sizes[args.task]


user_path = '/pub/data/huangpei/'
with open(args.dataset_path + '/dataset_%d.pkl' % max_vocab_size, 'rb') as f:
    dataset = pickle.load(f)
with open(args.dataset_path + '/word_candidates_sense.pkl', 'rb') as fp:
    word_candidate = pickle.load(fp)
with open(args.dataset_path + '/pos_tags.pkl', 'rb') as fp:
    pos_tags_train = pickle.load(fp)  # 保存所有文本所有单词的词性
with open(args.dataset_path + '/pos_tags_test.pkl', 'rb') as fp:
    pos_tags_test = pickle.load(fp)  # 保存所有文本所有单词的词性

if args.train:
    pos_tags = pos_tags_train
else:
    pos_tags = pos_tags_test

inv_full_dict = dataset.inv_full_dict
full_dict = dataset.dict
np.random.seed(3333)
tf.set_random_seed(3333)
pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def example_ca():
    text = 'this is the best movie i have ever seen'
    text = text.split(' ')
    print(text)
    pos_tags = pos_tagger.tag(text)
    print(pos_tags)
    text_ids = [full_dict[w] for w in text]
    syn_num = []
    for i in range(len(text)):
        pos = pos_tags[i][1]
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
        syns = word_candidate[text_ids[i]][pos]
        print(syns)
        print([inv_full_dict[w] for w in syns])
    return None

"""加载模型"""
def load_model():
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
    print("Model built!")

    return model


"""加载原始数据"""
def load_data():
    jar = user_path + 'stanford-postagger-full-2020-11-17/stanford-postagger.jar'
    model = user_path + 'stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
    wnl = WordNetLemmatizer()

    # get data
    texts, labels = dataloader.read_corpus(args.dataset_path)
    texts, labels = texts[:args.data_size], labels[:args.data_size]
    # 对数据做词性标注和词干还原，并删除符号
    lemma_texts = []
    for j in tqdm(range(len(texts))):
        text = texts[j]
        pos_tags = pos_tagger.tag(text)
        for i in range(len(text)):
            pos = pos_tags[i][1]
            if pos.startswith('N'):
                text[i] = wnl.lemmatize(text[i], pos='n')
            elif pos.startswith('V'):
                text[i] = wnl.lemmatize(text[i], pos='v')
            elif pos.startswith('J'):
                text[i] = wnl.lemmatize(text[i], pos='a')
            elif pos.startswith('R'):
                text[i] = wnl.lemmatize(text[i], pos='r')
            else:
                text[i] = wnl.lemmatize(text[i])
        lemma_texts.append(text)
    print("Data import finished!")

    return lemma_texts, labels


"""加载同义词词典，并写入./data/English_syn.json文件"""
def get_syn():
    syn_file = './data/English_syn.txt'
    syn_dict = {} # 保存同义词词典
    # 扫描文件，获得词典
    lines = open(syn_file, 'r').readlines()
    for i in range(len(lines)):
        if 'KEY: ' in lines[i]:
            key = lines[i].strip().replace('KEY: ', '').replace('.', '').split(' ')[0]
        elif 'SYN: ' in lines[i]:
            syns = lines[i].strip().replace('SYN: ', '').replace('.', '').split(',')
        elif '=' in lines[i]:
            try:
                key = key.lower()
                syns = [syn.lower() for syn in syns]
            except:
                pass
            syn_dict[key] = syns
            key = None
            syns = None
        i += 1

    # 处理词典（处理同义词中的联系词）
    for word in syn_dict.copy():
        syns = syn_dict[word]
        # print(word,syns)
        # 删除同义词为空的
        if syns is None:
            del syn_dict[word]
            continue
        # 更新同义词中包含联系词的
        try:
            i = syns.index(r'[See*')
        except:
            continue
        to_look_up = syns[i].split(' ')[1].replace(']', '')  # 待联系词
        to_look_up_syns = syn_dict[to_look_up]  # 待联系词的同义词
        syn_dict[word] = syns + to_look_up_syns  # 更新当前词的同义词（同时包含待联系词的同义词）
    print('Size of synonymous dict: %d' % len(syn_dict))
    # write ro json
    with open('./data/English_syn.json', 'w') as f:
        json.dump(syn_dict, f)
    return


"""给定一条文本，获得每个位置同义词个数等信息，写入文件./data/test_text.txt，作为fastCA的输入；
获得该文本各个位置同义词列表，用于结合CA映射回词语，进一步获得替换后文本"""
def pre_for_fastCA_dict(text):
    # 1. 获取待测试文本各个位置的同义词个数
    syn_dict = json.load(open('data/English_syn.json', 'r'))
    syn_words_num = []  # 保存该文本各个位置同义词个数
    text_syns = []  # 保存该文本各个位置同义词（获取CA后，用其映射回词语）
    for word in text:
        if word in syn_dict.keys():  # 若有同义词
            syns = syn_dict[word]
            num = len(syns) + 1  # 1代表该词本身
            text_syns.append([word] + syn_dict[word])
        else:
            num = 1
            text_syns.append([word])
        syn_words_num.append(num)
    # 2. 写入文件
    file = './data/test_text.txt'
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
def pre_for_fastCA_wordnet(text):
    # 1. 获取待测试文本各个位置的同义词个数
    syn_words_num = []  # 保存该文本各个位置同义词个数
    text_syns = []  # 保存该文本各个位置同义词（获取CA后，用其映射回词语）
    for word in text:
        synonyms = wordnet.synsets(word)
        lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        lemmas.sort()
        if len(lemmas) == 0:
            num = 1
            text_syns.append([word])
        else:
            num = len(lemmas)
            text_syns.append(lemmas)
        syn_words_num.append(num)
    # 2. 写入文件
    file = './data/test_text.txt'
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
def pre_for_fastCA_hownet(idx, text):
    # fastca_ana = open(args.ca_dir + '/pertub_pos.txt', 'w')  # 统计所有数据的可替换位置个数
    text = [x for x in text if x != 5169]  # 注：文本里有5169(\x85)，而词性标注结果没有，导致无法对齐。将其删除
    # 1. 加载数据及候选替换集
    pos_tag = pos_tags[idx]  # 当前文本所有单词词性
    # 2. 获取待测试文本各个位置的同义词个数
    syn_words_num = [1] * len(text)  # 保存该文本各个位置同义词个数（包含自己）
    text_syns = [[t] for t in text]  # 保存该文本各个位置同义词（包含自己。获取CA后，用其映射回词语）
    pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    pertub_psts = []  # 保存该文本的所有可替换位置
    for i in range(len(text)):
        pos = pos_tag[i][1]  # 当前词语词性
        # 若当前词语词性不为形容词、名词、副词和动词，不替换
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
        neigbhours = word_candidate[text[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
        if len(neigbhours) >0:
            pertub_psts.append(i)
        text_syns[i] += neigbhours
        syn_words_num[i] += len(neigbhours)
    # 2. 写入文件
    # 若无可替换位置，不写
    if len(pertub_psts) == 0:
        return None
    # 写入该文本的替换位置和替换数量
    print('%d data has %d in %d pertub positions.' % (idx, len(pertub_psts), len(text)))
    # print('%d data has %d in %d pertub positions.' % (idx, len(pertub_psts), len(text)), file=fastca_ana)
    # print(pertub_psts)
    # print(pertub_psts, file=fastca_ana)
    # fastca_ana.flush()
    fastca_in_file = args.ca_dir + '/' + str(idx) + '_fastca_in.txt' # fastca所需文件
    t = '2'  # 2-路覆盖
    with open(fastca_in_file, 'w') as w:
        w.write(t + '\n')  # 第一行：写入t
        w.write(str(len(text)) + '\n')  # 第二行：写入文本长度（词语个数）
        syn_words_num = [str(sw) for sw in syn_words_num]
        w.write(' '.join(syn_words_num) + '\n')  # 第三行：写入所有位置同义词个数
        w.close()
    return text_syns


def get_all_fastcaInfo():
    # 加载数据集
    # 测试集
    test_x = dataset.test_seqs2
    # 训练集
    train_x = dataset.train_seqs2

    if args.train:
        # 取出不鲁棒数据
        train_robust_file = args.dataset_path + '/robust_samp%d_train25_%s.txt' % (args.sample_num, args.target_model)
        train_robust = open(train_robust_file, 'r').read().splitlines()
        train_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in train_robust]
        non_robust_id = np.nonzero(np.array(train_robust))[0]
        # 不鲁棒数据写入文件
        with open(args.ca_dir + '/non_robust_file.txt','w') as fp:
            print('Non robust file:', non_robust_id)
            print('Non robust file:', non_robust_id, file=fp)
            fp.flush()
        # 获得一条数据的fastca input数据
        for idx in non_robust_id:
            text = train_x[idx]
            _ = pre_for_fastCA_hownet(idx, text)
    else:
        # 取出不鲁棒数据
        test_robust_file = args.dataset_path + '/robust_samp%d_test25_%s.txt' % (args.sample_num, args.target_model)
        test_robust = open(test_robust_file, 'r').read().splitlines()
        test_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in test_robust]
        non_robust_id = np.nonzero(np.array(test_robust))[0]
        with open(args.ca_dir + '/non_robust_file.txt', 'w') as fp:
            print('Non robust file:', non_robust_id)
            print('Non robust file:', non_robust_id, file=fp)
            fp.flush()
        # 获得fastca input
        # 获得一条数据的fastca input数据
        for idx in non_robust_id:
                text = test_x[idx]
                _ = pre_for_fastCA_hownet(idx, text)

"""运行fastCA，将获得的覆盖数组转化为文本，并写入文件"""
def get_replace_text(idx, text, true_label):
    # 1. 准备fastCA输入文件，并获得同义词列表
    # text_syn = pre_for_fastCA_dict(text)
    # text_syn = pre_for_fastCA_wordnet(text)
    text_syn = pre_for_fastCA_hownet(idx, text)
    text_syn_flatten = [w for s in text_syn for w in s]  # 将该文本所有位置的同义词（包含自己）按顺序拉伸成一维列表
    # 2. 运行fastCA
    print('Run fastCA...')  # 输入：data/test_text.txt 输出：data/CA_array.txt
    os.system('/pub/data/huangpei/fastca/fastCA/FastCA /pub/data/huangpei/TextFooler/%s 100 1' % args.ca_dir + '/fastca_in.txt')

    # 3. 读取覆盖数组，转化为文本，并写入文件
    fastca_out_file = open(args.ca_dir + '/fastca_out.txt', 'w')
    # file = 'data/CA_array.txt'
    # out_file = 'adv_results/imdb_fastca_dict/' + str(idx) +'.txt'
    # out_file = 'adv_results/imdb_fastca_wordnet/' + str(idx) +'.txt'
    out_file = args.ca_dir + '/' + str(idx) +'.txt'
    w = open(out_file, 'w')
    # 第一行写入真实标签和原始文本（lemm之后）
    # w.write(str(true_label) + ' ' + ' '.join([inv_full_dict[i] for i in text]) + '\n' )
    # w.write(str(true_label) + ' ' + ' '.join(text) + '\n' )
    w.write(str(true_label) + ' ' + ' '.join([str(t) for t in text]) + '\n')

    lines = open(fastca_out_file, 'r').readlines()
    all_replace_text = []
    for i in range(len(lines)):
        replace_text = []
        line = [int(n) for n in lines[i].strip().split(' ')]
        for j in line:
            replace_word = text_syn_flatten[j]
            replace_text.append(replace_word)
        all_replace_text.append(replace_text)
        w.write(str(true_label) + ' ' + ' '.join([str(i) for i in replace_text])+'\n')  # 写入真实标签和替换文本(编号)
        w.flush()
    w.close()
    print('%d data wrote finished' % idx)
    return all_replace_text


"""获得所有替换文本"""
def get_replace_text_all():
    # 加载数据集
    # 测试集
    test_x = dataset.test_seqs2
    test_y = dataset.test_y
    # 训练集
    train_x = dataset.train_seqs2
    train_y = dataset.train_y

    if args.train:
        # 只对不鲁棒的点获得覆盖数组
        # 取出不鲁棒数据
        train_robust_file = args.dataset_path + '/robust_samp%d_train25_%s.txt' % (args.sample_num, args.target_model)
        train_robust = open(train_robust_file, 'r').read().splitlines()
        train_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in train_robust]
        non_robust_id = np.nonzero(np.array(train_robust)).tolist()
        train_x = train_x[non_robust_id]
        train_y = train_y[non_robust_id]
        for idx in range(len(train_y)):
            text, true_label = train_x[idx], train_y[idx]
            _ = get_replace_text(idx, text, true_label)
    else:
        # 只对不鲁棒的点获得覆盖数组
        # 取出不鲁棒数据
        test_robust_file = args.dataset_path + '/robust_samp%d_test25_%s.txt' % (args.sample_num, args.target_model)
        test_robust = open(test_robust_file, 'r').read().splitlines()
        test_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in test_robust]
        non_robust_id = np.nonzero(np.array(test_robust))[0]
        with open(args.dataset_path + '/non_robust_file_samp%d_test25_%s.txt' % (args.sample_num, args.target_model), 'w') as fp:
            print('Non robust file:', non_robust_id)
            print('Non robust file:', non_robust_id, file=fp)
            fp.flush()

        # 获得覆盖数组
        for idx in non_robust_id:
                text, true_label = test_x[idx], test_y[idx]
                _ = get_replace_text(idx, text, true_label)


"""针对所有替换样本，获得攻击结果"""
def attack(attack_data_dir, result_file):
    predictor = load_model()
    adv_w = open(result_file, 'w')
    ori_predict_true = 0
    change_rates = []  # 记录对于各个测试数据，覆盖数组中改变label的数据比例
    num_adv = 0  # 记录所有替换样本的个数（）
    changed_adv = 0  # 记录所有替换文本中，会改变label的个数
    filenames = os.listdir(attack_data_dir)
    filenames.sort()
    for file in filenames:  # 对于每条数据
        print('---%s---' % file.replace('.txt', ''))
        print('---%s---' % file.replace('.txt', ''), file=adv_w)
        data = open(attack_data_dir + file, 'r').readlines()
        true_label = int(data[0].split(' ')[0])
        ori_text = data[0].split(' ')[1:]
        # first check the prediction of the original text
        orig_probs, _ = predictor([ori_text])
        orig_probs = orig_probs.squeeze()
        orig_label = torch.argmax(orig_probs)
        if orig_label != true_label:  # 预测错误
            print('Predict false!')
            change_rates.append(-1)
            adv_w.write('Predict false.\n')
        else:
            ori_predict_true += 1
            label_change_replace = 0  # 保存对于该测试样本，覆盖数组中label变化的个数
            all_replace_text = data[1:]  # 除了第一行是原始数据
            num_adv += len(all_replace_text)
            for replace_text in all_replace_text:
                replace_text = replace_text.split(' ')
                replace_probs, _ = predictor([replace_text])
                replace_probs = replace_probs.squeeze()
                replace_label = torch.argmax(replace_probs)
                if orig_label != replace_label:
                    label_change_replace += 1
                    changed_adv += 1
                    adv_w.write('1 ')  # 标志当前替换文本会改变label
                else:
                    adv_w.write('0 ')
            adv_w.write('\n ')
            print('Num of replace text: %d' % len(all_replace_text))
            print('Num of replace text: %d' % len(all_replace_text), file=adv_w)
            change_rate = float(label_change_replace) / float(len(all_replace_text))  # 覆盖数组中，会改变label的所占比例
            change_rates.append(change_rate)
            print('Rates of label changed for all replace text: %f' % change_rate)
            print('Rates of label changed for all replace text: %f' % change_rate, file=adv_w)

    print('*****************End of attack*****************', file=adv_w)
    print('Origin data size:%d, Origin predict true:%d, Attack success:%d'
          % (len(filenames), ori_predict_true, len(change_rates) - change_rates.count(-1) - change_rates.count(0)))
    print('Origin data size:%d, Origin predict true:%d, Attack success:%d, All replace text num:%d, Label changed in all replace text:%d'
        % (len(filenames), ori_predict_true, len(change_rates) - change_rates.count(-1) - change_rates.count(0), num_adv, changed_adv), file=adv_w)

    adv_w.close()
def attack_bdlstm(attack_data_dir, result_file):
    predictor = load_model()
    adv_w = open(result_file, 'w')
    ori_predict_true = 0
    change_rates = []  # 记录对于各个测试数据，覆盖数组中改变label的数据比例
    num_adv = 0  # 记录所有替换样本的个数（）
    changed_adv = 0  # 记录所有替换文本中，会改变label的个数
    filenames = os.listdir(attack_data_dir)
    filenames.sort(key=lambda x: int(x[:-4]))  # 按文件名（数字）排序
    j =0
    for file in filenames:  # 对于每条数据
        j += 1
        # print('%s' % file.replace('.txt', ''))
        # print('%s' % file.replace('.txt', ''), file=adv_w)
        data = open(attack_data_dir + file, 'r').readlines()
        true_label = int(data[0].strip().split(' ')[0])
        ori_text = data[0].strip().split(' ')[1:]
        ori_text = pad_sequences([ori_text], maxlen=250, padding='post')
        # first check the prediction of the original text
        orig_probs = predictor.predict(ori_text)[0]
        orig_label = np.argmax(orig_probs)
        if orig_label != true_label:  # 预测错误
            print('%s -1' % file.replace('.txt', ''))
            print('%s -1' % file.replace('.txt', ''), file=adv_w)
            change_rates.append(-1)
        else:
            ori_predict_true += 1
            label_change_replace = 0  # 保存对于该测试样本，覆盖数组中label变化的个数
            all_replace_text = [d.strip().split(' ')[1:] for d in data]  # 去除掉第一个位置，是标签
            for replace_text in all_replace_text:
                replace_text = pad_sequences([replace_text], maxlen=250, padding='post')
                replace_probs = predictor.predict(replace_text)[0]
                replace_label = np.argmax(replace_probs)
                if orig_label != replace_label:
                    label_change_replace += 1
                    changed_adv += 1
                    # adv_w.write('1 ')  # 标志当前替换文本会改变label
                else:
                    # adv_w.write('0 ')
                    pass

            num_adv += len(all_replace_text)
            change_rate = float(label_change_replace) / float(len(all_replace_text))  # 覆盖数组中，会改变label的所占比例
            # adv_w.write('\n ')
            # print('Num of replace text: %d' % len(all_replace_text))
            # print('Num of replace text: %d' % len(all_replace_text), file=adv_w)
            change_rates.append(change_rate)
            # print('Rates of label changed for all replace text: %f' % change_rate)
            # print('Rates of label changed for all replace text: %f' % change_rate, file=adv_w)
            print('%s %d %d' % (file.replace('.txt', ''), len(all_replace_text), label_change_replace))
            print('%s %d %d' % (file.replace('.txt', ''), len(all_replace_text), label_change_replace), file=adv_w)
        adv_w.flush()

    # print('*****************End of attack*****************', file=adv_w)
    # print('Origin data size:%d, Origin predict true:%d, Attack success:%d'
    #       % (len(filenames), ori_predict_true, len(change_rates) - change_rates.count(-1) - change_rates.count(0)))
    # print('Origin data size:%d, Origin predict true:%d, Attack success:%d, All replace text num:%d, Label changed in all replace text:%d'
    #     % (len(filenames), ori_predict_true, len(change_rates) - change_rates.count(-1) - change_rates.count(0), num_adv, changed_adv), file=adv_w)

    adv_w.close()


"""对于一条文本，获取它周围的一组随机数据（基于wordnet）"""
def get_text_around(text, sample_num =5):
    text = text.split(' ')
    all_sample_text = []
    
    # 标记哪些位置有同义词
    has_syn = []
    for i in range(len(text)):
        word = text[i]
        # 获得该word的同义词
        synonyms = wordnet.synsets(word)
        lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        lemmas.sort()
        if len(lemmas) > 0:
            has_syn.append(1)
        else:
            has_syn.append(0)
    has_syn_psts = list(np.where(np.array(has_syn) == 1)[0])
    
    # 从有同义词的位置中，随机选择sample_num个位置
    psts = random.sample(has_syn_psts, sample_num)
    for pst in psts:
        sample_text = text.copy()
        word = text[pst]
        # 获得该word的同义词
        synonyms = wordnet.synsets(word)
        lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        lemmas.sort()
        
        # 随机选取一个同义词并替换
        sample_word = random.sample(lemmas, 1)[0]
        sample_text[pst] = sample_word
        # 得到新的替换文本
        all_sample_text.append(' '.join(sample_text))
    return all_sample_text
def get_text_around_all():
    i_data_path = 'adv_results/imdb_fastca_wordnet/'
    o_data_path = 'adv_results/imdb_fastca_wordnet_sample/'
    filenames = os.listdir(i_data_path)
    filenames.sort()
    for f in filenames:
        if f in ['3.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']:
            print('write')
            w = open(o_data_path + f, 'w') # 对于每条测试数据
            all_replace_text = open(i_data_path+f, 'r').readlines()
            for replace_text in all_replace_text:
                replace_text = replace_text.strip()
                text_around = get_text_around(replace_text)
                w.write(replace_text+'\n')  # 对于每个替换样本，首先写入原始替换样本
                for ta in text_around:
                    w.write(ta + '\n') # 写入该替换样本的所有sample样本（sample_num个）
            w.flush()
            w.close()


"""在同义词空间中，随机采样文本，并测试采样文本的对抗性"""
def test_sample(text_index=0):
    # 1. 获得指定索引数据周围随机采样的样本
    # 获得原始文本
    dataset_path = 'data/imdb'
    texts, labels = dataloader.read_corpus(dataset_path)
    data = list(zip(texts, labels))
    ori_text, true_label = data[text_index]
    print(text_index)
    print('Original text:')
    print(' '.join(ori_text))
    print('True label: %d' % true_label)
    # 预测原始文本
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
    ori_probs, sentence_emb = predictor([ori_text]) # sentence_emb: torch.Size([1, 300])，数值在0-1之间
    ori_probs = ori_probs.squeeze()
    ori_label = torch.argmax(ori_probs)
    if ori_label != true_label:
        print('predict false')
        exit()
    # 获得应当采样的次数
    fastca_text = open('adv_results/imdb_fastca_wordnet/' + str(text_index) + '.txt', 'r').readlines()
    sample_num = len(fastca_text)
    sample_num =10000  # 增加采样次数，观察对抗样本个数变化
    # 获得采样样本
    w = open('adv_results/imdb_sample/' + str(text_index) + '.txt', 'w')
    all_sample_text = []
    for n in range(sample_num):
        sample_text = ori_text.copy()
        for i in range(len(sample_text)):
            word = sample_text[i]
            # 获得该word的同义词（包含自己）
            synonyms = wordnet.synsets(word)
            lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
            lemmas.sort()
            if len(lemmas) > 0:
                # 随机采样一个同义词并替换
                sample_word = random.sample(lemmas, 1)[0]
                # sample_word = sample_word.replace('_', ' ') # 同义词可能是以_连接的短语
                sample_text[i] = sample_word
        # 将采样文本写入文件
        w.write(' '.join(sample_text) + '\n')
        all_sample_text.append(sample_text)
    w.close()
    
    # 2. 测试采样样本的对抗性
    label_change_sample = 0
    for text in all_sample_text:
        probs, sentence_embeds = predictor([text]).squeeze() # hhhhh
        label = torch.argmax(probs)
        if label != ori_label:
            label_change_sample += 1
    print('Rates of label changed for all sample text: %f %d' % (float(label_change_sample) / float(sample_num), label_change_sample))
    
    
"""可视化数据"""
def vis(text_index):
    # 0. 加载模型和原始数据
    # 获得原始文本
    dataset_path = 'data/imdb'
    texts, labels = dataloader.read_corpus(dataset_path)
    data = list(zip(texts, labels))
    ori_text, true_label = data[text_index]
    print(text_index)
    print('Original text:')
    print(' '.join(ori_text))
    print('True label: %d' % true_label)
    # 预测原始文本
    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses = args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location = 'cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses = args.nclasses, hidden_size = 100, cnn = True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location = 'cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses = args.nclasses, max_seq_length = args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")
    ori_probs, sentence_emb = predictor([ori_text])  # sentence_emb: torch.Size([1, 300])，数值在0-1之间
    ori_probs = ori_probs.squeeze()
    ori_label = torch.argmax(ori_probs)
    if ori_label != true_label:
        print('predict false')
        exit()
    
    # 1. 获得t路覆盖数组中对抗样本的sentence embds并赋予tsne-label
    ca_path = 'adv_results/imdb_fastca_wordnet/' + str(text_index) + '.txt'
    ca_text = open(ca_path, 'r').readlines()
    ca_sentence_embs = []
    ca_tsne_labels = []
    for text in ca_text:
        text = text.strip().split(' ')
        probs, sentence_emb = predictor([text])
        probs = probs.squeeze()
        label = torch.argmax(probs)
        if label != ori_label:
            # 若为对抗样本，整理成tsne的输入
            ca_sentence_embs.append(sentence_emb.squeeze().cpu().numpy().tolist())
            ca_tsne_labels.append(1) # t路覆盖的对抗样本标记为1
        
    # 2. 获得随机采样中对抗样本的sentence embds并赋予tsne-label
    sa_path= 'adv_results/imdb_sample/' + str(text_index) + '.txt'
    sa_text = open(sa_path, 'r').readlines()
    sa_sentence_embs = []
    sa_tsne_labels = []
    for text in sa_text:
        text = text.strip().split(' ')
        probs, sentence_emb = predictor([text])
        probs = probs.squeeze()
        label = torch.argmax(probs)
        if label != ori_label:
            # 若为对抗样本，整理成tsne的输入
            sa_sentence_embs.append(sentence_emb.squeeze().cpu().numpy().tolist())
            sa_tsne_labels.append(0) # 随机采样的对抗样本标记为0
    
    # 3. 调用tsne并作图
    ca_sentence_embs.extend(sa_sentence_embs)
    sentence_embs = np.array(ca_sentence_embs) # num*300
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    tsne_2D.fit_transform(sentence_embs)
    vectors_2D = tsne_2D.embedding_
    print(vectors_2D.shape)
    x = vectors_2D[:, 0]
    y = vectors_2D[:, 1]
    plt.scatter(x[:int(vectors_2D.shape[0]/2)], y[:int(vectors_2D.shape[0]/2)], c = 'r', label=('CA'))
    plt.scatter(x[int(vectors_2D.shape[0]/2):], y[int(vectors_2D.shape[0]/2):], c = 'y', label=('Sample'))

    plt.legend()
    plt.savefig('vis.pdf')
    
    return

"""可视化替换空间"""
def vis_replace_space(text_index):
    # 0. 加载模型
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
    print("Model built!")

    # 1. 获得t路覆盖数组中对抗样本的replace_vector
    ca_path = 'adv_results/imdb_fastca_wordnet/' + str(text_index) + '.txt'
    ca_text = open(ca_path, 'r').readlines()
    ca_replace_vectors = []
    for text in ca_text:
        text = text.strip().split(' ')
        ca_replace_vec = []
        for word in text:
            synonyms = wordnet.synsets(word)
            lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
            lemmas.sort()
            try:
                replace_lemma_index = lemmas.index(word)
                ca_replace_vec.append(replace_lemma_index)
            except:
                ca_replace_vec.append(-1)
        ca_replace_vectors.append(ca_replace_vec)
    ca_replace_vectors = np.array(ca_replace_vectors)


    # 2. 获得随机采样中对抗样本的sentence embds并赋予tsne-label
    sa_path = 'adv_results/imdb_sample/' + str(text_index) + '.txt'
    sa_text = open(sa_path, 'r').readlines()
    sa_replace_vectors = []
    for text in sa_text:
        text = text.strip().split(' ')
        sa_replace_vec = []
        for word in text:
            synonyms = wordnet.synsets(word)
            lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
            lemmas.sort()
            try:
                replace_lemma_index = lemmas.index(word)
                sa_replace_vec.append(replace_lemma_index)
            except:
                sa_replace_vec.append(-1)
        sa_replace_vectors.append(sa_replace_vec)
    sa_replace_vectors = np.array(sa_replace_vectors)

    # 3. 调用tsne并作图
    replace_vectors = np.vstack((ca_replace_vectors, sa_replace_vectors))
    print(replace_vectors.shape)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    tsne_2D.fit_transform(replace_vectors)
    vectors_2D = tsne_2D.embedding_
    print(vectors_2D.shape)
    x = vectors_2D[:, 0]
    y = vectors_2D[:, 1]
    plt.scatter(x[:ca_replace_vectors.shape[0]], y[:ca_replace_vectors.shape[0]], c='r', label=('CA'))
    plt.scatter(x[ca_replace_vectors.shape[0]:], y[ca_replace_vectors.shape[0]:], c='y', label=('Sample'))

    plt.legend()
    plt.savefig('vis_replace_space.pdf')

    return


"""枚举所有2位置的替换组合，并测试所有样本数据是否能被攻击成功"""
def enumerate_all_2_comb():
    # 1. 读取原始数据
    # get data
    texts, labels = dataloader.read_corpus(args.dataset_path)
    data = list(zip(texts, labels))
    # data = [data[8]]
    data = data[:args.data_size]
    print("Data import finished!")
    # 2. 加载模型
    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses = args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location = 'cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses = args.nclasses, hidden_size = 100, cnn = True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location = 'cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses = args.nclasses, max_seq_length = args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    w = open('adv_results/enumerate_2_comb_10.txt', 'w')
    # 对于每条原始数据
    for idx, (text, true_label) in enumerate(data):
        print(idx)
        # 1. 过滤预测错误数据
        probs, sentence_emb = predictor([text])
        probs = probs.squeeze()
        ori_label = torch.argmax(probs)
        if ori_label != true_label:
            print('%d predict false' % idx)
            print('%d predict false' % idx, file=w)
            continue

        attack_success = 0

        # 2. 枚举2-位置组合结果
        perms = list(combinations(range(len(text)), 2))  # C_len(text)^2
        for ii in tqdm(range(len(perms))):  # 对于每种位置组合
            perm = perms[ii]
            # 对于第一个位置，获得所有同义词
            synonyms_0 = wordnet.synsets(text[perm[0]])
            lemmas_0 = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_0])))
            lemmas_0.sort()
            # 对于第二个位置，所有同义词
            synonyms_1 = wordnet.synsets(text[perm[1]])
            lemmas_1 = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_1])))
            lemmas_1.sort()
            # 若存在位置没有同义词，则抛弃该位置组合
            if len(lemmas_0) == 0 or len(lemmas_1) == 0:
                continue

            # 对所有同义词组合，测试对抗性
            for syn_0 in lemmas_0:
                for syn_1 in lemmas_1:
                    replace_text = text.copy()
                    replace_text[perm[0]] = syn_0
                    replace_text[perm[1]] = syn_1
                    # 测试替换样本对抗性
                    replace_probs, replace_sentence_emb = predictor([replace_text])
                    replace_probs = replace_probs.squeeze()
                    replace_label = torch.argmax(replace_probs)
                    if replace_label != ori_label:  # 对抗成功
                        attack_success = 1
                        print('%d data attack success when replace %d and %d word to:' % (idx, perm[0], perm[1]))
                        print(' '.join(replace_text))
                        w.write('%d data attack success when replace %d and %d word to:\n' % (idx, perm[0], perm[1]))
                        w.write(' '.join(replace_text)+'\n')
                        break
                if attack_success == 1:
                    break
            if attack_success == 1:
                break
        if attack_success != 1:
            w.write('%d data attack failed.\n' % idx)


"""
统计fastCA对抗样本中的平均替换词数
    hownet:30%
    wordnet:50%
"""
def count_replace_num(result_file):
    adv_results = open(result_file).readlines()
    for line in adv_results:
        line = line.strip()
        if '---' in line:
            text_index = int(line.replace('---',''))
        elif len(line.split(' ')) > 100:
            all_adv_index = [i for i, x in enumerate(line.split(' ')) if x == '1']
            data = open('adv_results/imdb_fastca_hownet/' + str(text_index) + '.txt', 'r').readlines()
            ori_text = data[0].split(' ')[1:]
            all_replace_text = data[1:]
            for adv_index in all_adv_index:
                adv_text = all_replace_text[adv_index].split(' ')
                replace_pst = []  # 记录替换的位置
                for i in range(len(adv_text)):
                    if adv_text[i] != ori_text[i]:
                        replace_pst.append(i)
                replace_num = len(replace_pst)  # 记录替换的个数
                print(float(replace_num)/(len(ori_text)))
    exit()
            
    
def main():
    predictor = load_model()
    data = list(zip(load_data()))
    
    """Attack"""
    adv_w = open('adv_results/imdb_hownet_adv_results.txt', 'w')
    # wordnet_lemmatizer = WordNetLemmatizer()
    ori_predict_true = 0
    # label_change_lemm = 0
    change_rates = []  # 记录对于各个测试数据，覆盖数组中改变label的数据比例
    num_adv = 0  # 记录所有替换样本的个数（）
    changed_adv = 0  # 记录所有替换文本中，会改变label的个数
    for idx, (text, true_label) in enumerate(data):
        adv_w.write('------\n')
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
            with open('adv_results/imdb_fastca_hownet/' + str(idx) + '.txt', 'r') as r:
                # changed_w = open('adv_results/imdb_adv_results/' + str(idx) + '.txt', 'w')
                all_replace_text = r.readlines()
                replace_text_num.append(len(all_replace_text))
                for replace_text in all_replace_text:
                    replace_text = replace_text.split(' ')
                    replace_probs = predictor([replace_text]).squeeze()
                    replace_label = torch.argmax(replace_probs)
                    if orig_label != replace_label:
                        label_change_replace += 1
                        changed_adv += 1
                        # print('Label changed at this time.')
                        # changed_w.write('1 ')
                        # adv_w.write('1-' + ' '.join(replace_text)) # 标志当前替换文本会改变label
                        adv_w.write('1 ') # 标志当前替换文本会改变label

                    else:
                        # changed_w.write('0 ')
                        # adv_w.write('0-' + ' '.join(replace_text))
                        adv_w.write('0 ')
                adv_w.write('\n ')
            print('Num of replace text: %d' % len(all_replace_text))
            print('Num of replace text: %d' % len(all_replace_text), file = adv_w)
            change_rate = float(label_change_replace) / float(len(all_replace_text))  # 覆盖数组中，会改变label的所占比例
            print('Rates of label changed for all replace text: %f' % change_rate)
            print('Rates of label changed for all replace text: %f' % change_rate, file = adv_w)
            change_rates.append(change_rate)
            num_adv += len(all_replace_text)
            

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
            adv_w.write('Predict false.\n')

    print('*****************End of attack*****************', file = adv_w)
    print('Origin data size:%d, Origin predict true:%d, Attack success:%d'
          % (args.data_size, ori_predict_true, len(change_rates) - change_rates.count(-1) - change_rates.count(0)))
    print('Origin data size:%d, Origin predict true:%d, Attack success:%d, All replace text num:%d, Label changed in replace text:%d'
          % (args.data_size, ori_predict_true, len(change_rates)-change_rates.count(-1)-change_rates.count(0), num_adv, changed_adv), file = adv_w)
    
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

    # get_all_fastcaInfo()

    example_ca()

    # 先修改run_attack_classification.py
    # get_replace_text_all()
    # attack(attack_data_dir='adv_results/imdb_fastca_hownet/', result_file='adv_results/imdb_fastca_hownet_adv_results.txt')
    # count_replace_num(result_file='adv_results/imdb_fastca_hownet_adv_results.txt')
    # get_text_around_all()
    # main()
    
    # 处理第text_index条数据
    # text_index = 0
    # test_sample(text_index)
    # vis(text_index)
    # vis_replace_space(text_index)

    # enumerate_all_2_comb()

    # 先修改run_attack_classification.py
    # attack_bdlstm(attack_data_dir='adv_results/imdb_fastca_hownet/', result_file='adv_results/imdb_fastca_hownet_bdlstm_adv_results.txt')
