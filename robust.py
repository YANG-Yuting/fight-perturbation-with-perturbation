from __future__ import print_function
from random import choice
import argparse
import torch
import pickle
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

import dataloader
from train_classifier import Model, eval_model
from attack_classification import NLI_infer_BERT, eval_bert
# from attack_nli import NLI_infer_BERT as NLI_infer_BERT_nli
import time
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, RandomSampler
from gen_pos_tag import pos_tagger
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='eval', help='train, eval')  # tiz,
parser.add_argument("--dataset", type=str, default="mr_adv", help="which dataset") # when ends with "_adv" means loading adversarial examples.

parser.add_argument("--robust_file",
                    type=str,
                    default='')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--sample_num",
                    type=int,
                    default=5000)
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
                    default='wordLSTM',
                    help="Target models for text classification: fasttext, charcnn, word level lstm "
                         "For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--target_model_path",
                    type=str,
                    default='models/wordLSTM/mr',
                    help="pre-trained target model path")
parser.add_argument("--word_embeddings_path",
                    type=str,
                    default='glove.6B/glove.6B.200d.txt',
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
                    default=3000,
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
args = parser.parse_args()

args.dataset_path = 'data/adversary_training_corpora/' + args.task
seq_len_list = {'imdb': 256, 'mr': 128, 'fake': 512}
args.max_seq_length = seq_len_list[args.task]
sizes = {'imdb': 50000, 'mr': 20000, 'fake': 50000}
max_vocab_size = sizes[args.task]
with open('data/adversary_training_corpora/%s/dataset_%d.pkl' % (args.task, max_vocab_size), 'rb') as f:
    dataset = pickle.load(f)
inv_full_dict = dataset.inv_full_dict
full_dict = dataset.full_dict



def load_model():
    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.max_seq_length, args.word_embeddings_path, nclasses=args.nclasses).cuda()
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

def load_data():
    sizes = {'imdb': 50000, 'mr': 20000}
    max_vocab_size = sizes[args.task]

    with open(args.dataset_path + '/word_candidates_sense_top5.pkl', 'rb') as fp:
        word_candidate = pickle.load(fp)
    with open(args.dataset_path + '/pos_tags_test.pkl', 'rb') as fp:
        pos_tags_test = pickle.load(fp)  # 保存所有文本所有单词的词性
    with open(args.dataset_path + '/pos_tags.pkl', 'rb') as fp:
        pos_tags = pickle.load(fp)  # 保存所有文本所有单词的词性
    pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    f = open(args.dataset_path + '/dataset_%d.pkl' % max_vocab_size, 'rb')
    dataset = pickle.load(f)
    inv_full_dict = dataset.inv_full_dict
    full_dict = dataset.full_dict

    # print(full_dict['\x85'])  # imdb:5196

    # ca = word_candidate[full_dict['film']]
    # print(ca)
    # print([inv_full_dict[i] for i in ca['noun']])
    # print([inv_full_dict[i] for i in ca['verb']])
    # exit(0)

    print('Data load!')

    return dataset, inv_full_dict, full_dict, word_candidate, pos_tags_test, pos_tags, pos_list


# 获得候选集（对于给定词语，获得它5个同义词列表的同义词的交集，均包含自己）
def get_candidates(word, pos, word_candidate, tf_vocabulary, text_words, position, if_bound): # word是词语id,text_words是字符串
    syns_l1 = word_candidate[word][pos] + [word] # 包含自己；一级同义词
    # 1.直接获得原位置的一级同义词，返回
    # syns_inter = syns_l1
    # 1. 对所有同义词的同义词获取交集 acc:mr-lstm, 69.6%
    # for syn in syns_l1:
    #     syns_l2 =  word_candidate[syn][pos] + [syn, word]# 默认同义词和原词一定有一致的词性；将原词也加上（大多具有对称性，有的因不在top5而被过滤了）
    #     syns_inter = list(set(syns_inter).intersection(set(syns_l2)))
    # 2. 从同义词里随机选一个/第一个，获得其同义词，再得交集 69.6%
    # syns_l2 = word_candidate[syns_l1[0]][pos] + [syns_l1[0], word]  # 默认同义词和原词一定有一致的词性；将原词也加上（大多具有对称性，有的因不在top5而被过滤了）
    # syns_inter = list(set(syns_inter).intersection(set(syns_l2)))
    # 3. 从同义词中获得词频最高的那个，获得其同义词，再得交集
    # syn_max, ti_max = word, 0 # 默认词频最大的同义词是自己
    tf_1gram_all, tf_2gram_former_all,tf_2gram_latter_all, tf_2gram_all  = {},{},{},{}
    for syn in syns_l1:
        tf_1gram, tf_2gram_former, tf_2gram_latter = 0,0,0
        # 1-gram
        if inv_full_dict[syn] in tf_vocabulary.keys():
            tf_1gram = tf_vocabulary[inv_full_dict[syn]]
        tf_1gram_all[syn] = tf_1gram

        # 2-gram
        if if_bound == 'not_bound':
            # 左2-gram
            two_gram_former = text_words[position-1] + ' ' + inv_full_dict[syn]  # 获得文本中该位置前面的词语和该位置同义词的拼接（2-gram）
            if two_gram_former in tf_vocabulary.keys():
                tf_2gram_former = tf_vocabulary[two_gram_former]
            tf_2gram_former_all[two_gram_former] = tf_2gram_former
            tf_2gram_all[two_gram_former] = tf_2gram_former
            # 右2-gram
            two_gram_latter = inv_full_dict[syn] + ' ' + text_words[position + 1]
            if two_gram_latter in tf_vocabulary.keys():
                tf_2gram_latter = tf_vocabulary[two_gram_latter]
            tf_2gram_latter_all[two_gram_latter] = tf_2gram_latter
            tf_2gram_all[two_gram_latter] = tf_2gram_latter
        elif if_bound == 'left_bound':
            two_gram_latter = inv_full_dict[syn] + ' ' + text_words[position + 1]
            if two_gram_latter in tf_vocabulary.keys():
                tf_2gram_latter = tf_vocabulary[two_gram_latter]
            tf_2gram_latter_all[two_gram_latter] = tf_2gram_latter
            tf_2gram_all[two_gram_latter] = tf_2gram_latter
        elif if_bound == 'right_bound':
            two_gram_former = text_words[position - 1] + ' ' + inv_full_dict[syn]  # 获得文本中该位置前面的词语和该位置同义词的拼接（2-gram）
            if two_gram_former in tf_vocabulary.keys():
                tf_2gram_former = tf_vocabulary[two_gram_former]
            tf_2gram_former_all[two_gram_former] = tf_2gram_former
            tf_2gram_all[two_gram_former] = tf_2gram_former

    if tf_2gram_all.values() == 0: # 若所有2-gram都未在训练集中出现过，则在1-gram中寻找最佳
        syn_best_1gram = max(tf_1gram_all, key=tf_1gram_all.get)  # id
        syn_best = syn_best_1gram
    else:
        if if_bound == 'left_bound':
            syn_best_2gram = max(tf_2gram_latter_all, key=tf_2gram_latter_all.get).split(' ')[0]
        elif if_bound == 'right_bound':
            syn_best_2gram = max(tf_2gram_former_all, key=tf_2gram_former_all.get).split(' ')[1]
        elif if_bound == 'not_bound':
            if max(tf_2gram_latter_all) > max(tf_2gram_former_all):
                syn_best_2gram = max(tf_2gram_latter_all, key=tf_2gram_latter_all.get).split(' ')[0]
            else:
                syn_best_2gram = max(tf_2gram_former_all, key=tf_2gram_former_all.get).split(' ')[1]
        syn_best_2gram = full_dict[syn_best_2gram]  # str转id
        syn_best = syn_best_2gram

    # 就用1-gram
    # syn_best_1gram = max(tf_1gram_all, key=tf_1gram_all.get)  # id
    # syn_best = syn_best_1gram

    syns_l2 = word_candidate[syn_best][pos] + [syn_best] # 默认同义词和原词一定有一致的词性；加上原词word，adv acc:68.8，不加，73.1
    syns_inter = syns_l2 # 不求交集了，直接返回最高词频同义词的同义词

    # syns_inter = list(set(syns_inter).intersection(set(syns_l2)))


    return syns_inter

"""为一条文本(idx!=-1，来自数据集；idx=-1，新生成文本)采样"""
# text是word id
# def gen_sample_aText(idx,text, train, pos_tags_test, pos_tags, word_candidate, pos_list,sample_num):  # text一条文本，list
def gen_sample_aText(text, word_candidate, pos_list,sample_num):  # text一条文本，list
    # 获得同义词空间随机采样样本

    if args.task == 'mr':
        text = [x for x in text if x != 50000]  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
    elif args.task == 'imdb':
        text = [x for x in text if x != 5169]  # 注：文本里有5169(\x85)，而词性标注结果没有，导致无法对齐。将其删除

    # 加载词频信息
    # file = open('data/adversary_training_corpora/mr/tfidf.txt', 'r')
    # js = file.read()
    # dic = json.loads(js)
    # file.close()

    # 加载词频信息
    tf_vocabulary = pickle.load(open('data/adversary_training_corpora/mr/tf_vocabulary.pkl', "rb"))

    # if idx == -1:
    #     pos_tag = pos_tagger.tag(text)  # 新文本，现词性标注
    # else:
    #     if train:
    #         pos_tag = pos_tags[idx]
    #     else:
    #         pos_tag = pos_tags_test[idx]
    full_dict['<oov>'] = len(full_dict.keys())
    inv_full_dict[len(full_dict.keys())] = '<oov>'
    text_words = [inv_full_dict[id] for id in text]  # id转word
    pos_tag = pos_tagger.tag(text_words)  # 改为直接做词性标注，分词需要词语
    sample_texts = []

    # 按照对f分类的重要性，对句子中的词语排序（参考TF攻击attack_classification_hownet_top5.py中的做法，get importance）

    # for s in range(args.sample_num):
    for s in range(sample_num): # tiz: 为方便外部调用，加入sample_num参数
        sample_text = text.copy()
        for j in range(len(sample_text)):  # 对于每个词语
            word = sample_text[j]
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
            if j == len(sample_text) -1:
                if_bound = 'right_bound'
            neigbhours = get_candidates(word, pos, word_candidate, tf_vocabulary, text_words, j, if_bound)
            try:
                sample_text[j] = choice(neigbhours)  # 候选集中随机选择一个
            except:
                print(word, pos, neigbhours)
                syns_l1 = word_candidate[word][pos] + [word]  # 包含自己
                print(syns_l1)
                for syn in syns_l1:
                    syns_l2 = word_candidate[syn][pos] + [syn]  # 默认同义词和原词一定有一致的词性
                    print(syns_l2)
                exit(0)

            # print(inv_full_dict[word], [inv_full_dict[n] for n in neigbhours])
        # print([inv_full_dict[t] for t in sample_text])
        sample_texts.append(sample_text)
    return sample_texts


class myDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


"""计算一条文本的鲁棒性"""
def robust_aText(idx, text, true_label, model, train, inv_full_dict, pos_tags_test, pos_tags, word_candidate, pos_list):
    # 1. 获得采样数据
    sample_texts = gen_sample_aText(idx, text, train, pos_tags_test, pos_tags, word_candidate, pos_list, inv_full_dict)
    # 2. 测试采样数据鲁棒性
    # 判断原始文本是否预测正确
    text1 = [inv_full_dict[id] for id in text]  # 转为str（网络输入需要词语）
    # ori_probs = model.module.text_pred([text1])
    ori_probs = model.text_pred([text1])
    # ori_probs = model([text1])
    ori_probs = ori_probs.squeeze()
    ori_label = torch.argmax(ori_probs).item()
    # 预测采样文本
    sample_texts = [[inv_full_dict[id] for id in t] for t in sample_texts]  # 转为str
    sample_labels = [true_label] * len(sample_texts)


    if args.target_model == 'bert':
        robust_score = eval_bert(model, sample_texts, sample_labels)
    else:
        # 注意：这一步会转化成网络对应的id，利用model.word2id。因此与数据集加载的id不一致。且维度转置了，变成了(len(text), sample_num)
        sample_texts, sample_labels = dataloader.create_batches(sample_texts, sample_labels, args.max_seq_length, args.batch_size, model.word2id, )
        robust_score = eval_model(model, sample_texts, sample_labels)

    predict = (ori_label == true_label)  # 是否正确预测
    return predict, robust_score


def robust():
    # 2.加载辅助数据
    dataset, inv_full_dict, full_dict, word_candidate, pos_tags_test, pos_tags, pos_list = load_data()


    # 1.加载模型
    model = load_model()

    # 2. 加载数据集
    # 测试集
    test_x = dataset.test_seqs2
    test_y = dataset.test_y
    # test_y = [[0, 1] if t == 1 else [1, 0] for t in test_y]
    # 取前25%
    # test_size = len(test_y)
    # test_x = test_x[:int(test_size*0.25)]
    # test_y = test_y[:int(test_size*0.25)]
    # test_x = test_x[:1000]
    # test_y = test_y[:1000]
    # 训练集
    train_x = dataset.train_seqs2
    train_y = dataset.train_y
    # train_y = [[0, 1] if t == 1 else [1, 0] for t in train_y]
    # 取前25%
    # train_size = len(train_y)
    # train_x = train_x[:int(train_size*0.25)]
    # train_y = train_y[:int(train_size*0.25)]

    # 3.测试集遍历数据获得鲁棒性
    if args.train:
        # 训练集遍历数据获得鲁棒性
        print('For train set...')
        # if 'adv' in args.target_model_path:
        #     print('adv model')
        #     robust_file = args.dataset_path + '/robust_samp%d_train%d_%s_adv.txt' % (args.sample_num, percent, args.target_model)
        # else:
        #     robust_file = args.dataset_path + '/robust_samp%d_train%d_%s.txt' % (args.sample_num, percent, args.target_model)
        train = True
        f = open(args.robust_file, 'w')
        for idx in tqdm(range(len(train_y))):

            time_start = time.time()
            text = train_x[idx]
            true_label = train_y[idx]
            predict, robust_value = robust_aText(idx, text, true_label, model, train, inv_full_dict,
                                                 pos_tags_test, pos_tags, word_candidate, pos_list)
            time_end = time.time()
            # print('Time:' , time_end - time_start)
            # print('%d %d %f' % (idx, predict, robust_value))
            print('%d %d %f' % (idx, predict, robust_value), file=f)
            f.flush()
    else:
        print('For test set...')
        # if 'adv' in args.target_model_path:
        #     print('adv model')
        #     if 'nonrobust' in args.target_model_path:
        #         robust_file = args.dataset_path + '/robust_samp%d_test%d_%s_adv_nonrobust.txt' % (args.sample_num, percent, args.target_model)
        #     else:
        #         robust_file = args.dataset_path + '/robust_samp%d_test%d_%s_adv.txt' % (args.sample_num, percent, args.target_model)
        # else:
        #     robust_file = args.dataset_path + '/robust_samp%d_test%d_%s.txt' % (args.sample_num, percent, args.target_model)

        train = False
        f = open(args.robust_file, 'w')
        for idx in tqdm(range(len(test_y))):
            time_start = time.time()
            text = test_x[idx]
            true_label = test_y[idx]
            predict, robust_value = robust_aText(idx, text, true_label, model, train, inv_full_dict,
                                                 pos_tags_test, pos_tags, word_candidate, pos_list)
            time_end =time.time()
            # print('Time:', time_end - time_start)
            # print('%d %d %f' % (idx, predict, robust_value))
            print('%d %d %f' % (idx, predict, robust_value), file=f)
            f.flush()

def ana_robust():
    # train_robust_file = args.dataset_path + '/robust_samp%d_train100_%s.txt' % (args.sample_num, args.target_model)
    # # test_robust_file = args.dataset_path + '/robust_samp%d_test25_%s.txt' % (args.sample_num, args.target_model)
    # train_robust = open(train_robust_file, 'r').read().splitlines()
    # test_robust = open(test_robust_file, 'r').read().splitlines()

    # train_robust = np.array([float(r.split(' ')[2]) for r in train_robust])
    # pers = np.percentile(train_robust, (25, 50, 75, 100), interpolation='midpoint')
    # print(pers)
    # train_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in train_robust]
    # lt1 = sum(train_robust)
    # print('%s(train): %d/%d(%f) data has robust value less than 1 for %s' %
    #       (args.task, lt1, len(train_robust), float(lt1)/float(len(train_robust)),  args.target_model))

    robust_data = open(args.robust_file, 'r').read().splitlines()
    robust_values = []
    for rv in robust_data:
        predict = int(rv.split(' ')[1])
        # if predict == 1:
        if predict == 1 or predict == 0:
            robust_value = float(rv.split(' ')[2])
            robust_values.append(robust_value)
    print(len(robust_values))
    robust_values = np.array(robust_values)
    pers = np.percentile(robust_values, (25, 50, 75, 100), interpolation='midpoint')
    print(pers)
    print('mean:', np.mean(robust_values))
    lt09 = np.where(robust_values < 0.9)[0]
    print('%s(test): %d/%d(%f) data has robust value less than 0.9 for %s' %
          (args.task, lt09.shape[0], robust_values.shape[0], float(lt09.shape[0])/float(robust_values.shape[0]), args.target_model))
    lt095 = np.where(robust_values < 0.95)[0]
    print('%s(test): %d/%d(%f) data has robust value less than 0.95 for %s' %
          (args.task, lt095.shape[0], robust_values.shape[0], float(lt095.shape[0]) / float(robust_values.shape[0]),
           args.target_model))
    lt097 = np.where(robust_values < 0.97)[0]
    print('%s(test): %d/%d(%f) data has robust value less than 0.97 for %s' %
          (args.task, lt097.shape[0], robust_values.shape[0], float(lt097.shape[0]) / float(robust_values.shape[0]),
           args.target_model))
    lt099 = np.where(robust_values < 0.99)[0]
    print('%s(test): %d/%d(%f) data has robust value less than 0.99 for %s' %
          (args.task, lt099.shape[0], robust_values.shape[0], float(lt099.shape[0]) / float(robust_values.shape[0]),
           args.target_model))
    lt0995 = np.where(robust_values < 0.995)[0]
    print('%s(test): %d/%d(%f) data has robust value less than 0.995 for %s' %
          (args.task, lt0995.shape[0], robust_values.shape[0], float(lt0995.shape[0]) / float(robust_values.shape[0]),
           args.target_model))
    lt1 = np.where(robust_values < 1)[0]
    print('%s(test): %d/%d(%f) data has robust value less than 1 for %s' %
          (args.task, lt1.shape[0], robust_values.shape[0], float(lt1.shape[0])/float(robust_values.shape[0]), args.target_model))
    eq1 = np.where(robust_values == 1)[0]
    print('%s(test): %d/%d(%f) data has robust value 1 for %s' %
          (args.task, eq1.shape[0], robust_values.shape[0], float(eq1.shape[0])/float(robust_values.shape[0]), args.target_model))

"""寻找鲁棒的点，获得其覆盖数组个数及对抗样本个数，再计算需要的随机采样词数"""
def ca_vs_sample():
    # 1.加载模型
    model_path = 'data/Imdb/bdlstm_models'
    model = bd_lstm(embedding_matrix)
    print('Loading model...')
    model.load_weights(model_path)
    # 2. 加载原始数据
    print('Loading data...')
    train_x = dataset.train_seqs2
    train_y = dataset.train_y
    train_y = [[0, 1] if t == 1 else [1, 0] for t in train_y]

    train_robust_file = 'adv_results/imdb_bdlstm_train_robust.txt'  # 训练集的鲁棒性
    train_robust = open(train_robust_file, 'r').read().splitlines()
    train_robust = [float(r.split(' ')[1]) for r in train_robust]
    adv_result_file = 'adv_results/imdb_fastca_hownet_bdlstm_adv_results.txt'  # 训练集覆盖数组的攻击结果
    adv_result = open(adv_result_file, 'r').read().splitlines()
    adv_result = [a.split(' ')[1:] for a in adv_result]
    w_file = open('adv_results/ca_vs_sample.txt' , 'w')
    ratio = []
    for i in range(1000):  # 对于每条数据
        robust_value = train_robust[i]
        if robust_value == -1: # 若原始预测错误
            print('%d -1' % i)
            print('%d -1' % i, file=w_file)
            continue
        adv_num = int(adv_result[i][1])  # 对抗样本个数
        if adv_num == 0:  # 若CA找不到对抗样本
            print('%d -2' % i)
            print('%d -2' % i, file=w_file)
            continue
        if robust_value < 1:  # 若该点鲁棒性小于1
            print('%d -3' % i)
            print('%d -3' % i, file=w_file)
            continue
        # 若该数据点的鲁棒性为1且CA能找到对抗样本
        ca_num = int(adv_result[i][0])  # CA个数
        ori_text = train_x[i]
        true_label = train_y[i]
        base_sample_num = 1000
        sample_num = base_sample_num
        # 第一次采样，1000条
        sample_texts = gen_sample_aText(i, ori_text, sample_num, train=1)
        sample_texts = pad_sequences(sample_texts, maxlen=250, padding='post')
        sample_labels = np.array([true_label] * len(sample_texts))
        sample_adv_num = (1 - model.evaluate(sample_texts, sample_labels, verbose=0)[1]) * len(sample_texts)  # 保存采样样本中的对抗样本个数
        k = 0  # while循环的上界，50，即采样50000个样本
        while k < 100 and float(sample_adv_num)/float(adv_num) < 0.9:  # 若采样的对抗样本少于CA得到对抗数组的0.9
            # 又采样1000次
            add_sample_texts = gen_sample_aText(i, ori_text, base_sample_num, train=1)
            add_sample_texts = pad_sequences(add_sample_texts, maxlen=250, padding='post')
            add_sample_labels = np.array([true_label] * len(add_sample_texts))
            add_sample_adv_num = (1 - model.evaluate(add_sample_texts, add_sample_labels, verbose=0)[1]) * len(add_sample_texts)
            sample_num += add_sample_adv_num
            sample_adv_num += add_sample_adv_num
            k += 1
        if k == 100:  # 采样了50000个样本都没有找到足够的对抗样本
            print('%d -4 %d %d ' % (i, sample_adv_num, adv_num))
            print('%d -4 %d %d ' % (i, sample_adv_num, adv_num), file=w_file)
            continue
        print('%d %d %d %d %d %f' % (i, sample_num, sample_adv_num, ca_num, adv_num, float(sample_num)/float(ca_num)))
        print('%d %d %d %d %d %f' % (i, sample_num, sample_adv_num, ca_num, adv_num, float(sample_num)/float(ca_num)), file=w_file)
        w_file.flush()
        ratio.append(float(sample_num)/float(ca_num))
    print('Average ratio: %f' % np.array(ratio).mean())
    w_file.close()



if __name__ == '__main__':
    robust()
    ana_robust()

