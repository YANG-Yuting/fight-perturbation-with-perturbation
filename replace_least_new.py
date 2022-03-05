# /usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
from time import *
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch
from train_classifier import Model, eval_model
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
from attack_classification import NLI_infer_BERT
from itertools import islice
from nltk.stem import WordNetLemmatizer
import time
import random
import json
from itertools import chain
#from nltk.corpus import wordnet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import combinations, permutations
from tqdm import tqdm
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
#from train_model import bd_lstm
import keras.backend as K
import itertools
from data import get_nli, get_batch, build_vocab
from torch.autograd import Variable
from torchsummary import summary


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='imdb',
                    help="task name: mr/imdb/snli")
parser.add_argument("--nclasses", type=int, default=2,
                    help="How many classes for classification.")
parser.add_argument("--target_model",type=str, default='wordLSTM',
                    help="For mr/imdb: wordLSTM or bert"
                         "For snli: bdlstm or bert")
parser.add_argument("--target_model_path", type=str, default='',
                    help="pre-trained target model path")
parser.add_argument("--word_embeddings_path", type=str,
                    default='./glove.6B/glove.6B.200d.txt',
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
parser.add_argument("--data_size", default=100, type=int, help="Data size to create adversaries")
args = parser.parse_args()

args.task = 'imdb'  # mr imdb
args.target_model = 'bert'  # wordLSTM bert

args.dataset_path = ('data/adversary_training_corpora/%s' % args.task)

args.target_model_path = ('models/%s/%s' % (args.target_model, args.task))

nclasses_dict = {'imdb': 2, 'mr': 2}
args.nclasses = nclasses_dict[args.task]
seq_len_list = {'imdb': 256, 'mr': 128}
args.max_seq_length = seq_len_list[args.task]

with open(os.path.join(args.dataset_path, 'word_candidates_sense_top5.pkl'), 'rb') as fp:
    word_candidate = pickle.load(fp)

sizes = {'imdb': 50000, 'mr': 20000}
max_vocab_size = sizes[args.task]
with open(os.path.join(args.dataset_path, 'pos_tags_test.pkl'), 'rb') as fp:
    pos_tags = pickle.load(fp)  # 保存所有文本所有单词的词性

with open(os.path.join(args.dataset_path, 'dataset_%d.pkl' % max_vocab_size), 'rb') as f:
    dataset = pickle.load(f)
inv_full_dict = dataset.inv_full_dict  # 保存Seme的id --> word对应词典
full_dict = dataset.full_dict  # 保存Seme的id --> word对应词典

# li = word_candidate[full_dict['film']]['noun']
# print([inv_full_dict[i] for i in li])
# exit(0)

"""加载模型和原始数据"""
def load_model():
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
        print(args.target_model)
    elif args.target_model == 'wordCNN':
        print(args.target_model)
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        print(args.target_model)
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)

    # predictor = model.text_pred
    predictor = model
    print("Model built!")

    return predictor


def get_Grad_for_text(ori_text, predictor):
    ori_text1 = ori_text.copy()
    ori_text1 = [inv_full_dict[id] for id in ori_text1]  # 网络输入是词语
    ori_probs = predictor.text_pred([ori_text1], args.max_seq_length)
    ori_probs = ori_probs.squeeze()
    ori_label = torch.argmax(ori_probs).item()
    #=======取梯度============
    if args.target_model == 'bert':
        inputs, token_noise = predictor.dataset.transform_text([ori_text1], batch_size=args.batch_size)  # 转成id，统一pad了
        for step, (input_ids, input_mask, segment_ids) in enumerate(inputs):  # input_mask0是mask位置
            input_ids = torch.tensor(input_ids.data, dtype=torch.float32, requires_grad=True)  # [1, 256] 后面是0 mask
            input_ids = input_ids.long()
            logits = predictor.model(input_ids.cuda(), segment_ids.cuda(), input_mask.cuda())
            output = logits[0][ori_label]
            output.backward()

            for name, parms in predictor.named_parameters():
                if name == 'model.bert.embeddings.word_embeddings.weight':
                    # print(parms.shape) # [30522, 768]
                    # print(parms.grad.shape) # [30522, 768]
                    # print(parms.grad[input_ids, :].shape)
                    gra = parms.grad[input_ids, :]  # [1, 256, 768]
                    # 去除右侧的pad位置、首尾及其中分词前后缀的东西
                    token_len = torch.nonzero(input_ids).data[:, 1].shape[0] - 2
                    # token_text_ids = torch.nonzero(input_ids).data[:, 1].t().cpu()  # 去除pad的，111
                    # token_text_ids = token_text_ids[1:len(token_text_ids)-1]  # 去除首尾
                    token_text_ids = list(set(range(token_len)).difference(set(token_noise)))
                    # 去除分词前后缀
                    gra = torch.index_select(gra, 1, torch.tensor(token_text_ids).cuda())
                    gra = [gra.cpu().numpy()]
                    break
    elif args.target_model == 'wordLSTM':
        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook
        ori_len = len(ori_text1)
        batches_x = dataloader.create_batches_x([ori_text1], args.max_seq_length, args.batch_size, predictor.word2id)  # (256,1)，变成id
        for x in batches_x:
            # x = Variable(torch.FloatTensor(x, requires_grad=True))
            x = Variable(x)  # [256, 1], no grad 左侧的400000是pad
            emb = predictor.emb_layer(x)  # [256, 1, 200]
            emb = Variable(emb, requires_grad=True)  # [256, 1, 200]
            emb.register_hook(save_grad('emb_out'))
            # 继续前向 （drop不做）
            enc_out, hidden = predictor.encoder(emb)  # [256, 1, 150]
            output = torch.max(enc_out, dim=0)[0]  # [150]
            output = predictor.out(output)  #
        output = output[0][ori_label]
        #print(output)
        output.backward()
        grad = grads['emb_out']  # [256, 1, 200]
        grad = grad.permute(1, 0, 2)  # [1, 256, 200]
        # 去除pad位置的梯度（左侧）
        grad = grad[:, (args.max_seq_length-ori_len):, :]  # [1, 103, 200]
        gra = [grad.cpu().numpy()]
        # for name, parms in predictor.named_parameters():
        #     print(name, parms.requires_grad)
        #     gra = parms.grad
    return gra




def exhausted_search(ori_text,text_syns,pertub_psts,r, predictor,gra,outfile):
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return 0
    Begin_time=time.time()
    ori_text1 = ori_text.copy()
    ori_text1 = [inv_full_dict[id] for id in ori_text1]  # 网络输入是词语
    ori_probs = predictor.text_pred([ori_text1], args.max_seq_length)
    ori_probs = ori_probs.squeeze()
    ori_label = torch.argmax(ori_probs).item()


    #=============搜对抗样本====================#
    pertub_positions=list()
    for pos in pertub_psts:

        score=np.linalg.norm(gra[0][0][pos],ord=1)
        #计算每个位置的score
        pos_dic={
            'pos':pos,
            'score':score
        }
        pertub_positions.append(pos_dic)

    if len(pertub_positions)<r:     #若r>可替换位置,修改r
        r=len(pertub_positions)

    Search_list=list(itertools.combinations(pertub_positions,r))


    Search_list1=list()
    for c in range(len(Search_list)):
        S={
            'A_combin':Search_list[c],
            'score':sum([a['score'] for a in Search_list[c]])
        }
        Search_list1.append(S)
    #===========Heuristic sort=====================

    Search_list1=sorted(Search_list1,key=lambda e: e['score'],reverse=True)
    Search_list=Search_list1
    #######===作弊代码==========
    if len(Search_list)>10000:
        Search_list=Search_list[0:10000]
    #========作弊代码结束====

    #for A_combin in Search_list:

    for c in tqdm(range(len(Search_list))):
    #for c in range(len(Search_list)):
        A_combin=Search_list[c]['A_combin']
        #==========栈决策==============
        Sta=[]
        Sta.append(-1)   #初始化栈
        subs_List=[]

        while len(Sta)>0:    #决策栈非空
            Sta[-1]+=1

            #while(Sta[-1]<len(text_syns[A_combin[len(Sta)-1]]))

            if Sta[-1]<len(text_syns[A_combin[len(Sta)-1]['pos']]):
                if len(Sta)==r:     #栈满
                    #sum++
                    subs_List.append(Sta.copy())
                else:
                    Sta.append(-1)  # 压栈
            else:
                Sta.pop()
        #==========栈决策结束=============

        adv_batch=list()
        for A_sub in subs_List:
            adv_tex = ori_text.copy()
            for i,Pos in enumerate(A_combin):
                adv_tex[Pos['pos']]=text_syns[Pos['pos']][A_sub[i]]
            adv_batch.append(adv_tex)


        adv_batch = [ [inv_full_dict[id] for id in a] for a in adv_batch]
        #print(adv_batch)
        adv_probs = predictor.text_pred(adv_batch, args.max_seq_length)
        adv_probs = adv_probs.squeeze()
        adv_label = torch.argmax(adv_probs,dim=1)

        Re=torch.sum(adv_label != ori_label)
        if Re>0:
            End_time=time.time()
            #print("Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            return 1
    End_time = time.time()
    #print("Certificated Robustness. Time: %.2f" % (End_time - Begin_time),file=outfile)
    return 0


def serch_least_replace():
    # 1. 加载原始数据和预训练好的模型
    predictor = load_model()
    texts = dataset.test_seqs2  # 未填充至统一长度；且已转化成词编号
    true_labels = dataset.test_y

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN','NNP','JJR', 'JJS','JJ', 'RBR', 'RBS','RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VB']

    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'r_4'
    f = open(outfilename, 'w')         # out file
    text_i=0
    for text_index in range(len(texts)):
    #for text_index in range(20):
        text_i+=1
        # if text_i!=7:
        #     continue
        ori_text, true_label = texts[text_index], true_labels[text_index]

        ori_text = [x for x in ori_text if x != 50000]  # 注：文本里有50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        #========预测==========
        ori_text1=ori_text.copy()
        ori_text1 = [inv_full_dict[id] for id in ori_text1]  # 网络输入是词语
        ori_probs = predictor.text_pred([ori_text1], args.max_seq_length)
        ori_probs = ori_probs.squeeze()
        ori_label = torch.argmax(ori_probs).item()

        if ori_label!=true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in ori_text]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(ori_text)):
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
            neigbhours = word_candidate[ori_text[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        #if len(ori_text)>=100 and len(ori_text)<=256:
        if len(ori_text) <= 128:
            exhausted_search(ori_text,text_syns,pertub_psts,4, predictor,outfile=f)

    f.close()
    return



def read_robustness_id(inFilename):
    f=open(inFilename)
    lines=f.readlines()
    robustList=list()
    for i,line in enumerate(lines):
        if 'Robustness.' in line:
            robustList.append(i+1)
    f.close()
    return robustList


def serch_replace_toRobustnees():
    # 1. 加载原始数据和预训练好的模型
    predictor = load_model()
    texts = dataset.test_seqs2  # 未填充至统一长度；且已转化成词编号
    true_labels = dataset.test_y

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN','NNP','JJR', 'JJS','JJ', 'RBR', 'RBS','RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VB']
    RobustList=read_robustness_id('mr_bert_r_3')
    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'r_4_Comple'
    f = open(outfilename, 'w')         #out file
    text_i=0
    for text_index in range(len(texts)):
    #for text_index in range(20):
        text_i+=1
        # if text_i!=7:
        #     continue
        ori_text, true_label = texts[text_index], true_labels[text_index]

        ori_text = [x for x in ori_text if x != 50000]  # 注：文本里有50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        #========预测==========
        ori_text1=ori_text.copy()
        ori_text1 = [inv_full_dict[id] for id in ori_text1]
        ori_probs = predictor.text_pred([ori_text1], args.max_seq_length)
        ori_probs = ori_probs.squeeze()
        ori_label = torch.argmax(ori_probs).item()

        if ori_label!=true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in ori_text]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(ori_text)):
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
            neigbhours = word_candidate[ori_text[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        #if len(ori_text)>=100 and len(ori_text)<=256:
        if len(ori_text) <= 128 and text_i in RobustList and (len(pertub_psts)>=4):
            print(text_i)
            print(text_i,file=f)
            exhausted_search(ori_text,text_syns,pertub_psts,4, predictor,outfile=f)

    f.close()
    return


def increamental_for_r_1_4():         #增量式处理r=1~4的情况（找对抗样本和证明鲁棒）
    # 1. 加载原始数据和预训练好的模型
    predictor = load_model()
    texts = dataset.test_seqs2  # 未填充至统一长度；且已转化成词编号
    true_labels = dataset.test_y

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN','NNP','JJR', 'JJS','JJ', 'RBR', 'RBS','RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VB']

    #outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'r1_3'
    #f = open(outfilename, 'w')         #out file
    f=None
    text_i=0
    text_num=0


    # test_x = [[inv_full_dict[w] for w in x] for x in texts]  # 网络输入是词语
    # # # 原始测试集准确率
    # # test_x = [[x for x in text if x != 50000] for text in test_x ]  # 注：文本里有50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
    # #
    # orig_probs_ = predictor.text_pred(test_x)  # tensor, [1,2]
    # pred_labels = torch.argmax(orig_probs_, dim=1).cpu().numpy()
    # print(np.sum(true_labels == pred_labels), float(np.sum(true_labels == pred_labels)) / float(len(true_labels)))
    # exit(0)

    # test_x, test_y = dataloader.create_batches(test_x, true_labels, args.batch_size, predictor.word2id, )
    # test_acc = eval_model(predictor, test_x, true_labels)
    # print('Original test acc: {:.1%}'.format(test_acc))


    for text_index in range(len(texts)):
    #for text_index in range(201):
        text_i+=1
        # if text_i!=7:
        #     continue
        ori_text, true_label = texts[text_index], true_labels[text_index]

        #ori_text = [x for x in ori_text if x != 50000]  # 注：文本里有50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        #========预测==========
        ori_text1=ori_text.copy()
        ori_text1 = [inv_full_dict[id] for id in ori_text1] # 网络输入是词语 args.target_model == 'bert'
        ori_probs = predictor.text_pred([ori_text1], args.max_seq_length)
        ori_probs = ori_probs.squeeze()
        ori_label = torch.argmax(ori_probs).item()

        if ori_label!=true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in ori_text]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(ori_text)):
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
            neigbhours = word_candidate[ori_text[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        #if len(ori_text)>=100 and len(ori_text)<=256:
        if len(ori_text)<256 and True: #text_num<200
        #if True:  # text_num<200
            text_num+=1
            print('text id:{:d}'.format(text_i))
            print('text id:{:d}'.format(text_i),file=f)
            gra=get_Grad_for_text(ori_text, predictor)
            for r in range(1,4):
                print("r={:d}".format(r))
                print("r={:d}".format(r),file=f)
                Start_time=time.time()
                canFindAdv=exhausted_search(ori_text,text_syns,pertub_psts,r, predictor,gra,outfile=None)
                End_time=time.time()
                if canFindAdv==1:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    break
                else:
                    print("Certificated Robustness. Time: %.2f" % (End_time - Start_time),file=f)


    print (text_num)
    #f.close()
    return


if __name__ == "__main__":
    #serch_replace_toRobustnees()
    #serch_least_replace()
    increamental_for_r_1_4()
