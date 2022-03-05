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
from scipy.special import comb
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
import heapq

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
                    default=128,
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

args.task = 'mr'  # mr imdb
args.target_model = 'wordLSTM'  # wordLSTM bert

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
        model = Model(args.max_seq_length, args.word_embeddings_path, nclasses=args.nclasses).cuda()
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


def get_Grad_for_text(ori_text, predictor):                             #暂时没用
    ori_text1 = ori_text.copy()
    ori_text1 = [inv_full_dict[id] for id in ori_text1]  # 网络输入是词语
    ori_probs = predictor.text_pred([ori_text1])
    ori_label = torch.argmax(ori_probs,dim=1)

    #=======取梯度============
    if args.target_model == 'bert':
        inputs, token_noise = predictor.dataset.transform_text([ori_text1], batch_size=args.batch_size)  # 转成id，统一pad了
        for step, (input_ids, input_mask, segment_ids) in enumerate(inputs):  # input_mask0是mask位置
            input_ids = torch.tensor(input_ids.data, dtype=torch.float32, requires_grad=True)  # [1, 256] 后面是0 mask
            input_ids = input_ids.long()
            logits = predictor.model(input_ids.cuda(), segment_ids.cuda(), input_mask.cuda())
            ori_label = ori_label.item()
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
        ori_label = ori_label.item()
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


def filt_top_texts(adv_bach,adv_probs,ture_lable,preserve_bach_size):


    adv_bach_with_score=list()
    for i in range(0,len(adv_bach)):
        SS={
            'adv_text_id':i,
            'score':adv_probs[i][ture_lable].item()
        }
        adv_bach_with_score.append(SS)


    adv_bach_with_score=heapq.nsmallest(min(preserve_bach_size,len(adv_bach_with_score)),adv_bach_with_score,key=lambda e:e['score'])


    #adv_bach_with_score = sorted(adv_bach_with_score, key=lambda e: e['score'], reverse=False)
    adv_bach1=list()

    for i in range(0,min(preserve_bach_size,len(adv_bach_with_score))):
        adv_bach1.append(adv_bach[adv_bach_with_score[i]['adv_text_id']].copy())


    return adv_bach1

def update_firstRposition_score(position_conf_score,pos_list):
    position_conf_score1=position_conf_score.copy()
    for i in range(len(pos_list)):
        for j in range(len(position_conf_score1)):
            if position_conf_score1[j]['pos']==pos_list[i]:
                SS=position_conf_score1[j].copy()
                del position_conf_score1[j]
                position_conf_score1.insert(0,SS)
    return position_conf_score1




def update_position_score(true_lable,adv_bach,text_syns,t,position_conf_score,predictor): #更新t位置之后的打分.
    position_conf_score1=position_conf_score.copy()
    for i in range(t,len(position_conf_score)):
        adv_batch = list()
        for j in range(0,10):        #考虑10个文本给后面打分
            for k in range(0,len(text_syns[position_conf_score[i]['pos']])):
                a_adv = adv_bach[j].copy()
                a_adv[position_conf_score[i]['pos']] = text_syns[position_conf_score[i]['pos']][k]
                adv_batch.append(a_adv)

        adv_batch = [[inv_full_dict[id] for id in a] for a in adv_batch]
        adv_probs = predictor.text_pred(adv_batch)
        pre_confidence=adv_probs.cpu().numpy()[:,true_lable]
        position_conf_score1[i]['score']=np.min(pre_confidence)

    position_conf_score1=position_conf_score1[t:]
    position_conf_score1 = sorted(position_conf_score1, key=lambda e: e['score'], reverse=False)
    return position_conf_score[0:t]+position_conf_score1

def filt_best_adv(ori_text,true_lable,adv_bach,adv_labels,predictor):
    best_changeNum=9999
    best_adv=None
    changeList=None
    for i in range(len(adv_bach)):
        changeNum=0
        if adv_labels[i]!=true_lable:
            tempList = list()
            for j in range(len(ori_text)):
                if ori_text[j]!=adv_bach[i][j]:
                    tempList.append(j)
                    changeNum+=1
            if changeNum<best_changeNum:
                changeList=tempList
                best_changeNum=changeNum
                best_adv=adv_bach[i]
    #finetune一下
    No_changed=True
    while No_changed:
        adv_batch=list()
        for pos in changeList:
            adv_text=best_adv.copy()
            adv_text[pos]=ori_text[pos]
            adv_batch.append(adv_text)
        #=====判断
        adv_batch1=adv_batch.copy()
        adv_batch1 = [[inv_full_dict[id] for id in a] for a in adv_batch1]
        adv_probs = predictor.text_pred(adv_batch1)
        pre_confidence = adv_probs.cpu().numpy()[:, true_lable]
        adv_label = torch.argmax(adv_probs,dim=1)
        Re=torch.sum(adv_label != true_lable)

        if Re==0:
            No_changed=False
        else:
            i=np.argmin(pre_confidence)
            best_adv = adv_batch[i]
            del changeList[i]
            best_changeNum = best_changeNum - 1

    return best_adv.copy(),best_changeNum




def Pseudo_DP(ori_text,true_lable,text_syns,position_conf_score,r,predictor):
    adv_bach_size=128*15

    if len(position_conf_score)<r:
        print("certified robust at r={:d}".format(r))
        return 0

    #====排序conf_score===
    position_conf_score=sorted(position_conf_score,key=lambda e: e['score'],reverse=False)

    #========栈模式===========
    Sta = []
    Sta.append(-1)  # 初始化栈
    first_adv_bach=list()

    while len(Sta) > 0:  # 决策栈非空
        Sta[-1] += 1

        # while(Sta[-1]<len(text_syns[A_combin[len(Sta)-1]]))

        if Sta[-1] < len(text_syns[position_conf_score[len(Sta)-1]['pos']]):
            if len(Sta) == r:  # 栈满
                # sum++
                #subs_List.append(Sta.copy())
                adv_tex = ori_text.copy()
                for i in range(0,len(Sta)):
                    adv_tex[position_conf_score[i]['pos']] = text_syns[position_conf_score[i]['pos']][Sta[i]]
                first_adv_bach.append(adv_tex)
            else:
                Sta.append(-1)  # 压栈
        else:
            Sta.pop()
    # ==========栈决策结束=============

    # #=初始化第一批测试数据,前四个位置的全替换===
    # first_adv_bach=list()
    # for i1 in range(0,len(text_syns[position_conf_score[0]['pos']])):
    #     for i2 in range(0,len(text_syns[position_conf_score[1]['pos']])):
    #         for i3 in range(0, len(text_syns[position_conf_score[2]['pos']])):
    #             for i4 in range(0, len(text_syns[position_conf_score[3]['pos']])):
    #                 adv_tex = ori_text.copy()
    #                 adv_tex[position_conf_score[0]['pos']] = text_syns[position_conf_score[0]['pos']][i1]
    #                 adv_tex[position_conf_score[1]['pos']] = text_syns[position_conf_score[1]['pos']][i2]
    #                 adv_tex[position_conf_score[2]['pos']] = text_syns[position_conf_score[2]['pos']][i3]
    #                 adv_tex[position_conf_score[3]['pos']] = text_syns[position_conf_score[3]['pos']][i4]
    #                 first_adv_bach.append(adv_tex)

    first_adv_bach1 = [[inv_full_dict[id] for id in a] for a in first_adv_bach]
    first_adv_probs = predictor.text_pred(first_adv_bach1)
    first_adv_label = torch.argmax(first_adv_probs, dim=1)

    Re = torch.sum(first_adv_label != true_lable)
    if Re > 0:
        # print("r=4,Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
        #print("r={:d},Found an adversarial example.".format(r))
        return None,r
    else:
        pass
        #print("t={:d},failed.".format(r-1))

    last_adv_bach=first_adv_bach
    last_adv_probs=first_adv_probs
    for t in tqdm(range(r,len(position_conf_score))):

        last_adv_bach=filt_top_texts(last_adv_bach, last_adv_probs, true_lable, adv_bach_size) #过滤保留打分好的

        position_conf_score=update_position_score(true_lable, last_adv_bach, text_syns, t, position_conf_score, predictor)

        temp_adv_bach=list()      #每条数据扩大r位置可替换词个数倍后的待测试样本
        for tex_id in range(0,len(last_adv_bach)):
            for i in range(1,len(text_syns[position_conf_score[t]['pos']])):
                adv_tex=last_adv_bach[tex_id].copy()
                adv_tex[position_conf_score[t]['pos']] = text_syns[position_conf_score[t]['pos']][i]
                temp_adv_bach.append(adv_tex)

        #=====预测=====
        last_adv_bach=temp_adv_bach
        temp_adv_bach1 = [[inv_full_dict[id] for id in a] for a in temp_adv_bach]
        temp_adv_probs = predictor.text_pred(temp_adv_bach1)
        last_adv_probs=temp_adv_probs
        temp_adv_label = torch.argmax(temp_adv_probs, dim=1)


        Re = torch.sum(temp_adv_label != true_lable)
        if Re > 0:
            # print("r=4,Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            #print("t=%d,Found an adversarial example." % (t))
            best_adv,changeNum=filt_best_adv(ori_text,true_lable,last_adv_bach,temp_adv_label,predictor)

            #print("Best changed %d" % (changeNum))

            return best_adv,changeNum
        else:
            pass
            #print("t=%d,failed." % (t))

    return None,0



def exhausted_search_r1(ori_text,true_lable,text_syns,pertub_psts,r,predictor):
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return 0,None

    position_conf_score = list()
    for i in tqdm(range(len(pertub_psts))):
        adv_batch = list()
        for j in range(len(text_syns[pertub_psts[i]])):
            adv_tex = ori_text.copy()
            adv_tex[pertub_psts[i]] = text_syns[pertub_psts[i]][j]
            adv_batch.append(adv_tex)

        adv_batch = [ [inv_full_dict[id] for id in a] for a in adv_batch]
        adv_probs = predictor.text_pred(adv_batch)
        adv_label = torch.argmax(adv_probs, dim=1)


        pre_confidence = adv_probs.cpu().numpy()[:, true_lable]

        SS={
            'pos':pertub_psts[i],
            'score': np.min(pre_confidence)
        }
        position_conf_score.append(SS)

        Re=torch.sum(adv_label != true_lable)
        if Re>0:
            #print("Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            return 1,position_conf_score

    #print("Certificated Robustness. Time: %.2f" % (End_time - Begin_time),file=outfile)
    return 0,position_conf_score

def exhausted_search_S(ori_text,true_lable,text_syns,pertub_psts,r,predictor,position_conf_score):
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return 0
    Begin_time=time.time()

    gra=position_conf_score

    #=============每个位置的分数====================#
    #gra = sorted(gra, key=lambda e: e['score'], reverse=False)
    pertub_positions=list()
    for i in range(len(pertub_psts)):
    #for pos in pertub_psts:
        pos=pertub_psts[i]
        if(gra[i]['pos']!=pos):
            print('wrong!')
            exit(0)
        pos_dic={
            'pos':pos,
            'score': gra[i]['score']
        }
        pertub_positions.append(pos_dic)
    #===================================================




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

    Search_list1=sorted(Search_list1,key=lambda e: e['score'],reverse=False)
    Search_list=Search_list1

    best_combin_list=()
    best_score=9999

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
        adv_probs = predictor.text_pred(adv_batch)
        adv_label = torch.argmax(adv_probs,dim=1)

        #=====================算分===========================
        pre_confidence = adv_probs.cpu().numpy()[:, true_lable]

        temp_score = np.min(pre_confidence)

        if temp_score<best_score:
            best_combin_list=list() #清空
            for com in A_combin:
                best_combin_list.append(com['pos'])

        #=================================================

        Re=torch.sum(adv_label != true_lable)
        if Re>0:
            position_conf_score = update_firstRposition_score(position_conf_score, best_combin_list)
            return 1,position_conf_score

    return 0,position_conf_score

def exhausted_search(ori_text,true_lable,text_syns,pertub_psts,r, predictor,gra,outfile):
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return 0
    Begin_time=time.time()


    #=============每个位置的分数====================#
    # pertub_positions=list()
    # for pos in pertub_psts:
    #
    #     score=np.linalg.norm(gra[0][0][pos],ord=1)
    #     #计算每个位置的score
    #     pos_dic={
    #         'pos':pos,
    #         'score':score
    #     }
    #     pertub_positions.append(pos_dic)
    #===================================================

    #=============每个位置的分数====================#
    #gra = sorted(gra, key=lambda e: e['score'], reverse=False)
    pertub_positions=list()
    for i in range(len(pertub_psts)):
    #for pos in pertub_psts:
        pos=pertub_psts[i]
        if(gra[i]['pos']!=pos):
            print('wrong!')
            exit(0)
        pos_dic={
            'pos':pos,
            'score': gra[i]['score']
        }
        pertub_positions.append(pos_dic)
    #===================================================




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
    #######===调试代码==========
    # if len(Search_list)>10000:
    #     Search_list=Search_list[0:10000]
    #========调试代码结束====

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
        adv_probs = predictor.text_pred(adv_batch)
        adv_label = torch.argmax(adv_probs,dim=1)

        Re=torch.sum(adv_label != true_lable)
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
        if text_i < 9:
            continue
    #for text_index in range(20):
        text_i+=1
        # if text_i!=7:
        #     continue
        ori_text, true_label = texts[text_index], true_labels[text_index]

        ori_text = [x for x in ori_text if x != 50000]  # 注：文本里有50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        #========预测==========
        ori_text1=ori_text.copy()
        ori_text1 = [inv_full_dict[id] for id in ori_text1]  # 网络输入是词语
        ori_probs = predictor.text_pred([ori_text1])
        ori_label = torch.argmax(ori_probs, dim=1)

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
        ori_probs = predictor.text_pred([ori_text1])
        ori_label = torch.argmax(ori_probs, dim=1)

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

    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'r1_4_2.3'
    f = open(outfilename, 'w')         #out file
    #f=None
    text_i=0
    text_num=0


    #for text_index in range(len(texts)):
    for text_index in range(0,400):
        text_i+=1
        if text_num==200:
             break
        ori_text, true_label = texts[text_index], true_labels[text_index]

        if args.task == 'mr':
            ori_text = [x for x in ori_text if x != 50000]  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
        elif args.task == 'imdb':
            ori_text = [x for x in ori_text if x != 5169]  # 注：文本里有5169(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        #========预测==========
        ori_text1=ori_text.copy()
        ori_text1 = [inv_full_dict[id] for id in ori_text1] # 网络输入是词语 args.target_model == 'bert'
        ori_probs = predictor.text_pred([ori_text1])
        ori_label = torch.argmax(ori_probs, dim=1)

        if ori_label!=true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in ori_text]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(min(len(ori_text),args.max_seq_length)):
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
        if  True: #text_num<200
        #if True:  # text_num<200
            text_num+=1
            Start_time = time.time()
            print('text id:{:d}'.format(text_index))
            print('text id:{:d}'.format(text_index),file=f)
            #gra=get_Grad_for_text(ori_text, predictor)
            if len(pertub_psts) < 1:
                End_time = time.time()
                print("Certified Robustness. Time:{:2f}".format(End_time - Start_time))
                print("Certified Robustness. Time:{:2f}".format(End_time - Start_time), file=f)
                continue


            print("r=1")
            print("r=1", file=f)
            canFindAdv, position_conf_score = exhausted_search_r1(ori_text, true_label, text_syns, pertub_psts, 1, predictor)
            End_time = time.time()
            if canFindAdv == 1:
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time))
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time), file=f)
                #print("Text length: %d  Changed: %d   %.4f" % (len(ori_text), 1, 1 / float(len(ori_text))))
                #print("Text length: %d  Changed: %d   %.4f" % (len(ori_text), 1, 1 / float(len(ori_text))), file=f)
                continue  # 处理下一条
            else:
                print("Certified Robustness. Time: %.2f" % (End_time - Start_time))
                print("Certified Robustness. Time: %.2f" % (End_time - Start_time), file=f)

            if len(pertub_psts) < 2:
                continue
            canFindAdv = 0
            for r in range(2,5):
                print("r={:d}".format(r))
                print("r={:d}".format(r),file=f)

                Start_time=time.time()
                canFindAdv, position_conf_score = exhausted_search_S(ori_text, true_label, text_syns, pertub_psts, r,
                                                                     predictor, position_conf_score)
                End_time=time.time()
                if canFindAdv==1:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    break
                else:
                    print("Certificated Robustness. Time: %.2f" % (End_time - Start_time),file=f)

    print (text_num)
    f.close()
    return

def MPDP():
    # 1. 加载原始数据和预训练好的模型
    predictor = load_model()
    texts = dataset.test_seqs2  # 未填充至统一长度；且已转化成词编号
    true_labels = dataset.test_y

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN', 'NNP', 'JJR', 'JJS', 'JJ', 'RBR', 'RBS', 'RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'VB']

    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'MPDP2.4'
    f = open(outfilename, 'w')         #out file
    #f = None
    text_i = 0
    text_num = 0

    #for text_index in range(len(texts)):
    for text_index in range(0,1200):
        # if text_num>1000:
        #     break
        text_i += 1

        ori_text, true_label = texts[text_index], true_labels[text_index]

        if args.task == 'mr':
            ori_text = [x for x in ori_text if x != 50000]  # 注：50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
        elif args.task == 'imdb':
            ori_text = [x for x in ori_text if x != 5169]  # 注：文本里有5169(\x85)，而词性标注结果没有，导致无法对齐。将其删除

        # if len(ori_text) > args.max_seq_length:
        #     print('Skip too long.')
        #     continue

        # ========预测==========
        ori_text1 = ori_text.copy()
        ori_text1 = [inv_full_dict[id] for id in ori_text1]  # 网络输入是词语 args.target_model == 'bert'
        ori_probs = predictor.text_pred([ori_text1])
        ori_label = torch.argmax(ori_probs, dim=1)

        if ori_label != true_label:
            # print("Predict false")
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
        # if len(ori_text)>=100 and len(ori_text)<=256:
        #if len(ori_text) < 500 and len(ori_text) > args.max_seq_length:  # text_num<200    if len(ori_text) > args.max_seq_length:
        if True:
        #if True and text_index>=133:  # text_num<200 text_index>=22  43 48 过滤成功 133 有毒
            text_num += 1
            Start_time = time.time()
            print('text id:{:d}'.format(text_index))
            print('text id:{:d}'.format(text_index),file=f)

            if len(pertub_psts)<1:
                print("Certified Robustness. Time:{:2f}".format(End_time - Start_time))
                print("Certified Robustness. Time:{:2f}".format(End_time - Start_time),file=f)
                continue

            #gra = get_Grad_for_text(ori_text, predictor)      #取梯度

            print("r=1")
            print("r=1",file=f)
            canFindAdv,position_conf_score=exhausted_search_r1(ori_text,true_label, text_syns, pertub_psts, 1, predictor)
            End_time = time.time()
            if canFindAdv == 1:
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time))
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time), file=f)
                print("Text length: %d  Changed: %d   %.4f" % (len(ori_text),1,1/float(len(ori_text))))
                print("Text length: %d  Changed: %d   %.4f" % (len(ori_text), 1, 1 / float(len(ori_text))),file=f)
                continue     #处理下一条
            else:
                print("Certified Robustness. Time: %.2f" % (End_time - Start_time))
                print("Certified Robustness. Time: %.2f" % (End_time - Start_time), file=f)

            if len(pertub_psts)<2:
                continue
            r=2
            canFindAdv=0
            break_flag=False
            while (r<=4 and comb(len(pertub_psts),r)<=1000)or r==2:
                print('r={:d}'.format(r))
                print('r={:d}'.format(r),file=f)
                canFindAdv = 0
                canFindAdv, position_conf_score = exhausted_search_S(ori_text, true_label, text_syns, pertub_psts, r,
                                                                     predictor, position_conf_score)
                End_time = time.time()
                if canFindAdv == 1:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time))
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_text), r,float(r)/len(ori_text)))
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_text), r, float(r)/len(ori_text)),file=f)
                    break_flag=True
                    break
                else:
                    print("Certified Robustness. Time: %.2f" % (End_time - Start_time))
                    print("Certified Robustness. Time: %.2f" % (End_time - Start_time), file=f)
                    r=r+1
                    if len(pertub_psts)<r:
                        break_flag = True
                        break

            if break_flag:
                continue  # 处理下一条
            else:
                print("Apply Pseudo DP")
                print("Apply Pseudo DP",file=f)
                _,best_r=Pseudo_DP(ori_text, true_label, text_syns, position_conf_score,r,predictor)
                End_time = time.time()
                #if best_r>0 and float(best_r)/len(ori_text)<0.25:
                if best_r > 0:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time))
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_text), best_r,float(best_r)/len(ori_text)))
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_text), best_r, float(best_r) / len(ori_text)),file=f)
                else:
                    print("Failed. Time: %.2f" % (End_time - Start_time))
                    print("Failed. Time: %.2f" % (End_time - Start_time), file=f)
        print("\n")

    print(text_num)
    f.close()
    return

if __name__ == "__main__":
    #serch_replace_toRobustnees()
    #serch_least_replace()
    #increamental_for_r_1_4()
    MPDP()
