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
from PDPsetting import *
from config import args
from attack_classification_hownet_top5 import get_syn_words

import os
os.environ["CUDA_VISIBLE_DEVICES"] = Use_GPU_id


def TopK_texts(adv_bach,adv_probs,ture_lable,K):

    adv_bach_with_score=list()
    for i in range(0,len(adv_bach)):
        SS={
            'adv_text_id':i,
            'score':adv_probs[i][ture_lable].item()
        }
        adv_bach_with_score.append(SS)

    adv_bach_with_score=heapq.nsmallest(min(K,len(adv_bach_with_score)),adv_bach_with_score,key=lambda e:e['score'])

    adv_bach1=list()
    for i in range(0,min(K,len(adv_bach_with_score))):
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

def Look_Ahead(ori_text,true_lable,adv_bach,text_syns,t,position_conf_score,predictor): #更新t位置之后的打分.
    position_conf_score1=position_conf_score.copy()

    pos_len=len(position_conf_score)
    sample_len=min(10,len(adv_bach))

    if args.task=='imdb' and args.target_model=='bert':
        if t>2 and (t % 10)!=0:
            pos_len=min(int(t+len(position_conf_score)*0.15),len(position_conf_score))
        sample_len=1

    adv_batch1 = list()
    record_list=list()
    last_Num=0
    for i in range(t,pos_len):
        SS={
            'start':last_Num,
            'end':-1
        }
        for j in range(0,sample_len):        #考虑10个文本给后面打分
            for k in range(0,len(text_syns[position_conf_score[i]['pos']])):
                a_adv = adv_bach[j].copy()
                a_adv[position_conf_score[i]['pos']] = text_syns[position_conf_score[i]['pos']][k]
                adv_batch1.append(a_adv)
                last_Num+=1
        SS['end']=last_Num
        record_list.append(SS)

    # adv_batch1 = [[args.inv_full_dict[id] for id in a] for a in adv_batch1]
    adv_probs = predictor([ori_text for i in range(len(adv_batch1))], adv_batch1)

    pre_confidence=adv_probs.cpu().numpy()[:,true_lable]

    for i in range(t, pos_len):
        position_conf_score1[i]['score'] = np.min(pre_confidence[record_list[i-t]['start']:record_list[i-t]['end']])

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
        # adv_batch1 = [[args.inv_full_dict[id] for id in a] for a in adv_batch1]
        adv_probs = predictor([ori_text for i in range(len(adv_batch1))],adv_batch1)
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

def Pseudo_DP(ori_text,true_lable,text_syns,pertub_psts,predictor):

    if args.task=='imdb' and args.target_model=='bert':
        adv_bach_size=TOPK_bach
    else:
        adv_bach_size=TOPK_bach

    #=============预处理阶段============================
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return None,0

    position_conf_score = list()
    first_adv_bach= list()
    first_adv_probs=None
    print("Preprocessing..")
    for i in tqdm(range(len(pertub_psts))):
        adv_batch = list()
        for j in range(1,len(text_syns[pertub_psts[i]])):
            adv_tex = ori_text.copy()
            adv_tex[pertub_psts[i]] = text_syns[pertub_psts[i]][j]
            adv_batch.append(adv_tex)

        first_adv_bach = first_adv_bach + adv_batch
        # adv_batch1 = [ [args.inv_full_dict[id] for id in a] for a in adv_batch]
        # try:
        adv_probs = predictor([ori_text for i in range(len(adv_batch))], adv_batch)
        # except:
        #     print(pertub_psts, text_syns, adv_batch)

        if first_adv_probs is None:
            first_adv_probs=adv_probs
        else:
            first_adv_probs=torch.cat((first_adv_probs,adv_probs),0)

        adv_label = torch.argmax(adv_probs,dim=1)

        pre_confidence = adv_probs.cpu().numpy()[:, true_lable]

        SS={
            'pos':pertub_psts[i],
            'score': np.min(pre_confidence)
        }
        position_conf_score.append(SS)

        Re=torch.sum(adv_label != true_lable)

        if Re>0:
            # return_adv=None
            # for ad_i in range(adv_label.shape[-1]):
            #     if adv_label[ad_i]!=true_lable:
            #         return_adv=adv_batch[ad_i]
            #print("Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            return None,1
    #==================================================================


    if len(position_conf_score)<2:
        print("certified Robustness at r=1")
        return None,0

    #====排序conf_score===
    position_conf_score=sorted(position_conf_score,key=lambda e: e['score'],reverse=False)

    last_adv_bach=first_adv_bach
    last_adv_probs=first_adv_probs


    for t in tqdm(range(1,len(position_conf_score))):
        last_adv_bach=TopK_texts(last_adv_bach, last_adv_probs, true_lable, adv_bach_size) #过滤保留打分好的

        position_conf_score=Look_Ahead(ori_text, true_lable, last_adv_bach, text_syns, t, position_conf_score, predictor)
        #print(position_conf_score[t]['pos'],position_conf_score[t]['score'])

       # s_time = time.time()
        temp_adv_bach=list()      #每条数据扩大r位置可替换词个数倍后的待测试样本
        for tex_id in range(0,len(last_adv_bach)):
            for i in range(1,len(text_syns[position_conf_score[t]['pos']])):
                adv_tex=last_adv_bach[tex_id].copy()
                adv_tex[position_conf_score[t]['pos']] = text_syns[position_conf_score[t]['pos']][i]
                temp_adv_bach.append(adv_tex)

        #=====预测=====
        last_adv_bach=temp_adv_bach
        # temp_adv_bach1 = [[args.inv_full_dict[id] for id in a] for a in temp_adv_bach]
        temp_adv_probs = predictor([ori_text for i in range(len(temp_adv_bach))], temp_adv_bach)
        last_adv_probs=temp_adv_probs
        temp_adv_label = torch.argmax(temp_adv_probs,dim=1)
        #e_time = time.time()
        #print("predict time:{:.2f}".format(e_time - s_time))
        Re = torch.sum(temp_adv_label != true_lable)
        if Re > 0:
            # print("r=4,Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            #print("t=%d,Found an adversarial example." % (t))
            best_adv,changeNum=filt_best_adv(ori_text,true_lable,last_adv_bach,temp_adv_label,predictor)
            #print("Best changed %d" % (changeNum))

            # best_adv = [inv_full_dict[id] for id  in best_adv]
            # best_adv=' '.join(best_adv)
            # print(best_adv)

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

        # adv_batch = [ [args.inv_full_dict[id] for id in a] for a in adv_batch]
        adv_probs = predictor(adv_batch,adv_batch)
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

def luanch_attack():
    # 1. 加载原始数据和预训练好的模型
    """Get data to attack"""
    # origion
    # texts, labels = dataloader.read_corpus(args.dataset_path, clean = False, FAKE = False, shuffle = False)
    # tiz
    if args.train_set:
    # train set
        texts = args.datasets.train_seqs2
        texts = [[args.inv_full_dict[word] for word in text]for text in texts]  # 网络输入是词语
        labels = args.datasets.train_y
        data = list(zip(texts, labels))
        data = data[:int(0.25*(len(labels)))]  # 训练集取前25%攻击
    else:
    # test set
        print("test set")
        texts = args.datasets.test_seqs2
        texts = [[args.inv_full_dict[word] for word in text]for text in texts]  # 网络输入是词语
        labels = args.datasets.test_y
        data = list(zip(texts, labels))
        data = data[:400]  # 测试集取前200条攻击
    print("Data import finished!")
    print('Attaked data size', len(data))

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args, args.max_seq_length, args.embedding, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        print('Load model from: %s' % args.target_model_path)
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args, args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
        print('Load model from: %s' % args.target_model_path)

    model.eval()
    predictor = model.text_pred()


    # predictor = load_model()
    # texts = dataset.test_seqs2  # 未填充至统一长度；且已转化成词编号
    # true_labels = dataset.test_y

    if result_filename!=None:
        f = open(result_filename, write_mod)         #out file
    else:
        f=None

    attack_num=0
    if attack_data_num!=0:
        attack_num=attack_data_num
    else:
        attack_num=len(texts)

    text_num = 0
    T_pre = 0  # 记录正确预测个数
    # 2. 加载对抗结果，获得对抗样本
    for text_index in range(attack_start_textid,attack_start_textid+attack_num):

        # if text_index not in sp_texts:
        #     continue
        ori_text, true_label = data[text_index][0], data[text_index][1]
        ori_text = [word for word in ori_text if word != '\x85']

        # ========预测==========
        ori_text1 = ori_text.copy()
        orig_probs = predictor([ori_text1], [ori_text1])
        orig_label = torch.argmax(orig_probs,dim=1)

        #
        # aad_text1 = [inv_full_dict[id] for id in aad_text]
        # aad_probs = predictor.text_pred([aad_text1])
        # aad_label = torch.argmax(aad_probs, dim=1)
        #
        # if aad_label!=ori_label:
        #     tt_result=np.array(aad_text)!=np.array(ori_text)
        #     print(tt_result)
        #     print(np.argwhere(tt_result==True))
        #     print('hahahaha')

        if orig_label != true_label or T_pre>200:
            # print("Predict false")
            continue
        else:
            T_pre+=1

        # 获得同义词
        pertub_psts = []
        length = min(len(ori_text), args.max_seq_length)
        text_syns = get_syn_words(text_index, [i for i in range(length)], ori_text)  # 扰动位置为所有
        for i in range(len(ori_text)):
            text_syns[i] = [ori_text[i]] + text_syns[i]
        for ii in range(length):
            if len(text_syns[ii]) > 1: #有同义词
                pertub_psts.append(ii)



        # text_syns = [[t] for t in ori_text]  # 保存该文本各个位置同义词（包含自己。）
        # pertub_psts = []  # 保存该文本的所有可替换位置
        # pos_tag = pos_tags[text_index]
        # for i in range(min(len(ori_text),args.max_seq_length)):
        #     pos = pos_tag[i][1]  # 当前词语词性
        #     # 若当前词语词性不为形容词、名词、副词和动词，不替换
        #     if pos not in pos_list:
        #         continue
        #     if pos.startswith('JJ'):
        #         pos = 'adj'
        #     elif pos.startswith('NN'):
        #         pos = 'noun'
        #     elif pos.startswith('RB'):
        #         pos = 'adv'
        #     elif pos.startswith('VB'):
        #         pos = 'verb'
        #     neigbhours = word_candidate[ori_text[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
        #     if len(neigbhours) == 0:
        #         continue
        #
        #     pertub_psts.append(i)
        #     text_syns[i] += neigbhours

        if len(ori_text) <filt_text_length:  # text_num<200
            text_num += 1
            Start_time = time.time()
            print('text id:{:d}'.format(text_index))
            print('text id:{:d}'.format(text_index),file=f)

            if len(pertub_psts)<1:
                End_time= time.time()
                print("Certified Robustness. Time:{:2f}".format(End_time - Start_time))
                print("Certified Robustness. Time:{:2f}".format(End_time - Start_time),file=f)
                continue

            #gra = get_Grad_for_text(ori_text, predictor)      #取梯度

            print("Apply Pseudo DP")
            print("Apply Pseudo DP",file=f)
            Find_adv,best_r=Pseudo_DP(ori_text, true_label, text_syns, pertub_psts,predictor)
            End_time = time.time()
            if best_r > 0:
                # F_probs = predictor([ori_text], [Find_adv])
                # F_label = torch.argmax(F_probs, dim=1)
                # if F_label != true_label:
                print("Found an adversarial example. Time: %.2f" % (End_time - Start_time))
                print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                print("Text length: %d  Changed: %d  %.4f" % (len(ori_text), best_r,float(best_r)/len(ori_text)))
                print("Text length: %d  Changed: %d  %.4f" % (len(ori_text), best_r, float(best_r) / len(ori_text)),file=f)
                # else:
                # print("Failed. Time: %.2f" % (End_time - Start_time))
                # print("Failed. Time: %.2f" % (End_time - Start_time), file=f)
            else:
                print("Failed. Time: %.2f" % (End_time - Start_time))
                print("Failed. Time: %.2f" % (End_time - Start_time), file=f)
        #print("\n")

    print(text_num)
    if result_filename!=None:
        f.close()
    return

if __name__ == "__main__":
    luanch_attack()