# /usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
from time import *
import torch
import time
from tqdm import tqdm
import pickle
from scipy.special import comb
from keras.layers import *
import itertools
import heapq

from model_nli import Model as bdlstm_model
from encap_snli_bert import Model as bert_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='snli')
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--nclasses", type=int, default=3, help="How many classes for classification.")
parser.add_argument("--target_model", type=str, default='bdlstm', help="For snli: bdlstm or bert")
args = parser.parse_args()

with open('dataset/nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
with open('dataset/word_candidates_sense_top5.pkl','rb') as fp:
    word_candidate=pickle.load(fp)
with open('dataset/all_seqs.pkl', 'rb') as fh:
    _, _, test = pickle.load(fh)
with open('dataset/pos_tags_test.pkl','rb') as fp:
    pos_tags = pickle.load(fp)

test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]
# print('the length of test cases is:', len(test_s1))
true_labels = test['label']
np.random.seed(3333)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}


"""加载模型"""
def load_model():
    # construct the model
    print("Building Model...")
    if args.target_model == 'bdlstm':
        model = bdlstm_model()  # 模型初始化的时候回自动加载断点等
        print(args.target_model)
    elif args.target_model == 'bert':
        model = bert_model(inv_vocab)  # 模型初始化的时候回自动加载断点等
        print(args.target_model)

    print("Model built!")
    predictor = model
    return predictor



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




def update_position_score(true_lable,ori_s1, adv_bach,text_syns,t,position_conf_score,predictor): #更新t位置之后的打分.
    position_conf_score1=position_conf_score.copy()
    for i in range(t,len(position_conf_score)):
        adv_batch = list()
        for j in range(0,10):        #考虑10个文本给后面打分
            for k in range(0,len(text_syns[position_conf_score[i]['pos']])):
                a_adv = adv_bach[j].copy()
                a_adv[position_conf_score[i]['pos']] = text_syns[position_conf_score[i]['pos']][k]
                adv_batch.append(a_adv)

        ori_s1s = [ori_s1] * len(adv_batch)
        adv_probs = predictor.pred(ori_s1s, adv_batch)
        pre_confidence = adv_probs[:, true_lable]
        position_conf_score1[i]['score'] = np.min(pre_confidence)

    position_conf_score1=position_conf_score1[t:]
    position_conf_score1 = sorted(position_conf_score1, key=lambda e: e['score'], reverse=False)
    return position_conf_score[0:t]+position_conf_score1

def filt_best_adv(ori_s1, ori_s2,true_lable,adv_bach,adv_labels,predictor):
    best_changeNum=9999
    best_adv=None
    changeList=None
    for i in range(len(adv_bach)):
        changeNum=0
        if adv_labels[i]!=true_lable:
            tempList = list()
            for j in range(len(ori_s2)):
                if ori_s2[j]!=adv_bach[i][j]:
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
            adv_text[pos]=ori_s2[pos]
            adv_batch.append(adv_text)
        #=====判断
        adv_batch1=adv_batch.copy()
        ori_s1s = [ori_s1] * len(adv_batch1)
        adv_probs = predictor.pred(ori_s1s, adv_batch1)
        pre_confidence = adv_probs[:, true_lable]
        adv_label = np.argmax(adv_probs, axis=1)
        Re = np.sum(adv_label != true_lable)

        if Re==0:
            No_changed=False
        else:
            i=np.argmin(pre_confidence)
            best_adv = adv_batch[i]
            del changeList[i]
            best_changeNum = best_changeNum - 1

    return best_adv.copy(),best_changeNum




def Pseudo_DP(ori_s1, ori_s2,true_lable,text_syns,position_conf_score,r,predictor):
    adv_bach_size=128*12

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
                adv_tex = ori_s2.copy()
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

    # first_adv_bach1 = [[inv_full_dict[id] for id in a] for a in first_adv_bach]
    # first_adv_probs = predictor.text_pred(first_adv_bach1)
    # first_adv_label = torch.argmax(first_adv_probs, dim=1)

    ori_s1s = [ori_s1] * len(first_adv_bach)
    first_adv_probs = predictor.pred(ori_s1s, first_adv_bach)
    first_adv_label = np.argmax(first_adv_probs, axis=1)

    Re = np.sum(first_adv_label != true_lable)
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
        position_conf_score=update_position_score(true_lable, ori_s1, last_adv_bach, text_syns, t, position_conf_score, predictor)

        temp_adv_bach=list()      #每条数据扩大r位置可替换词个数倍后的待测试样本
        for tex_id in range(0,len(last_adv_bach)):
            for i in range(1,len(text_syns[position_conf_score[t]['pos']])):
                adv_tex=last_adv_bach[tex_id].copy()
                adv_tex[position_conf_score[t]['pos']] = text_syns[position_conf_score[t]['pos']][i]
                temp_adv_bach.append(adv_tex)

        #=====预测=====
        last_adv_bach=temp_adv_bach
        ori_s1s = [ori_s1] * len(temp_adv_bach)
        temp_adv_bach1 = temp_adv_bach.copy()
        temp_adv_probs = predictor.pred(ori_s1s, temp_adv_bach1)
        last_adv_probs=temp_adv_probs
        temp_adv_label = np.argmax(temp_adv_probs, axis=1)



        Re = np.sum(temp_adv_label != true_lable)
        if Re > 0:
            # print("r=4,Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            #print("t=%d,Found an adversarial example." % (t))
            st_time = time.time()
            best_adv,changeNum=filt_best_adv(ori_s1, ori_s2, true_lable,last_adv_bach,temp_adv_label,predictor)
            end_time = time.time()
            print("filt_best_adv time:{:.2f}".format(end_time - st_time))
            #print("Best changed %d" % (changeNum))

            return best_adv,changeNum
        else:
            pass
            #print("t=%d,failed." % (t))

    return None,0



def exhausted_search_r1(ori_s1, ori_s2, true_lable,text_syns,pertub_psts,r,predictor):
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return 0,None

    position_conf_score = list()
    for i in tqdm(range(len(pertub_psts))):
        adv_batch = list()
        for j in range(len(text_syns[pertub_psts[i]])):
            adv_tex = ori_s2.copy()
            adv_tex[pertub_psts[i]] = text_syns[pertub_psts[i]][j]
            adv_batch.append(adv_tex)

        # adv_batch = [ [inv_full_dict[id] for id in a] for a in adv_batch]
        # adv_probs = predictor.text_pred(adv_batch)
        # adv_label = torch.argmax(adv_probs, dim=1)

        ori_s1s = [ori_s1] * len(adv_batch)
        adv_probs = predictor.pred(ori_s1s, adv_batch)
        adv_label = np.argmax(adv_probs, axis=1)


        pre_confidence = adv_probs[:, true_lable]

        SS={
            'pos':pertub_psts[i],
            'score': np.min(pre_confidence)
        }
        position_conf_score.append(SS)

        Re=np.sum(adv_label != true_lable)
        if Re>0:
            #print("Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            return 1,position_conf_score

    #print("Certificated Robustness. Time: %.2f" % (End_time - Begin_time),file=outfile)
    return 0,position_conf_score

def exhausted_search_S(ori_s1, ori_s2,true_lable,text_syns,pertub_psts,r,predictor,position_conf_score):
    if len(pertub_psts)==0:
        print("Certificated Robustness.")
        return 0


    gra=position_conf_score

    #=============每个位置的分数====================#
    #gra = sorted(gra, key=lambda e: e['score'], reverse=False)
    pertub_positions=list()
    for i in range(len(pertub_psts)):
    #for pos in pertub_psts:
        pos=pertub_psts[i]
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
            adv_tex = ori_s2.copy()
            for i,Pos in enumerate(A_combin):
                adv_tex[Pos['pos']]=text_syns[Pos['pos']][A_sub[i]]
            adv_batch.append(adv_tex)


        # adv_batch = [ [inv_full_dict[id] for id in a] for a in adv_batch]
        # #print(adv_batch)
        # adv_probs = predictor.text_pred(adv_batch)
        # adv_label = torch.argmax(adv_probs,dim=1)

        ori_s1s = [ori_s1] * len(adv_batch)
        adv_probs = predictor.pred(ori_s1s, adv_batch)
        adv_label = np.argmax(adv_probs, axis=1)

        #=====================算分===========================
        pre_confidence = adv_probs[:, true_lable]

        temp_score = np.min(pre_confidence)

        if temp_score<best_score:
            best_combin_list=list() #清空
            for com in A_combin:
                best_combin_list.append(com['pos'])

        #=================================================

        Re=np.sum(adv_label != true_lable)
        if Re>0:
            position_conf_score = update_firstRposition_score(position_conf_score, best_combin_list)
            return 1,position_conf_score

    return 0,position_conf_score

def exhausted_search(ori_s1,ori_s2,true_lable,text_syns,pertub_psts,r, predictor,gra,outfile):
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
    if len(Search_list)>10000:
        Search_list=Search_list[0:10000]
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
            adv_tex = ori_s2.copy()
            for i,Pos in enumerate(A_combin):
                adv_tex[Pos['pos']]=text_syns[Pos['pos']][A_sub[i]]
            adv_batch.append(adv_tex)

        # adv_batch = [ [inv_full_dict[id] for id in a] for a in adv_batch]
        # #print(adv_batch)
        # adv_probs = predictor.text_pred(adv_batch)
        # adv_label = torch.argmax(adv_probs,dim=1)

        ori_s1s = [ori_s1] * len(adv_batch)
        adv_probs = predictor.pred(ori_s1s, adv_batch)
        adv_label = np.argmax(adv_probs, axis=1)

        Re=np.sum(adv_label != true_lable)
        if Re>0:
            End_time=time.time()
            #print("Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            return 1
    End_time = time.time()
    #print("Certificated Robustness. Time: %.2f" % (End_time - Begin_time),file=outfile)
    return 0


def serch_least_replace():
    # 1. 加载预训练好的模型
    predictor = load_model()

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN','NNP','JJR', 'JJS','JJ', 'RBR', 'RBS','RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VB']

    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'r_4'
    f = open(outfilename, 'w')         # out file
    text_i=0
    for text_index in range(len(test_s1)):
        if text_i < 9:
            continue
    #for text_index in range(20):
        text_i+=1
        # if text_i!=7:
        #     continue
        ori_s1, ori_s2, true_label = test_s1[text_index], test_s2[text_index], true_labels[text_index]


        #========预测==========

        ori_probs = predictor.pred([ori_s1], [ori_s2])
        ori_label = np.argmax(ori_probs, axis=1)

        if ori_label!=true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in ori_s2]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(ori_s2)):
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
            neigbhours = word_candidate[ori_s2[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        #if len(ori_text)>=100 and len(ori_text)<=256:
        if len(ori_s2) <= 128:
            exhausted_search(ori_s1, ori_s2, text_syns,pertub_psts,4, predictor,outfile=f)

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

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN','NNP','JJR', 'JJS','JJ', 'RBR', 'RBS','RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VB']
    RobustList=read_robustness_id('mr_bert_r_3')
    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'r_4_Comple'
    f = open(outfilename, 'w')         #out file
    text_i=0
    for text_index in range(len(test_s2)):
    #for text_index in range(20):
        text_i+=1
        # if text_i!=7:
        #     continue
        ori_s1, ori_s2, true_label = test_s1[text_index], test_s2[text_index], true_labels[text_index]

        #========预测==========
        ori_probs = predictor.pred([ori_s1], [ori_s2])
        ori_label = np.argmax(ori_probs, axis=1)

        if ori_label!=true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in ori_s2]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(ori_s2)):
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
            neigbhours = word_candidate[ori_s2[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        #if len(ori_text)>=100 and len(ori_text)<=256:
        if len(ori_s2) <= 128 and text_i in RobustList and (len(pertub_psts)>=4):
            print(text_i)
            print(text_i,file=f)
            exhausted_search(ori_s1, ori_s2, text_syns, pertub_psts,4, predictor,outfile=f)

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


    #test_x = [[inv_full_dict[w] for w in x] for x in texts]  # 网络输入是词语
    # # 原始测试集准确率
    # test_x = [[x for x in text if x != 50000] for text in test_x ]  # 注：文本里有50000(\x85)，而词性标注结果没有，导致无法对齐。将其删除
    #
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
        if len(ori_text)<256 and True: #text_num<200
        #if True:  # text_num<200
            text_num+=1
            print('text id:{:d}'.format(text_i))
            print('text id:{:d}'.format(text_i),file=f)
            gra=get_Grad_for_text(ori_text, predictor)


            for r in range(1,4):
                print("r={:d}".format(r))
                print("r={:d}".format(r),file=f)

                exhausted_search_r1(ori_text,text_syns,pertub_psts,r, predictor,outfile=None)#测试

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

def increamental_search():
    # 1. 加载预训练好的模型
    predictor = load_model()

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN', 'NNP', 'JJR', 'JJS', 'JJ', 'RBR', 'RBS', 'RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'VB']

    outfilename='FindAdresult/'+args.task+'_'+args.target_model+'_'+'MPDP'
    f = open(outfilename, 'w')         #out file
    #f = None
    text_i = 0
    text_num = 0

    for text_index in range(len(test_s1)):
        # if text_index < 440:  # 266 bug点 for bdlstm 1942 for bert
        #     continue
        # for text_index in range(201):
        text_i += 1

        ori_s1, ori_s2, true_label = test_s1[text_index], test_s2[text_index], true_labels[text_index]

        if len(ori_s2) > args.max_seq_length:
            print('Skip too long.')
            continue

        # ========预测==========

        ori_label = np.argmax(predictor.pred([ori_s1], [ori_s2])[0])
        if ori_label != true_label:
            # print("Predict false")
            continue

        ori_s2 = ori_s2.copy()
        text_syns = [[t] for t in ori_s2]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(ori_s2)):
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
            neigbhours = word_candidate[ori_s2[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        # if len(ori_text)>=100 and len(ori_text)<=256:
        if len(ori_s2) < 256:  # text_num<200
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
            canFindAdv,position_conf_score=exhausted_search_r1(ori_s1, ori_s2, true_label, text_syns, pertub_psts, 1, predictor)
            End_time = time.time()
            if canFindAdv == 1:
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time))
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time), file=f)
                print("Text length: %d  Changed: %d   %.4f" % (len(ori_s2), 1, 1/float(len(ori_s2))))
                print("Text length: %d  Changed: %d   %.4f" % (len(ori_s2), 1, 1 / float(len(ori_s2))), file=f)
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
                canFindAdv, position_conf_score = exhausted_search_S(ori_s1, ori_s2, true_label, text_syns, pertub_psts, r,
                                                                     predictor, position_conf_score)
                End_time = time.time()
                if canFindAdv == 1:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time))
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_s2), r,float(r)/len(ori_s2)))
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_s2), r, float(r)/len(ori_s2)),file=f)
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
                _,best_r=Pseudo_DP(ori_s1, ori_s2, true_label, text_syns, position_conf_score,r,predictor)
                End_time = time.time()
                #if best_r>0 and float(best_r)/len(ori_text)<0.25:
                if best_r > 0:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time))
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_s2), best_r,float(best_r)/len(ori_s2)))
                    print("Text length: %d  Changed: %d  %.4f" % (len(ori_s2), best_r,float(best_r)/len(ori_s2)), file=f)
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
    increamental_search()
