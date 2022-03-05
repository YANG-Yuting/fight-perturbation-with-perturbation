# /usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
from time import *
import torch
import time
from tqdm import tqdm
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
import keras.backend as K
import itertools
from data import get_nli, get_batch, build_vocab
from torch.autograd import Variable
from torchsummary import summary

from model_nli import Model as bdlstm_model
from encap_snli_bert import Model as bert_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# args
parser = argparse.ArgumentParser()
parser.add_argument("--nclasses", type=int, default=3, help="How many classes for classification.")
parser.add_argument("--target_model", type=str, default='bdlstm', help="For snli: bdlstm or bert")
args = parser.parse_args()

with open('dataset/nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
with open('dataset/word_candidates_sense_top5.pkl','rb') as fp:
    word_candidate=pickle.load(fp)
with open('dataset/all_seqs.pkl', 'rb') as fh:
    train, valid, test = pickle.load(fh)
with open('dataset/pos_tags_test.pkl','rb') as fp:
    pos_tags = pickle.load(fp)

test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]
# print('the length of test cases is:', len(test_s1))
np.random.seed(3333)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}

args.target_model = 'bdlstm'  # for snli: bdlstm bert


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

def update_firstRposition_score(position_conf_score,pos_list):
    position_conf_score1=position_conf_score.copy()
    for i in range(len(pos_list)):
        for j in range(len(position_conf_score1)):
            if position_conf_score1[j]['pos']==pos_list[i]:
                SS=position_conf_score1[j].copy()
                del position_conf_score1[j]
                position_conf_score1.insert(0,SS)
    return position_conf_score1

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
    return 0 ,position_conf_score

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


def exhausted_search(s1, s2, true_label, text_syns, pertub_psts, r, predictor, outfile):
    if len(pertub_psts) == 0:
        print("Certificated Robustness.")
        return 0
    Begin_time = time.time()
    s2_1 = s2.copy()
    ori_label = np.argmax(predictor.pred([s1], [s2])[0])
    predict = (ori_label == true_label)  # 是否正确预测


    N_ba=3
    #=======取梯度============
    if args.target_model == 'bdlstm':
        # for name, parms in predictor.nli_net.named_parameters():
        #     print(name, parms.requires_grad)
        #     gra = parms.grad
        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook
        # s1: list of len 19, s2: list of len 7
        # prepare batch
        s1_batch, s1_len = predictor.get_batch([s1], predictor.word_vec, 300)  # [19,1,300], [19]
        s2_batch, s2_len = predictor.get_batch([s2], predictor.word_vec, 300)  # [7,1,300], [7]
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda(), requires_grad=True)
        s2_batch.register_hook(save_grad('s2'))
        # model forward
        output = predictor.nli_net((s1_batch, s1_len), (s2_batch, s2_len))  # [[a,b,c]]
        # output = output.data.max(1)[1]
        output = output[0][ori_label]
        # backward
        output.backward()
        # get grad
        grad = grads['s2']  # [7,1,300]
        grad = grad.permute(1, 0, 2)  # [1,7,300]
        gra = [grad.cpu().numpy()]
    elif args.target_model == 'bert':
        # input_x = ([s1], [s2])
        # output = predictor.forward_tt(input_x)
        # output = output[0][ori_label]
        # output.backward()
        # for name, parms in predictor.model.named_parameters():
        #     if name == 'model.embeddings.word_embeddings.weight':  # 梯度全为0，不知为啥
        #         # print(name, parms.requires_grad)
        #         # print(parms.grad.shape)
        #         gra = parms.grad[s2, :]  # [30522, 768] --> [8, 768] (s2: list of len 8)
        #         gra = torch.unsqueeze(gra, 0)  # [1, 8, 768]
        #         try:
        #             gra = [gra.cpu().numpy()]
        #         except:
        #             print(output, true_label)
        #             print(gra)
        #             exit(0)
        #         break
        gra = torch.zeros(1, len(s2), 768)
        gra = [gra.cpu().numpy()]

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
            adv_tex = s2.copy()
            for i, Pos in enumerate(A_combin):
                adv_tex[Pos['pos']]=text_syns[Pos['pos']][A_sub[i]]
            adv_batch.append(adv_tex)

        s1s = [s1] * len(adv_batch)
        adv_probs = predictor.pred(s1s, adv_batch)
        adv_label = torch.tensor(np.argmax(adv_probs, axis=1))


        Re=torch.sum(adv_label != ori_label)

        if Re>0:
            End_time=time.time()
            #print("Found an adversarial example. Time: %.2f" % (End_time-Begin_time),file=outfile)
            return 1

    End_time = time.time()
    #print("Certificated Robustness. Time: %.2f" % (End_time - Begin_time),file=outfile)
    return 0



def increamental_for_r_1_4():         #增量式处理r=1~4的情况（找对抗样本和证明鲁棒）
    # 1. 加载训练好的模型
    predictor = load_model()

    # 2. 加载对抗结果，获得对抗样本
    pos_list = ['NNS', 'NNPS', 'NN','NNP','JJR', 'JJS','JJ', 'RBR', 'RBS','RB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VB']

    outfilename='FindAdresult/'+'snli_'+args.target_model+'_'+'r1_4'
    if not os.path.exists('FindAdresult'):
        os.makedirs('FindAdresult')
    f = open(outfilename, 'w')         #out file
    #f=None
    text_i=0
    #for text_index in range(len(texts)):
    for text_index in range(1000): # len(test_s2)
        text_i+=1
        # if text_i!=7:
        #     continue
        s1, s2, true_label = test_s1[text_index], test_s2[text_index], test['label'][text_index]

        # ========预测==========
        s2_1 = s2.copy()
        ori_label = np.argmax(predictor.pred([s1], [s2])[0])

        if ori_label != true_label:
            #print("Predict false")
            continue

        text_syns = [[t] for t in s2]  # 保存该文本各个位置同义词（包含自己。）
        pertub_psts = []  # 保存该文本的所有可替换位置
        pos_tag = pos_tags[text_index]
        for i in range(len(s2)):
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
            neigbhours = word_candidate[s2[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) == 0:
                continue
            pertub_psts.append(i)
            text_syns[i] += neigbhours
        #if len(ori_text)>=100 and len(ori_text)<=256:
        if len(s2) <= 256:
            print('text id:{:d}'.format(text_i))
            print('text id:{:d}'.format(text_i),file=f)
            print("r={:d}".format(1))
            print("r={:d}".format(1), file=f)
            Start_time=time.time()
            canFindAdv,position_conf_score=exhausted_search_r1(s1, s2, true_label, text_syns, pertub_psts, 1, predictor)
            End_time=time.time()
            if canFindAdv == 1:
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time))
                print("Found an adversarial example. Time: {:.2f}".format(End_time - Start_time), file=f)
                continue  # 处理下一条
            else:
                print("Certified Robustness. Time: %.2f" % (End_time - Start_time))
                print("Certified Robustness. Time: %.2f" % (End_time - Start_time), file=f)

            flag = 0
            for r in range(2,5):
                if (len(pertub_psts)<r):
                    flag = 1
                    break
                print("r={:d}".format(r))
                print("r={:d}".format(r),file=f)
                Start_time = time.time()
                canFindAdv, position_conf_score = exhausted_search_S(s1, s2, true_label, text_syns, pertub_psts,
                                                                     r,
                                                                     predictor, position_conf_score)
                End_time=time.time()
                if canFindAdv == 1:
                    print("Found an adversarial example. Time: %.2f" % (End_time - Start_time), file=f)
                    break
                else:
                    print("Certificated Robustness. Time: %.2f" % (End_time - Start_time),file=f)
            if flag:
                continue



    f.close()
    return


if __name__ == "__main__":
    #serch_replace_toRobustnees()
    #serch_least_replace()
    increamental_for_r_1_4()
