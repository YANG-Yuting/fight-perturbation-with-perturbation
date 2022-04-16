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
import math
from numpy.random import choice

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import datetime

import dataloader
import modules
from config import args

from nltk.tag import StanfordPOSTagger
jar = args.data_path + 'stanford-postagger-full-2020-11-17/stanford-postagger.jar'
model = args.data_path + 'stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

np.random.seed(8848)   #8848


class Model(nn.Module):
    def __init__(self, args, max_seq_length, embedding, hidden_size=150, depth=1, dropout=0.3, cnn=False, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(embs = dataloader.load_embedding(embedding), fix_emb=True)
        self.word2id = self.emb_layer.word2id
        self.max_seq_length = max_seq_length
        self.tf_vocabulary=pickle.load(open(args.data_path + '%s/tf_vocabulary.pkl' % args.task, "rb"))
        if cnn:
            self.encoder = modules.CNN_Text(self.emb_layer.n_d, widths = [3,4,5],filters=hidden_size)
            d_out = 3*hidden_size
        else:
            self.encoder = nn.LSTM(self.emb_layer.n_d,hidden_size//2,depth,dropout = dropout,bidirectional=True)
            d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)

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
            output = torch.max(output, dim=0)[0] #.squeeze()

        output = self.drop(output)
        return self.out(output)

    def text_pred(self):
        """
        org: original predictor
        Enhance: Enhancement version
        adv:adv trained
        SAFER：SAFER
        FGWS：FGWS
        """
        call_func = getattr(self, 'text_pred_%s' % self.args.kind, 'Didn\'t find your text_pred_*.')
        return call_func



    def text_pred_org(self, orig_text=None, text=None, batch_size=128):
        batches_x = dataloader.create_batches_x(text, self.max_seq_length, batch_size, self.word2id)
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                output = self.forward(x)
                outs.append(F.softmax(output, dim=-1))

        return torch.cat(outs, dim=0)

    def text_pred_adv(self, orig_text=None, text=None, batch_size=128):
        batches_x = dataloader.create_batches_x(text, self.max_seq_length, batch_size, self.word2id)
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                output = self.forward(x)
                outs.append(F.softmax(output, dim=-1))

        return torch.cat(outs, dim=0)

    def text_pred_Enhance(self, orig_texts, text, sample_num=256, batch_size=128):

        probs_boost_all = []
        perturbed_texts = perturb_texts(self.args, orig_texts, text, self.tf_vocabulary, change_ratio=0.25)
        #perturbed_texts=text
        Samples_x = gen_sample_multiTexts(self.args, orig_texts, perturbed_texts, sample_num, change_ratio=0.25)
        Sample_probs = self.text_pred_org(None, Samples_x, batch_size)
        lable_mum=Sample_probs.size()[-1]
        Sample_probs=Sample_probs.view(len(text),sample_num,lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l),dim=1)
            prob = num.float() / float(sample_num)
            probs_boost.append(prob.view(len(text),1))

        probs_boost_all=torch.cat(probs_boost,dim=1)

        return probs_boost_all


    def text_pred_SAFER(self,orig_texts, text, sample_num=256, batch_size=128):
        probs_boost_all = []
        Samples_x = gen_sample_multiTexts(self.args, orig_texts, text, sample_num, change_ratio=1)
        Sample_probs = self.text_pred_org(None, Samples_x, batch_size)
        lable_mum=Sample_probs.size()[-1]
        Sample_probs=Sample_probs.view(len(text),sample_num,lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
            prob = num.float() / float(sample_num)
            probs_boost.append(prob.view(len(text), 1))
        probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_FGWS(self,orig_texts, text, batch_size=128):
        gamma1=0.5
        perturbed_texts=perturb_FGWS(self.args, orig_texts, text, self.tf_vocabulary)
        pre_prob=self.text_pred_org(None, perturbed_texts, batch_size)
        ori_prob=self.text_pred_org(None, text, batch_size)
        lable=torch.argmax(ori_prob, dim=1)
        index=torch.arange(len(text)).cuda()
        D=ori_prob.gather(1, lable.view(-1, 1))-pre_prob.gather(1, lable.view(-1, 1))-gamma1
        probs_boost_all = torch.where(D > 0, pre_prob, ori_prob)
        # D=D.view(-1)
        # probs_boost_all=torch.ones_like(ori_prob)
        # probs_boost_all.index_put_((index,lable),0.5 - D)
        # probs_boost_all.index_put_((index,1-lable), 0.5+D)
        return probs_boost_all

    # def eval_model(self,input_x, input_y,):
    #     predictor = self.text_pred()
    #     ori_probs = predictor(input_x, input_x)
    #     ori_labels = torch.argmax(ori_probs, dim=1)
    #     input_y = torch.tensor(input_y).long().cuda()
    #     correct = ori_labels.eq(input_y).cpu().sum()
    #
    #     return correct.item() / float(len(input_x))

        # with torch.no_grad():
        #     for step in range(0, data_size, batch_size):
        #         input_x1 = input_x[step:min(step + batch_size, data_size)]  # 取一个batch
        #         input_y1 = input_y[step:min(step + batch_size, data_size)]
        #         output = predictor(input_x1, input_x1)
        #         pred = torch.argmax(output, dim=1)
        #         correct += torch.sum(torch.eq(pred, torch.LongTensor(input_y1).cuda()))
        #     acc = (correct.cpu().numpy()) / float(data_size)
        # return acc

    # def text_pred(self, orig_text, text, batch_size=128):
    #     batches_x = dataloader.create_batches_x(text, self.max_seq_length, batch_size, self.word2id)
    #     outs = []
    #     with torch.no_grad():
    #         for x in batches_x:
    #             x = Variable(x)
    #             output = self.forward(x)
    #             outs.append(F.softmax(output, dim=-1))
    #
    #     return torch.cat(outs, dim=0)



def perturb_texts(args, orig_texts=None, texts=None, tf_vocabulary=None, change_ratio = 1):
    select_sents = []
    if orig_texts is None:
        orig_texts=[None for i in range(len(texts))]
    for orig_text,text_str in zip(orig_texts,texts):
        # get candidates
        text_str = [word for word in text_str if word != '\x85']
        text_str = [word.replace('\x85', '') for word in text_str]
        if ' '.join(text_str) in args.candidate_bags.keys():
            candidate_bag = args.candidate_bags[' '.join(text_str)]
        else:
            if orig_text:
                orig_text = [word for word in orig_text if word != '\x85']
                orig_text = [word.replace('\x85', '') for word in orig_text]
                pos_tag = args.pos_tags[' '.join(orig_text)]  # for test set
            else:
                pos_tag = pos_tagger.tag(text_str)  # 0.45s

            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            candidate_bag = {}
            for j in range(len(text_ids)):
                word = text_ids[j]
                pos = pos_tag[j][1]
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list and word < len(args.word_candidate):
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours]

        replace_text = text_str.copy()
        for i in range(len(text_str) - 1):
            candi = candidate_bag[text_str[i]]
            if len(candi) == 1:
                continue
            else:
                eps=np.finfo(np.float64).eps
                sum_freq1=0.0
                sum_freq2 = 0.0
                max_freq = 0.0
                best_replace = replace_text[i]
                Ori_freq2 = 0.0
                Ori_freq1 = 0.0
                freq_list2 = []
                freq_list1=[]
                for c in candi:
                    freq1=0.0
                    freq2=0.0
                    two_gram = c + ' ' + text_str[i + 1]
                    if two_gram in tf_vocabulary.keys():
                        freq2=tf_vocabulary[two_gram]
                        if c==text_str[i]:
                            Ori_freq2 = tf_vocabulary[two_gram]+eps
                    if c in tf_vocabulary.keys():
                        freq1=tf_vocabulary[c]
                        if c==text_str[i]:
                            Ori_freq1 = tf_vocabulary[c]+eps
                    freq_list2.append(freq2+eps)
                    freq_list1.append(freq1+eps)

                sum_freq1=sum(freq_list1)
                sum_freq2=sum(freq_list2)
                lamda2=0.5     #0.5
                lamda1=0.5     #0.5

                Ori_freq2=Ori_freq2/sum_freq2
                Ori_freq1=Ori_freq1/sum_freq1
                Ori_freq=lamda2*Ori_freq2+lamda1*Ori_freq1

                for freq2, freq1,c in zip(freq_list2,freq_list1,candi):
                    freq2=freq2/sum_freq2
                    freq1=freq1/sum_freq1

                    freq=lamda2*freq2+lamda1*freq1
                    #sum_freq+=freq
                    #print(freq,end=' ')
                    if freq > max_freq:
                        max_freq = freq
                        best_replace = c

                r_seed = random.uniform(0, 1)

                #Ori_freq=0
                if (max_freq-Ori_freq) > r_seed:
                    replace_text[i] = best_replace

        select_sents.append(replace_text)

    return select_sents

# for FGWS #
def perturb_FGWS(args, orig_texts=None, texts=None, tf_vocabulary=None):
    select_sents = []
    if orig_texts is None:
        orig_texts=[None for i in range(len(texts))]
    for orig_text,text_str in zip(orig_texts,texts):
        if ' '.join(text_str) in args.candidate_bags.keys(): # if seen（train and test）
            candidate_bag = args.candidate_bags[' '.join(text_str)]
        else:
            if orig_text:
                pos_tag = args.pos_tags[' '.join(orig_text)]
            else:
                pos_tag = pos_tagger.tag(text_str)

            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            candidate_bag = {}
            for j in range(len(text_ids)):
                word = text_ids[j]
                pos = pos_tag[j][1]
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list and word < len(args.word_candidate):
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours]

        replace_text = text_str.copy()
        for i in range(len(text_str) - 1):
            candi = candidate_bag[text_str[i]]
            # 若候选集只有自己
            if len(candi) == 1:
                continue
            else:
                freq1=0
                bestc=replace_text[i]
                for c in candi:
                    if c in tf_vocabulary.keys():
                        if tf_vocabulary[c]>freq1:
                            freq1=tf_vocabulary[c]
                            bestc=c
                replace_text[i]=bestc
        select_sents.append(replace_text)
    return select_sents

"""sample"""
def gen_sample_multiTexts(args, orig_texts=None, texts=None, sample_num=64, change_ratio = 1):
    Finded_num=0
    all_sample_texts = []  # return: list of list of str, including all samples for each input
    if orig_texts is None:
        orig_texts=[None for i in range(len(texts))]
    for orig_text,text_str in zip(orig_texts,texts):
        text_str = [word for word in text_str if word != '\x85']
        text_str = [word.replace('\x85', '') for word in text_str]
        if ' '.join(text_str) in args.candidate_bags.keys():  # seen data (from train or test dataset)
            Finded_num+=1
            candidate_bag = args.candidate_bags[' '.join(text_str)]

            sample_texts=[]
            for ii in range(len(text_str)):
                word_str = text_str[ii]
                r_seed = np.random.rand(sample_num)
                n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                n[np.where(r_seed>change_ratio)]=word_str
                sample_texts.append(n)
            sample_texts = np.array(sample_texts).T.tolist()

        else:  # unseen data
            if orig_text:
                orig_text = [word for word in orig_text if word != '\x85']
                orig_text = [word.replace('\x85', '') for word in orig_text]
                pos_tag = args.pos_tags[' '.join(orig_text)]
            else:
                print('pos tagging for unseen data')
                pos_tag = pos_tagger.tag(text_str)

            # start_time = time.clock()
            text_ids = []
            for word_str in text_str:
                if word_str in args.full_dict.keys():
                    text_ids.append(args.full_dict[word_str])  # id
                else:
                    text_ids.append(word_str)  # str

            candidate_bag = {}
            for j in range(len(text_ids)):
                word = text_ids[j]
                pos = pos_tag[j][1]
                neigbhours = [word]
                if isinstance(word, int) and pos in args.pos_list and word < len(args.word_candidate):
                    if pos.startswith('JJ'):
                        pos = 'adj'
                    elif pos.startswith('NN'):
                        pos = 'noun'
                    elif pos.startswith('RB'):
                        pos = 'adv'
                    elif pos.startswith('VB'):
                        pos = 'verb'
                    neigbhours.extend(args.word_candidate[word][pos])
                candidate_bag[text_str[j]] = [args.inv_full_dict[i] if isinstance(i, int) else i for i in neigbhours]

            sample_texts = []
            for ii in range(len(text_str)):
                word_str = text_str[ii]
                r_seed = np.random.rand(sample_num)
                n = choice(candidate_bag[word_str], size=sample_num, replace=True)
                n[np.where(r_seed>change_ratio)]=word_str
                sample_texts.append(n)
            sample_texts = np.array(sample_texts).T.tolist()

            # use_time = (time.clock() - start_time)
            # print("Time for the left: ", use_time)

        all_sample_texts.extend(sample_texts)
    #print("{:d}/{:d} texts are finded".format(Finded_num,len(texts)))
    return all_sample_texts


"""read adversarial examples"""
def read_adv_example():
    with open('adv_results/train_set/org/tf_%s_%s_success.pkl' % (args.task, args.target_model), 'rb') as fp:
        input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fp)
        output_list = [adv.split(' ') for adv in output_list]
        return output_list, true_label_list


def eval_model(model, input_x, input_y):
    model.eval()
    print('Evaluate Model:{:s}'.format(args.kind))
    # N = len(valid_x)
    # criterion = nn.CrossEntropyLoss()
    batch_size=128
    correct = 0.0
    data_size = len(input_y)
    predictor = model.text_pred()
    with torch.no_grad():
        for step in range(0, data_size, batch_size):
            input_x1 = input_x[step:min(step + batch_size, data_size)]
            input_y1 = input_y[step:min(step + batch_size, data_size)]
            output = predictor(input_x1, input_x1)
            pred = torch.argmax(output, dim=1)
            correct+=torch.sum(torch.eq(pred,torch.LongTensor(input_y1).cuda()))
        acc=(correct.cpu().numpy())/float(data_size)
    return acc

# train_x, train_y：划分batch
# test_x，test_y：原始数据
def train_model(args, epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):
    model.train()
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    test_acc = eval_model(model, test_x, test_y)

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
    test_acc = eval_model(model, test_x, test_y)
    print('before train: acc={:.6f}'.format(test_acc))
    cnt = 0
    batch_size=128
    sample_size=100
    niter = epoch * len(train_x)//batch_size

    for step in range(0,len(train_x),batch_size):
        #print('step={:d}'.format(step//batch_size))
        train_x1=train_x[step:min(step+batch_size,len(train_x))]
        train_y1=train_y[step:min(step+batch_size,len(train_x))]

        niter += 1
        cnt += 1
        start_time = time.clock()
        Samples_x=gen_sample_multiTexts(args, None, train_x1, sample_size)   #sample_size for each point
        use_time = (time.clock() - start_time)
        Samples_y=[l for l in train_y1 for i in range(sample_size)]
        #print('text_pred_org')
        Sample_probs=model.text_pred_org(None, Samples_x)
        S_result=torch.eq(torch.argmax(Sample_probs, dim=1), torch.Tensor(Samples_y).cuda()).view(len(train_y1),sample_size)
        R_score=torch.sum(S_result,dim=1).view(-1)/float(sample_size)
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
        if len(adv_batch_x)!=0:
            #print('adv_batch_x len:{:d}'.format(len(adv_batch_x)))
            adv_x, adv_y = dataloader.create_batches(adv_batch_x, adv_batch_y, args.max_seq_length, batch_size,model.word2id, )

            adv_x = torch.cat(adv_x, dim=1)
            adv_y= torch.cat(adv_y, dim=0)
            model.train()
            model.zero_grad()
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     model = nn.DataParallel(model)
            #adv_loss=0
            adv_x, adv_y = Variable(adv_x), Variable(adv_y)
            output = model(adv_x)
            output = torch.reshape(output, (-1, args.nclasses))

            adv_loss =criterion(output, adv_y)
        else:
            adv_loss=0

        #print('finishid adv_loss={:.6f}'.format(adv_loss.item()))

        train_x1, train_y1 = dataloader.create_batches(train_x1, train_y1, args.max_seq_length, batch_size,
                                                     model.word2id, )

        train_x1 = torch.cat(train_x1, dim=1)
        train_y1= torch.cat(train_y1, dim=0)

        train_x1, train_y1 = Variable(train_x1), Variable(train_y1)
        output = model(train_x1)
        Norm_loss =criterion(output, train_y1)

        #print('finishid Norm_loss={:.6f}'.format(Norm_loss.item()))

        loss=0.5*Norm_loss+0.5*adv_loss
        #loss=Norm_loss
        loss.backward()
        optimizer.step()


    test_acc = eval_model(model, test_x, test_y)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(epoch, niter,optimizer.param_groups[0]['lr'],loss.item(),test_acc))

    if save_path:
        torch.save(model.state_dict(), save_path)
        print('save model when test acc=', best_test)
    return best_test

# our data argue method
def train_model2(args, epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):
    print('train:{:d}'.format(len(train_x)))
    criterion = nn.CrossEntropyLoss()
    test_acc = eval_model(model, test_x, test_y)
    print('before train: acc={:.6f}'.format(test_acc))
    cnt = 0
    batch_size=128
    sample_size=100
    niter = 0

    adv_batch_x = []
    adv_batch_y = []

    for step in range(0,len(train_x),batch_size):
        #print('step={:d}'.format(step//batch_size))
        train_x1=train_x[step:min(step+batch_size,len(train_x))] #取一个batch
        train_y1=train_y[step:min(step+batch_size,len(train_x))]
        #print('gen_sample_multiTexts')
        prob=model.text_pred_org(None, train_x1)
        O_result=torch.eq(torch.argmax(prob, dim=1), torch.Tensor(train_y1).long().cuda())
        Samples_x=gen_sample_multiTexts(args, None, train_x1, sample_size,1)   #sample_size for each point
        # print("gen_sample_multiTexts time used:", use_time)
        Samples_y=[l for l in train_y1 for i in range(sample_size)] #每个lable复制sample_size 次
        #print('text_pred_org')
        Sample_probs=model.text_pred_org(None, Samples_x)
        S_result=torch.eq(torch.argmax(Sample_probs, dim=1), torch.Tensor(Samples_y).long().cuda()).view(len(train_y1),sample_size)
        R_score=torch.sum(S_result,dim=1).view(-1).float()/float(sample_size) #每个训练点的鲁棒打分
        #print(R_score)

        for i in range(R_score.size()[0]):
            if R_score[i]<2.0/3.0 and O_result[i]==1.0:
                adv_count=1
                for j in range(sample_size):
                    if S_result[i][j].data!=train_y1[i]:
                        adv_batch_x.append(Samples_x[i*sample_size+j])
                        adv_batch_y.append(train_y1[i])
                        adv_count=adv_count-1
                        if adv_count==0:
                            break
            #print('filt_adv')
    print("adv_batch length:{:d}".format(len(adv_batch_x)))
    train_x=list(train_x)+adv_batch_x
    train_y=list(train_y)+adv_batch_y

    temp = list(zip(train_x, train_y))
    random.shuffle(temp)
    train_x[:], train_y[:] = zip(*temp)

    train_x1, train_y1 = dataloader.create_batches(train_x, train_y, args.max_seq_length, batch_size,
                                                   model.word2id, )


    for e in range(epoch):
        sum_loss=0
        model.train()
        model.zero_grad()
        for x,y in zip(train_x1,train_y1):
            niter += 1
            x, y = Variable(x), Variable(y)
            output = model(x)
            loss =criterion(output, y)
            loss.backward()
            optimizer.step()
            sum_loss+=float(loss.item())

        test_acc = eval_model(model, test_x, test_y)

        sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(e, niter,optimizer.param_groups[0]['lr'],sum_loss,test_acc))
        if save_path:
            torch.save(model.state_dict(), save_path)
            print('save model when test acc=', test_acc)
        lr_decay=1
        if lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= lr_decay
    return test_acc

def train_model_adv(args, epoch, model, optimizer,train_x, train_y,train_adv_x, trian_adv_y, test_x, test_y, best_test, save_path):

    print('train:{:d}'.format(len(train_x)))
    print('adv:{:d}'.format(len(train_adv_x)))
    test_acc = eval_model(model, test_x, test_y)
    print('before train: acc={:.6f}'.format(test_acc))


    train_x=list(train_x)+list(train_adv_x)
    train_y=list(train_y)+list(trian_adv_y)

    temp = list(zip(train_x, train_y))
    random.shuffle(temp)
    train_x[:], train_y[:] = zip(*temp)
    batch_size=128
    train_x1, train_y1 = dataloader.create_batches(train_x, train_y, args.max_seq_length, batch_size,
                                                   model.word2id, )

    niter=0
    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        sum_loss = 0
        model.train()
        model.zero_grad()
        for x, y in zip(train_x1, train_y1):
            niter += 1
            x, y = Variable(x), Variable(y)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())

        test_acc = eval_model(model, test_x, test_y)

        sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(e, niter,
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr'], sum_loss,
                                                                                                 test_acc))
        if save_path:
            torch.save(model.state_dict(), save_path)
            print('save model when test acc=', test_acc)
        lr_decay = 1.0
        if lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= lr_decay
    return test_acc



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
    train_x = args.datasets.train_seqs2
    train_x = [[args.inv_full_dict[word] for word in text] for text in train_x]  # 网络输入是词语
    train_y = args.datasets.train_y
    test_x = args.datasets.test_seqs2
    test_x = [[args.inv_full_dict[word] for word in text] for text in test_x]  # 网络输入是词语
    test_y = args.datasets.test_y

    # if args.task == 'mr':
    #     # train_x, train_y = dataloader.read_corpus('data/adversary_training_corpora/mr/train.txt', clean = False, FAKE = False, shuffle = True)
    #     # test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test.txt', clean = False, FAKE = False, shuffle = False) # 为了观察，暂时不shuffle
    #     train_x = args.datasets.train_seqs2
    #     train_x = [[args.inv_full_dict[word] for word in text] for text in train_x]  # 网络输入是词语
    #     train_y = args.datasets.train_y
    #     test_x = args.datasets.test_seqs2
    #     test_x = [[args.inv_full_dict[word] for word in text] for text in test_x]  # 网络输入是词语
    #     test_y = args.datasets.test_y
    # elif args.task == 'imdb':
    #     train_x, train_y = dataloader.read_corpus(os.path.join(args.data_path + 'imdb', 'train_tok.csv'), clean=False, FAKE=False, shuffle=True)
    #     test_x, test_y = dataloader.read_corpus(os.path.join(args.data_path + 'imdb', 'test_tok.csv'), clean=False, FAKE=False, shuffle=True)
    # elif args.task == 'fake':
    #     train_x, train_y = dataloader.read_corpus(args.data_path + '{}/train_tok.csv'.format(args.task), clean=False, FAKE=True, shuffle=True)
    #     # 关于fake的测试集
    #     # test_x, test_y = dataloader.read_corpus(args.data_path + '{}/test_tok.csv'.format(args.task), clean=False, FAKE=True, shuffle=True) # 原本是从文件中读取
    #     # test_y = [1-y for y in test_y] # tiz: 反转测试集标签试试 --> 更不对了
    #     train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=1) # tiz: 从训练集中获得测试集试试 --> 正常了
    # elif args.task == 'mr_adv':
    #    test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test_adv.txt', clean = False, FAKE = False, shuffle = True)


    print('Num of testset: %d' % (len(test_y)))

    print('Build model...')
    model = Model(args, args.max_seq_length, args.embedding, args.d, args.depth, args.dropout, args.cnn, args.nclasses).cuda() # tiz: 512 -->args.max_seq_length
    if args.target_model_path is not None:  # tiz
        print('Load pretrained model...')
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)

    if args.mode == 'eval':
        print('Eval...')
        # test_x, test_y = dataloader.create_batches(test_x, test_y, args.max_seq_length,args.batch_size, model.word2id, ) # tiz: 512 -->args.max_seq_length
        # test_acc0=eval_model(model, test_x, test_y)
        # print('Base classifier f acc:{:.6f}'.format(test_acc0))
        test_acc = eval_model(model, test_x, test_y)
        # test_acc = model.eval_model(test_x, test_y)
        print('Test acc: ', test_acc)
    else:
        print('Train...')
        train_acc = eval_model(model, train_x, train_y)
        print('Original train acc: ', train_acc)

        #train_x, train_y = dataloader.create_batches(train_x, train_y, args.max_seq_length, args.batch_size, model.word2id, ) # tiz: 512 -->args.max_seq_length
        need_grad = lambda x: x.requires_grad
        optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)

        best_test = 0
        # test_err = 1e+8
        #========我们的对抗训练法1====================
        # for epoch in range(args.max_epoch):
        #     best_test = train_model1(args, epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)
        #     if args.lr_decay > 0:
        #         optimizer.param_groups[0]['lr'] *= args.lr_decay
        # sys.stdout.write("test_err: {:.6f}\n".format(best_test))
        #=================================================

        #===========我们的对抗训练法2=============
        print('run train_model2')
        best_test = train_model2(args, 150, model, optimizer, train_x, train_y, test_x, test_y, best_test,args.save_path)
        #=================传统对抗训练(对比用)=============
        # print('run train_model_adv')
        # train_adv_x,  train_adv_y,= read_adv_example()
        # best_test = train_model_adv(args, 200, model, optimizer, train_x, train_y,train_adv_x,train_adv_y, test_x, test_y, best_test, args.save_path)
        #==================================


def get_test_label():
    test = pd.read_csv(args.data_path + 'fake/test.csv')['text'].tolist()
    print(pd.read_csv(args.data_path + 'fake/test.csv').isna().sum())
    labels = pd.read_csv(args.data_path + 'fake/submit.csv')['label'].tolist()
    with open(args.data_path + 'fake/test_tok.csv','w') as w:
        for i in range(len(test)):
            try:
                test[i] = test[i].replace('\n','')
                w.write(test[i] + ','+str(labels[i]+1) + '\n')
            except:
                print(test[i])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)
    # read_adv_example()

