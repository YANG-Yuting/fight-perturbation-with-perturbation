import argparse
import numpy as np
import criteria
import pickle
import time
import tensorflow_hub as hub
import tensorflow as tf

from model_nli import Model as bdlstm_model
from encap_snli_bert import Model as bert_model
import torch
from config import args
import os
from gen_pos_tag import pos_tagger
from data import get_batch
from torch import nn, optim
from torch.autograd import Variable
from train_classifier import gen_sample_multiTexts
import sys

from SNLI_BERT import adjustBatchInputLen

"""读取对抗样本"""
def read_adv_example():
    with open('/pub/data/huangpei/TextFooler/adv_results/train_set/org/tf_%s_%s_success.pkl' % (args.task, args.target_model), 'rb') as fp:
        input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fp)
        ori_s1 = [s[0] for s in input_list]
        adv_s2 = [adv.split(' ') for adv in output_list]
        return ori_s1, adv_s2, true_label_list

def creat_batchs(model,T_data,batch_size):  #just for train

    S1=T_data['s1']
    S2 = T_data['s2']
    L=T_data['label']
    out_batches=[]
    for step in range(0,len(S1),batch_size):
        s1=S1[step:min(step+batch_size,len(S1))]
        s2=S2[step:min(step+batch_size,len(S2))]
        l=L[step:min(step+batch_size,len(L))]

        input_x=(s1,s2)
        assert len(input_x[0]) == len(input_x[1]), "premise and hypothesis should share the same batch lens!"
        num_instance = len(input_x[0])
        batch = dict()
        batch["inputs"] = []
        batch["labels"] = torch.LongTensor(l).view((len(l),))
        for i in range(len(input_x[0])):
            tokens = list()
            tokens.append(model.tokenizer.cls_token)
            for k in [0, 1]:
                add_sep = False
                if k == 0:
                    add_sep = True

                input_x[k][i] = [w.lower() for w in input_x[k][i]]  ###############
                tokens.extend(input_x[k][i])

                if add_sep:
                    tokens.append("[SEP]")
            tokens = model.tokenizer.convert_tokens_to_ids(tokens)
            batch["inputs"].append(tokens)
        adjustBatchInputLen(batch)
        end_id = model.tokenizer.convert_tokens_to_ids("[SEP]")
        for i in range(len(input_x[0])):
            tokens = batch["inputs"][i]
            tokens.append(end_id)
        batch["inputs"] = torch.stack([torch.LongTensor(x) for x in batch['inputs']])
        out_batches.append(batch)
    return out_batches



def train_model(args, epoch, model, optimizer,train_data, test_data, best_test, save_path):
    for e in range(epoch):
        loss=trainepoch(epoch, model, optimizer, train_data)
        testacc=model.eval_model(test_data['s1'], test_data['s2'], test_data['label'], kind='f')
        print("loss:{:f}   Acc:{:f}".format(loss,testacc))
        if save_path:
            torch.save(model.state_dict(), save_path)
            print('save model when test acc=', testacc)
    return 0

def train_model_adv(epoch, model, optimizer,train_data,adv_data, best_test, save_path):
    train_data['s1']=train_data['s1']+adv_data['s1']
    train_data['s2']=train_data['s2']+adv_data['s2']
    train_data['label']=train_data['label'].tolist()+adv_data['label']
    testacc = model.eval_model(args.test_s1, args.test_s2, args.test_labels)
    print('Befor train acc:{:f}'.format(testacc))

    for e in range(epoch):
        loss=trainepoch(e, model, optimizer, train_data)
        testacc=model.eval_model(args.test_s1, args.test_s2, args.test_labels)
        print("loss:{:f}   Acc:{:f}".format(loss, testacc))
        if save_path:
            torch.save(model.model.state_dict(), save_path) ###############
            print('save model when test acc=', testacc)
    return 0

def train_model2(args, epoch, model, optimizer,train_data, best_test, save_path):

    adv_s1 = []
    adv_s2 = []
    adv_y=[]
    sample_size=100
    for step in range(0, len(train_data['s2']), args.batch_size):
        train_s1 = train_data['s1'][step:min(step + args.batch_size, len(train_data['s2']))]
        train_s2=train_data['s2'][step:min(step + args.batch_size, len(train_data['s2']))]
        train_label=train_data['label'][step:min(step + args.batch_size, len(train_data['s2']))]
        prob=model.text_pred_org(None, (train_s1,train_s2))
        O_result=torch.eq(torch.argmax(prob, dim=1), torch.Tensor(train_label).long().cuda())

        Samples_x=gen_sample_multiTexts(args, None, train_s2, sample_size,1)   #sample_size for each point
        # print("gen_sample_multiTexts time used:", use_time)

        s1s=[s for s in train_s1 for i in range(sample_size)] #每个s1复制sample_size 次
        Samples_y = [l for l in train_label for i in range(sample_size)]
        Sample_probs = model.text_pred_org(None, (s1s,Samples_x))

        S_result=torch.eq(torch.argmax(Sample_probs, dim=1), torch.Tensor(Samples_y).long().cuda()).view(len(train_s2),sample_size)
        R_score=torch.sum(S_result,dim=1).view(-1).float()/float(sample_size) #每个训练点的鲁棒打分

        for i in range(R_score.size()[0]):
            if R_score[i] < 2.0 / 3.0 and O_result[i] == 1.0:
                adv_count = 1
                for j in range(sample_size):
                    if S_result[i][j].data != train_label[i]:
                        adv_s1.append(s1s[i * sample_size + j])
                        adv_s2.append(Samples_x[i * sample_size + j])
                        adv_y.append(train_label[i])
                        adv_count = adv_count - 1
                        if adv_count == 0:
                            break
            # print('filt_adv')
    print("adv_batch length:{:d}".format(len(adv_s2)))

    train_data['s1']=train_data['s1']+adv_s1
    train_data['s2']=train_data['s2']+adv_s2
    train_data['label']=train_data['label'].tolist()+adv_y
    for e in range(epoch):
        loss=trainepoch(e, model, optimizer, train_data)
        testacc=model.eval_model(args.test_s1, args.test_s2, args.test_labels)
        print("loss:{:f}   Acc:{:f}".format(loss, testacc))
        if save_path:
            torch.save(model.model.state_dict(), save_path)
            print('save model when test acc=', testacc)
    return 0




def trainepoch(epoch,model,optimizer,train_data):
    print('TRAINING : Epoch ' + str(epoch))

    acc = 0
    local_loss = 0
    it=0
    trainer = model.model
    trainer.train()
    criterion = nn.CrossEntropyLoss()
    train_batches=creat_batchs(model,train_data,args.batch_size)
    for batch in train_batches:
        it+=1
        if it%100==0:
            print(it,end=' ')
        optimizer.zero_grad()
        _, logits = trainer(batch)
        loss=criterion(logits[:,[1,0,2]],batch['labels'])
        acc += sum(torch.argmax(logits[:,[1,0,2]], dim=-1) == batch['labels']).float()
        local_loss += loss.item()
        loss.backward()
        optimizer.step()

    return local_loss




def main():
    # Load data
    train_s1,train_s2,train_labels = args.train_s1, args.train_s2,args.train_labels
    adv_s1, adv_s2, adv_labels = read_adv_example()
    train_data = {'s1': train_s1, 's2': train_s2, 'label': train_labels}
    adv_data = {'s1': adv_s1, 's2': adv_s2, 'label': adv_labels}

    model = bert_model(args)  # 模型初始化的时候回自动加载断点等
    model=model.cuda()
    print(args.target_model)
    if args.mode == 'eval':  # tiz
        print('Eval...')
    else:
        print('Train...')
        optimizer = optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)
        epoch = 100
        best_acc = 0.0
        print("run train_model_adv")
        train_model_adv(epoch, model, optimizer, train_data, adv_data, best_acc, args.save_path)
        # print("run train_model2..")
        # train_model2(args, epoch, model, optimizer, train_data, best_acc, args.save_path)
    return 0


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main()