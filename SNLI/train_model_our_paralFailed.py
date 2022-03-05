import argparse
import numpy as np
import criteria
import pickle
import time
import tensorflow_hub as hub
import tensorflow as tf
from torch import nn, optim
from model_nli import Model as bdlstm_model
from encap_snli_bert import Model as bert_model
import torch
from config import args
import os
from gen_pos_tag import pos_tagger
from data import get_batch
import torch.nn as nn
from torch.autograd import Variable
from train_classifier import gen_sample_multiTexts
from torch.utils.data import DataLoader,TensorDataset
import sys

"""读取对抗样本"""
def read_adv_example():
    with open('/pub/data/huangpei/TextFooler/adv_results/train_set/org/tf_%s_%s_success.pkl' % (args.task, args.target_model), 'rb') as fp:
        input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fp)
        ori_s1 = [s[0] for s in input_list]
        adv_s2 = [adv.split(' ') for adv in output_list]
        return ori_s1, adv_s2, true_label_list

def train_model(args, epoch, model, optimizer,train_data, test_data, best_test, save_path):
    for e in range(epoch):
        trainepoch(e, model, optimizer, train_data)
        testacc=model.module.eval_model(test_data['s1'], test_data['s2'], test_data['label'], kind='f')

        if save_path:
            torch.save(model.module.nli_net.state_dict(), save_path)
            print('save model when test acc=', best_test)
    return 0

def train_model_adv(epoch, model, optimizer,train_data,adv_data,best_test, save_path):
    train_data['s1']=train_data['s1']+adv_data['s1']
    train_data['s2']=train_data['s2']+adv_data['s2']
    train_data['label']=train_data['label'].tolist()+adv_data['label']

    train_dataloader = transform_text(train_data)

    for e in range(epoch):
        trainepoch(e, model, optimizer, train_dataloader)
        testacc=model.eval_model(args.test_s1, args.test_s2, args.test_labels)
        if save_path:
            torch.save(model.nli_net.state_dict(), save_path)
            print('save model when test acc=', testacc)
    return 0

def train_model2(args, epoch, model, optimizer,train_data,test_data, best_test, save_path):

    adv_s1 = []
    adv_s2 = []
    adv_y=[]
    sample_size=100
    for step in range(0, len(train_data['s2']), args.batch_size):
        train_s1 = train_data['s1'][step:min(step + args.batch_size, len(train_data['s2']))]
        train_s2=train_data['s2'][step:min(step + args.batch_size, len(train_data['s2']))]
        train_label=train_data['label'][step:min(step + args.batch_size, len(train_data['s2']))]
        prob=model.module.text_pred_org(None, (train_s1,train_s2))
        O_result=torch.eq(torch.argmax(prob, dim=1), torch.Tensor(train_label).long().cuda())

        Samples_x=gen_sample_multiTexts(args, None, train_s2, sample_size,1)   #sample_size for each point
        # print("gen_sample_multiTexts time used:", use_time)

        s1s=[s for s in train_s1 for i in range(sample_size)] #每个s1复制sample_size 次
        Samples_y = [l for l in train_label for i in range(sample_size)]
        Sample_probs = model.module.text_pred_org(None, (s1s,Samples_x))

        S_result=torch.eq(torch.argmax(Sample_probs, dim=1), torch.Tensor(Samples_y).long().cuda()).view(len(train_data['s2']),sample_size)
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
        trainepoch(e, model, optimizer, train_data)
        testacc=model.module.eval_model(test_data['s1'], test_data['s2'], test_data['label'], kind='f')
        if save_path:
            torch.save(model.module.nli_net.state_dict(), save_path)
            print('save model when test acc=', best_test)
    return 0

def transform_text(train_data):
    # shuffle the data
    permutation = np.random.permutation(len(train_data['s1']))
    s1 = np.array(train_data['s1'])[permutation]
    s2 = np.array(train_data['s2'])[permutation]
    target = np.array(train_data['label'])[permutation]
    s1_len = np.array([len(x) for x in s1])
    s2_len = np.array([len(x) for x in s2])
    s1_ids, s2_ids = [],[]
    for a_s1, a_s2 in zip(s1, s2):
        # str转id
        a_s1_id = [args.full_dict[w] for w in a_s1]
        a_s2_id = [args.full_dict[w] for w in a_s2]
        # pad
        if len(a_s1_id) < args.max_seq_length:
            a_s1_id.extend([0]*(args.max_seq_length-len(a_s1_id)))
        else:
            a_s1_id = a_s1_id[:args.max_seq_length]
        if len(a_s2_id) < args.max_seq_length:
            a_s2_id.extend([0]*(args.max_seq_length-len(a_s2_id)))
        else:
            a_s2_id = a_s2_id[:args.max_seq_length]
        s1_ids.append(a_s1_id)
        s2_ids.append(a_s2_id)
    # 转tensor
    s1s = torch.tensor(s1_ids, dtype=torch.long)
    s2s = torch.tensor(s2_ids, dtype=torch.long)
    labels = torch.tensor(target, dtype=torch.long)
    s1_lens = torch.tensor(s1_len, dtype=torch.long)
    s2_lens = torch.tensor(s2_len, dtype=torch.long)
    # get dataloader
    train_dataset = TensorDataset(s1s,s2s, labels,s1_lens,s2_lens)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) # tiz
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    return train_dataloader


def trainepoch(epoch,model,optimizer,train_dataloader):
    print('\nTRAINING : Epoch ' + str(epoch))
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    model.nli_net.train()
    loss_fn = nn.CrossEntropyLoss()
    print('Num of batch in an epoch:', len(train_dataloader))
    for s1, s2, labels ,s1_len, s2_len, in train_dataloader:
        s_time = time.clock()
        s1_batch=s1.cuda()
        s2_batch=s2.cuda()
        tgt_batch=labels.cuda()
        # id转word vec
        s1_batch, _ = get_batch(s1_batch, model.word_vec, model.word_emb_dim)
        s2_batch, _ = get_batch(s2_batch, model.word_vec, model.word_emb_dim)   #word_emb_dim=300
        k = s1_batch.size(1)  # actual batch size
        s1_batch = s1_batch.cuda()
        s2_batch = s2_batch.cuda()
        # model forward
        output = model.nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()
        # assert len(pred) == len(s1[stidx:stidx + args.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / model.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in model.nli_net.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    p.grad.data.div_(k)  # divide by the actual batch size
                    total_norm += (p.grad.data.norm() ** 2).item()
        total_norm = np.sqrt(total_norm)

        if total_norm > 5.0:
            shrink_factor = 5.0 / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append(' loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            round(np.mean(all_costs), 2),
                            int(len(all_costs) * args.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            100.*float(correct)/(s1_batch.size[0])))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
        use_time = time.clock() - s_time
        print('Time of an batch: ', use_time)
    train_acc = 100 * float(correct)/len(s1)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def main():

    # Load data
    train_s1,train_s2,train_labels = args.train_s1, args.train_s2,args.train_labels
    adv_s1, adv_s2, adv_labels = read_adv_example()
    train_data = {'s1': train_s1, 's2': train_s2, 'label': train_labels}
    adv_data = {'s1': adv_s1, 's2': adv_s2, 'label': adv_labels}

    # Load model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = bdlstm_model(args)
        print(args.target_model)
        optimizer = optim.SGD(model.nli_net.parameters(), lr=2e-5)
        print("Model built!")
    testacc = model.eval_model(args.test_s1, args.test_s2, args.test_labels)
    print("befor training test acc:{:f}".format(testacc))

    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    model = model.cuda()
    model.nli_net = nn.DataParallel(model.nli_net)
    model.nli_net = model.nli_net.cuda()

    epoch =100
    best_acc = 0.0
    train_model_adv(epoch, model, optimizer, train_data, adv_data, best_acc, args.save_path)


    return 0


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main()