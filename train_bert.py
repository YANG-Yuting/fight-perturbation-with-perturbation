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
import pickle

import dataloader
from config import args
from attack_classification import NLI_infer_BERT
from train_classifier import gen_sample_multiTexts

"""读取对抗样本"""
def read_adv_example():
    with open('/pub/data/huangpei/TextFooler/adv_results/train_set/org/tf_%s_%s_success.pkl' % (args.task, args.target_model), 'rb') as fp:
        input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fp)
        output_list = [adv.split(' ') for adv in output_list]
        return output_list, true_label_list


# train_x, train_y：划分batch
# test_x，test_y：原始数据
# 对于bert，划分batch和转id，是在forward里做的。forward的输入是所有数据-->改写成输入是一个batch
def train_model(epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):
    model.train()

    dataloader, _ = model.module.dataset.transform_text(train_x,train_y, batch_size=args.batch_size)

    niter=0
    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        sum_loss = 0
        # model.zero_grad()
        print('Batch num:', len(dataloader))
        for input_ids, input_mask, segment_ids, labels in dataloader: # 每个batch
            # start_time = time.clock()
            niter += 1
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            output = model(input_ids, input_mask, segment_ids)
            loss = criterion(output, labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())
            # use_time = (time.clock() - start_time)
            # print("Time for a batch: ", use_time)

        test_acc = model.module.eval_model(test_x, test_y)
        sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(e, niter,optimizer.param_groups[0]['lr'], sum_loss,test_acc))
        if save_path:
            torch.save(model.module.state_dict(), save_path)
            print('save model when test acc=', test_acc)
        lr_decay = 1.0
        if lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= lr_decay
    return test_acc


def train_model_adv(epoch, model, optimizer,train_x, train_y,train_adv_x,train_adv_y, test_x, test_y, best_test, save_path):

    print('train:{:d}'.format(len(train_x)))
    print('adv:{:d}'.format(len(train_adv_x)))
    test_acc = model.module.eval_model(test_x, test_y)
    # test_acc = model.eval_model(test_x, test_y)
    print('before train: acc={:.6f}'.format(test_acc))


    train_x=list(train_x)+list(train_adv_x)
    train_y=list(train_y)+list(train_adv_y)

    temp = list(zip(train_x, train_y))
    random.shuffle(temp)
    train_x[:], train_y[:] = zip(*temp)
    batch_size=128

    dataloader, _ = model.module.dataset.transform_text(train_x, train_y, batch_size=args.batch_size)
    # dataloader, _ = model.dataset.transform_text0(train_x, train_y, batch_size=args.batch_size)  ########3# 0 for unparal
    model.train()
    niter = 0
    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        sum_loss = 0
        # model.zero_grad()
        #print('Batch num:', len(dataloader))
        for input_ids, input_mask, segment_ids, labels in dataloader:  # 每个batch
            # start_time = time.clock()
            niter += 1
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            output = model(input_ids, input_mask, segment_ids)
            loss = criterion(output, labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())
            # use_time = (time.clock() - start_time)
            # print("Time for a batch: ", use_time)

        test_acc = model.module.eval_model(test_x, test_y)
        # test_acc = model.eval_model(test_x, test_y)
        sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(e, niter,
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr'], sum_loss,
                                                                                                 test_acc))
        if save_path:
            # torch.save(model.module.state_dict(), save_path)
            torch.save(model.module.model.state_dict(), save_path+'/pytorch_model.bin')
            model.module.model.config.to_json_file(save_path + '/bert_config.json')
            model.module.dataset.tokenizer.save_vocabulary(save_path)
            # model.model.save_pretrained(save_path)
            print('save model when test acc=', test_acc)
        lr_decay = 1.0
        if lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= lr_decay
    return test_acc


#def train_model1(epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):


def train_model2(epoch, model, optimizer,train_x, train_y, test_x, test_y, best_test, save_path):
    print('train:{:d}'.format(len(train_x)))
    # test_acc = model.module.eval_model(test_x, test_y)
    test_acc = model.eval_model(test_x, test_y)
    print('before train: acc={:.6f}'.format(test_acc))

    sample_size=100
    niter = 0

    adv_batch_x = []
    adv_batch_y = []

    #dataloader, _ = model.module.dataset.transform_text(train_x, train_y, batch_size=args.batch_size)

    args.batch_size=512
    for step in range(0,len(train_x),args.batch_size):
        print(step)
        train_x1=train_x[step:min(step+args.batch_size,len(train_x))] #取一个batch
        train_y1=train_y[step:min(step+args.batch_size,len(train_x))]

        # prob=model.module.text_pred_org(None, train_x1)
        prob=model.text_pred_org(None, train_x1)

        O_result = torch.eq(torch.argmax(prob, dim=1), torch.Tensor(train_y1).long().cuda())
        Samples_x = gen_sample_multiTexts(args, None, train_x1, sample_size, 1)  # sample_size for each point
        # print("gen_sample_multiTexts time used:", use_time)
        Samples_y = [l for l in train_y1 for i in range(sample_size)]  # 每个lable复制sample_size 次
        # print('text_pred_org')
        # Sample_probs = model.module.text_pred_org(None, Samples_x)
        Sample_probs = model.text_pred_org(None, Samples_x)
        S_result = torch.eq(torch.argmax(Sample_probs, dim=1), torch.Tensor(Samples_y).long().cuda()).view(
            len(train_y1), sample_size)
        R_score = torch.sum(S_result, dim=1).view(-1).float() / float(sample_size)  # 每个训练点的鲁棒打分

        for i in range(R_score.size()[0]):
            if R_score[i] < 2.0 / 3.0 and O_result[i] == 1.0:
                adv_count = 1
                for j in range(sample_size):
                    if S_result[i][j].data != train_y1[i]:
                        adv_batch_x.append(Samples_x[i * sample_size + j])
                        adv_batch_y.append(train_y1[i])
                        adv_count = adv_count - 1
                        if adv_count == 0:
                            break
            # print('filt_adv')
    print("adv_batch length:{:d}".format(len(adv_batch_x)))
    train_x = list(train_x) + adv_batch_x
    train_y = list(train_y) + adv_batch_y

    temp = list(zip(train_x, train_y))
    random.shuffle(temp)
    train_x[:], train_y[:] = zip(*temp)



    # dataloader, _ = model.module.dataset.transform_text(train_x, train_y, batch_size=args.batch_size)
    args.batch_size = 128
    dataloader, _ = model.dataset.transform_text(train_x, train_y, batch_size=args.batch_size)

    model.train()
    niter = 0
    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        sum_loss = 0
        # model.zero_grad()
        # print('Batch num:', len(dataloader))
        for input_ids, input_mask, segment_ids, labels in dataloader:  # 每个batch
            # start_time = time.clock()
            niter += 1
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            output = model(input_ids, input_mask, segment_ids)
            loss = criterion(output, labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())
            # use_time = (time.clock() - start_time)
            # print("Time for a batch: ", use_time)

        # test_acc = model.module.eval_model(test_x, test_y)
        test_acc = model.eval_model(test_x, test_y)
        sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(e, niter,
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr'], sum_loss,
                                                                                                 test_acc))
        if save_path:
            # torch.save(model.module.state_dict(), save_path)
            # torch.save(model.state_dict(), save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.model.state_dict(), save_path+'/pytorch_model.bin')
            model.model.config.to_json_file(save_path + '/bert_config.json')
            model.dataset.tokenizer.save_vocabulary(save_path)
            print('save model when test acc=', test_acc)

        lr_decay = 1
        if lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= lr_decay
    return test_acc

def main(args):
    print('Load data...')
    if args.task == 'mr':
        # train_x, train_y = dataloader.read_corpus('data/adversary_training_corpora/mr/train.txt', clean = False, FAKE = False, shuffle = True)
        # test_x, test_y = dataloader.read_corpus('data/adversary_training_corpora/mr/test.txt', clean = False, FAKE = False, shuffle = False) # 为了观察，暂时不shuffle
        train_x = args.datasets.train_seqs2
        train_x = [[args.inv_full_dict[word] for word in text] for text in train_x]  # 网络输入是词语
        train_y = args.datasets.train_y
        test_x = args.datasets.test_seqs2
        test_x = [[args.inv_full_dict[word] for word in text] for text in test_x]  # 网络输入是词语
        test_y = args.datasets.test_y
    elif args.task == 'imdb':
        train_x, train_y = dataloader.read_corpus(os.path.join(args.data_path + 'imdb', 'train_tok.csv'), clean=False, FAKE=False, shuffle=True)
        test_x, test_y = dataloader.read_corpus(os.path.join(args.data_path + 'imdb', 'test_tok.csv'), clean=False, FAKE=False, shuffle=True)
        # 随机选1000个测试集
        # test_x, test_y = test_x[:1000], test_y[:1000]

    print('Num of testset: %d' % (len(test_y)))

    print('Build model...')
    model = NLI_infer_BERT(args, args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    print('Load model from: %s' % args.target_model_path)

    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23457', rank=0, world_size=1)
    # model = nn.DataParallel(model)
    model = model.cuda()

    if args.mode == 'eval':  # tiz
        print('Eval...')
        # train_acc = model.module.eval_model0(train_x, train_y)
        # print('Original train acc: ', train_acc)
        # test_acc = model.module.eval_model(test_x, test_y)
        test_acc = model.eval_model(test_x, test_y)
        print('Original test acc: ', test_acc)
    else:
        print('Train...')
        need_grad = lambda x: x.requires_grad
        optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
        epoch = 100
        best_test = 0
        # best_test = train_model(epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)

        # =================传统对抗训练(对比用)=============
        # print('run train_model_adv')
        # train_adv_x, train_adv_y, = read_adv_example()
        # best_test = train_model_adv(epoch, model, optimizer, train_x, train_y, train_adv_x, train_adv_y, test_x, test_y, best_test, args.save_path)
        # ==================================

        print('run train_model2')
        best_test = train_model2(epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)
