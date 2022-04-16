import pickle
import numpy as np

import os

from torch.autograd import Variable
import torch
import torch.nn as nn
from infersent_model import NLINet
import sys
sys.path.append('..')
from train_classifier import perturb_texts, gen_sample_multiTexts,perturb_FGWS
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()

        self.word_vec = pickle.load(open(args.data_path + 'snli/word_vec.pkl', 'rb'))
        self.word_vec[42391]=pickle.load(open(args.data_path + 'snli/glove_unk.pkl','rb'))
        #print(self.word_vec[42391])
        config_nli_model = {
            'n_words': len(self.word_vec),
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': 64,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 1,
            'encoder_type': 'InferSent',
            'use_cuda': True,
        }
        self.nli_net = NLINet(config_nli_model)
        self.nli_net.load_state_dict(torch.load(args.target_model_path, map_location='cuda:0'))
        self.word_emb_dim = 300
        self.max_seq_len = args.max_seq_length  # 64
        self.args = args

        self.nli_net.cuda()


    def get_batch(self,batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        # max_len = np.max(lengths)
        max_len = self.max_seq_len
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = word_vec[batch[i][j]]
        return torch.from_numpy(embed).float(), lengths


    def eval_model(self, s1, s2, target):
        predictor = self.text_pred()
        ori_probs = predictor(s2, (s1,s2))
        ori_labels = torch.argmax(ori_probs, dim=1)
        input_y = torch.tensor(target).long().cuda()
        correct = ori_labels.eq(input_y).cpu().sum()
        return correct.item() / len(s2)

    def softmax_pred(self, n):
        s = [np.exp(t) for t in n]
        ss = np.sum(s)
        result = [t / ss for t in s]
        return result


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


    def text_pred_org(self, ori_s2, text):
        self.nli_net.eval()  # tiz adds
        s1 = text[0]
        s2 = text[1]

        with torch.no_grad():
            s1 = [s.tolist() if not isinstance(s, list) else s for s in s1]
            s2 = [s.tolist() if not isinstance(s, list) else s for s in s2]

            new_s1, new_s2 = [], []
            for a, b in zip(s1, s2):
                a = [self.args.full_dict[w] for w in a]
                b = [self.args.full_dict[w] for w in b]
                if '<s>' not in a:
                    a = ['<s>'] + a + ['</s>']
                if '<s>' not in b:
                    b = ['<s>'] + b + ['</s>']
                new_s1.append(a)
                new_s2.append(b)

            new_pred = []
            for i in range(0, len(new_s1), self.args.batch_size):
                # prepare batch
                s1_batch, s1_len = self.get_batch(new_s1[i: min(len(new_s1), i + self.args.batch_size)], self.word_vec, self.word_emb_dim)
                s2_batch, s2_len = self.get_batch(new_s2[i: min(len(new_s1), i + self.args.batch_size)], self.word_vec, self.word_emb_dim)
                s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
                output = self.nli_net((s1_batch, s1_len), (s2_batch, s2_len))
                new_pred.append(F.softmax(output, dim=-1))

        return torch.cat(new_pred, dim=0)

    def text_pred_adv(self, ori_s2, text):
        self.nli_net.eval()
        s1 = text[0]
        s2 = text[1]

        with torch.no_grad():
            s1 = [s.tolist() if not isinstance(s, list) else s for s in s1]
            s2 = [s.tolist() if not isinstance(s, list) else s for s in s2]
            new_s1, new_s2 = [], []
            for a, b in zip(s1, s2):
                a = [self.args.full_dict[w] for w in a]
                b = [self.args.full_dict[w] for w in b]
                if '<s>' not in a:
                    a = ['<s>'] + a + ['</s>']
                if '<s>' not in b:
                    b = ['<s>'] + b + ['</s>']
                new_s1.append(a)
                new_s2.append(b)

            new_pred = []
            for i in range(0, len(new_s1), self.args.batch_size):
                # prepare batch
                s1_batch, s1_len = self.get_batch(new_s1[i: min(len(new_s1), i + self.args.batch_size)], self.word_vec, self.word_emb_dim)
                s2_batch, s2_len = self.get_batch(new_s2[i: min(len(new_s1), i + self.args.batch_size)], self.word_vec, self.word_emb_dim)
                s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
                output = self.nli_net((s1_batch, s1_len), (s2_batch, s2_len))
                new_pred.append(F.softmax(output, dim=-1))

        return torch.cat(new_pred, dim=0)

    def text_pred_Enhance(self, orig_s2, text, sample_num=256):
        self.nli_net.eval()
        s1 = text[0]
        s2 = text[1]
        with torch.no_grad():
            perturbed_s2 = perturb_texts(self.args, orig_s2, s2, self.args.tf_vocabulary, change_ratio=0.2)
            Samples_s2 = gen_sample_multiTexts(self.args, orig_s2, perturbed_s2, sample_num, change_ratio=0.25)

            s1s=[s for s in s1 for i in range(sample_num)]

            Sample_probs = self.text_pred_org(orig_s2, (s1s,Samples_s2))
            lable_mum=Sample_probs.shape[-1]
            Sample_probs = Sample_probs.view(len(s2),sample_num, lable_mum)
            probs_boost = []
            for l in range(lable_mum):
                num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l),dim=1)
                prob = num.float() / float(sample_num)
                probs_boost.append(prob.view(len(s2),1))
            probs_boost_all=torch.cat(probs_boost,dim=1)

        return probs_boost_all

    def text_pred_SAFER(self,orig_s2, text, sample_num=256):
        self.nli_net.eval()
        s1 = text[0]
        s2 = text[1]
        with torch.no_grad():
            Samples_s2 = gen_sample_multiTexts(self.args, orig_s2, s2, sample_num, change_ratio=1)
            s1s = [s for s in s1 for i in range(sample_num)]
            Sample_probs = self.text_pred_org(orig_s2,(s1s, Samples_s2))
            lable_mum=Sample_probs.size()[-1]
            Sample_probs=Sample_probs.view(len(s2),sample_num,lable_mum)
            probs_boost = []
            for l in range(lable_mum):
                num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
                prob = num.float() / float(sample_num)
                probs_boost.append(prob.view(len(s2), 1))
            probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_FGWS(self,orig_s2, text):
        self.nli_net.eval()
        s1 = text[0]
        s2 = text[1]
        gamma1=0.5
        with torch.no_grad():
            perturbed_s2=perturb_FGWS(self.args, orig_s2, s2, self.args.tf_vocabulary)
            pre_prob=self.text_pred_org(orig_s2, (s1,perturbed_s2))
            ori_prob=self.text_pred_org(orig_s2, (s1,s2))
            lable=torch.argmax(ori_prob, dim=1)
            index=torch.arange(len(text[1])).cuda()
            D=ori_prob.gather(1, lable.view(-1, 1))-pre_prob.gather(1, lable.view(-1, 1))-gamma1
            D=D.view(-1)
            #probs_boost_all=torch.ones_like(ori_prob)
            probs_boost_all=ori_prob.index_put((index,lable), -D)
            #probs_boost_all.index_put_((index,1-lable), 0.5+D)
        return probs_boost_all


