import argparse
import os
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from torch.autograd import Variable

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig,BertForMaskedLM
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, RandomSampler

from train_classifier import perturb_texts, gen_sample_multiTexts,perturb_FGWS
import pickle
import os
import itertools

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-large/3"  # tiz modifies the url to google.cn
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class NLI_infer_BERT(nn.Module):
    def __init__(self, args, pretrained_dir,nclasses,max_seq_length=128, batch_size=32):
        super(NLI_infer_BERT, self).__init__()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)
        self.args = args
        self.tf_vocabulary=pickle.load(open(args.data_path + '%s/tf_vocabulary.pkl' % args.task, "rb"))
        # self.parall()
        if self.args.prompt_generate:  # 若希望用prompt生成对抗样本，需要的bert模型是BertForMaskedLM
            self.model_gen = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        else:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

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


    def text_pred_org(self, orig_texts, text_data, batch_size=128):
        # Switch the model to eval mode.
        # self.model.eval()

        # transform text data into indices and create batches
        dataloader, _ = self.dataset.transform_text0(text_data, labels=[0] * len(text_data), batch_size=batch_size)

        probs_all = []
        for input_ids, input_mask, segment_ids,_ in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


    def text_pred_adv(self, orig_texts, text_data, batch_size=128):
        # Switch the model to eval mode.
        # self.model.eval()

        # transform text data into indices and create batches
        dataloader, _ = self.dataset.transform_text0(text_data, labels=[0] * len(text_data), batch_size=batch_size)

        probs_all = []
        for input_ids, input_mask, segment_ids,_ in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


    def text_pred_Enhance(self, orig_texts, text, sample_num=256, batch_size=128):
        probs_boost_all = []
        perturbed_texts = perturb_texts(self.args, orig_texts, text, self.tf_vocabulary, change_ratio=0.2)
        #perturbed_texts=text
        Samples_x = gen_sample_multiTexts(self.args, orig_texts, perturbed_texts, sample_num, change_ratio=0.25)
        Sample_probs = self.text_pred_org(None, Samples_x, batch_size)
        lable_mum=Sample_probs.size()[-1]
        Sample_probs=Sample_probs.view(len(text),sample_num,lable_mum)
        probs_boost = []
        for l in range(lable_mum):
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l),dim=1)  # 获得预测值的比例作为对应标签的概率
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
            num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)  # 获得预测值的比例作为对应标签的概率
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

        probs_boost_all=torch.where(D > 0, pre_prob, ori_prob)
        #D=D.view(-1)
        #probs_boost_all=torch.ones_like(ori_prob)
        #probs_boost_all.index_put_((index,lable),0.5 - D)
        #probs_boost_all.index_put_((index,1-lable), 0.5+D)
        return probs_boost_all



    # input_ids, input_mask, segment_ids：一个batch内
    def forward(self,input_ids, input_mask, segment_ids):
        # input_ids = input_ids.cuda()
        # input_mask = input_mask.cuda()
        # segment_ids = segment_ids.cuda()
        logits = self.model(input_ids, segment_ids, input_mask)
        probs = nn.functional.softmax(logits, dim=-1)
        return probs


    def parall(self):
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        # self.model = DistributedDataParallel(self.model)  # device_ids will include all GPU devices by default
        # self.model = self.model.cuda()
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model) #, device_ids=[0, 1, 2]

        # torch.distributed.init_process_group(backend='nccl')
        # # 为每个进程配置GPU
        # local_rank = torch.distributed.get_rank()
        # torch.cuda.set_device(local_rank)
        # device = torch.device("cuda", local_rank)
        # # 要先将model放到gpu上
        # self.model = self.model.to(device)
        # self.model = DistributedDataParallel(self.model, find_unused_parameters=True, device_ids=[local_rank],output_device=local_rank)

        # # 指定要用到的设备
        self.model = self.model.cuda().train()
        self.model = nn.DataParallel(self.model)

        # self.model = torch.nn.DataParallel(self.model)
        # # # 模型加载到设备0
        # self.model = self.model.cuda()

    def eval_model(self, input_x, input_y):
        self.model.eval()
        predictor = self.text_pred()
        ori_probs = predictor(input_x, input_x)
        ori_labels = torch.argmax(ori_probs, dim=1)
        input_y = torch.tensor(input_y).long().cuda()
        correct = ori_labels.eq(input_y).cpu().sum()
        self.model.train()
        return correct.item() / len(input_x)




class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        token_noise = []
        for (ex_index, text_a) in enumerate(examples):
            # print(text_a)
            tokens_a = tokenizer.tokenize(' '.join(text_a))  # 109

            for i in range(len(tokens_a)):
                tok = tokens_a[i]
                if '##' in tok:
                    token_noise.append(i)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features, token_noise

    # train
    def transform_text(self, data, labels, batch_size=128):
        # transform data into seq of embeddings
        eval_features, token_noise = self.convert_examples_to_features(data, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long) # tiz
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels)

        # Run prediction for full data
        # eval_sampler = SequentialSampler(eval_data)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data) # tiz
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader, token_noise

    # test
    def transform_text0(self, data, labels, batch_size=128):
        # transform data into seq of embeddings
        eval_features, token_noise = self.convert_examples_to_features(data, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long) # tiz
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        # eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data) # tiz
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader, token_noise


if __name__ == "__main__":
    pass