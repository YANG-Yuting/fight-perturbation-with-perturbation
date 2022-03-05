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

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# parser = argparse.ArgumentParser()
# ## Required parameters
# parser.add_argument("--task", type=str, default='imdb', help="task name: mr/imdb")
# parser.add_argument("--dataset_path",
#                     type=str,
#                     default='data/imdb',
#                     help="Which dataset to attack.")
# parser.add_argument("--nclasses",
#                     type=int,
#                     default=2,
#                     help="How many classes for classification.")
# parser.add_argument("--target_model",
#                     type=str,
#                     default='wordLSTM',
#                     help="Target models for text classification: fasttext, charcnn, word level lstm "
#                          "For NLI: InferSent, ESIM, bert-base-uncased")
# parser.add_argument("--target_model_path",
#                     type=str,
#                     default='models/wordLSTM/imdb',
#                     help="pre-trained target model path")
# parser.add_argument("--word_embeddings_path",
#                     type=str,
#                     default='glove.6B/glove.6B.200d.txt',
#                     help="path to the word embeddings for the target model")
# parser.add_argument("--counter_fitting_embeddings_path",
#                     type=str,
#                     default='data/counter-fitted-vectors.txt',
#                     help="path to the counter-fitting embeddings we used to find synonyms")
# parser.add_argument("--counter_fitting_cos_sim_path",
#                     type=str,
#                     default='data/cos_sim_counter_fitting.npy',
#                     help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
# parser.add_argument("--USE_cache_path",
#                     type=str,
#                     default='',
#                     help="Path to the USE encoder cache.")
# parser.add_argument("--output_dir",
#                     type=str,
#                     default='adv_results',
#                     help="The output directory where the attack results will be written.")
#
# ## Model hyperparameters
# parser.add_argument("--sim_score_window",
#                     default=15,
#                     type=int,
#                     help="Text length or token number to compute the semantic similarity score")
# parser.add_argument("--import_score_threshold",
#                     default=-1.,
#                     type=float,
#                     help="Required mininum importance score.")
# parser.add_argument("--sim_score_threshold",
#                     default=0,  # 0.7
#                     type=float,
#                     help="Required minimum semantic similarity score.")
# parser.add_argument("--synonym_num",
#                     default=50,
#                     type=int,
#                     help="Number of synonyms to extract")
# parser.add_argument("--batch_size",
#                     default=128,
#                     type=int,
#                     help="Batch size to get prediction")
# parser.add_argument("--data_size",
#                     default=1000,
#                     type=int,
#                     help="Data size to create adversaries")
# parser.add_argument("--perturb_ratio",
#                     default=0.,
#                     type=float,
#                     help="Whether use random perturbation for ablation study")
# parser.add_argument("--max_seq_length",
#                     default=128,
#                     type=int,
#                     help="max sequence length for BERT target model")
# args = parser.parse_args()
#
# seq_len_list = {'imdb': 256, 'mr': 128}
# args.max_seq_length = seq_len_list[args.task]
# sizes = {'imdb': 50000, 'mr': 20000}
# max_vocab_size = sizes[args.task]
#
# with open('data/adversary_training_corpora/%s/dataset_%d.pkl' % (args.task, max_vocab_size), 'rb') as f:
#     dataset = pickle.load(f)
# with open('data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
#     word_candidate = pickle.load(fp)
# with open('data/adversary_training_corpora/%s/pos_tags_test.pkl' % args.task, 'rb') as fp:  # 针对测试集获得对抗样本
#     pos_tags = pickle.load(fp)
#
# inv_full_dict = dataset.inv_full_dict
# full_dict = dataset.full_dict

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
    def __init__(self, args,pretrained_dir,nclasses,max_seq_length=128, batch_size=32):
        super(NLI_infer_BERT, self).__init__()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)
        self.args = args
        self.tf_vocabulary=pickle.load(open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/tf_vocabulary.pkl' % args.task, "rb"))
        # self.parall()
        if self.args.prompt_generate:  # 若希望用prompt生成对抗样本，需要的bert模型是BertForMaskedLM
            self.model_gen = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
        else:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

    def text_pred_prompt(self, ori_texts, texts, labels, batch_size=64):
        """利用prompt和MLM，进行mask fill"""
        self.model.eval()

        # transform text data into indices and create batches
        dataloader, _ = self.dataset.transform_text0(texts, labels=labels, batch_size=batch_size)
        outs = []
        for input_ids, input_mask, segment_ids, batch_labels in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            with torch.no_grad():
                prediction_scores = self.model(input_ids, segment_ids, input_mask)  # [b_size,128,30522]

                """方法一：每个MASK位置取最大作为补全词语"""
                # mask_ps = (input_ids == 103).nonzero()
                # pred_ids = torch.argmax(prediction_scores, dim=2)# .cpu().numpy().tolist()  # [b_size,128]
                # # 替换补全
                # for ps in mask_ps: # 对于所有的mask位置
                #     input_ids[ps[0],ps[1]] = pred_ids[ps[0],ps[1]]
                # # 截取、转word
                # for ii in range(len(input_ids)):
                #     input_ids_a = input_ids[ii].cpu().numpy().tolist()
                #     num_padding = len(input_mask[ii])-torch.sum(input_mask[ii])
                #     input_ids_a = input_ids_a[1: (self.args.max_seq_length-num_padding-5)-1]
                #     pred_words.append(self.dataset.tokenizer.convert_ids_to_tokens(input_ids_a))  # [128]

                """方法二：每个MASK位置保留Top k"""
                mask_ps = [] # list，每个list是该句子mask位置的所有索引
                for i in range(len(input_ids)):
                    m = np.where(input_ids[i].cpu().numpy() == 103)[0].tolist()
                    mask_ps.append(m)
                _, pred_ids = prediction_scores.topk(k=self.args.topk, dim=2, largest=True, sorted=True) # 每个位置保留topk
                # 替换补全
                num_candi_all = []  # 保存每条数据的候选集大小（对于每个句子，是k的mask个数次方）
                # 计算对于每个测试样例的所有候选集大小
                # 外层，每个测试样例，采样sample_size种mask版本；内层，对于每个mask版本，组合每个mask位置的topk种候选集
                # 存在特殊情况：可能采样时，无实际mask位置

                for i in range(len(input_ids)):  # 对于每条数据
                    mp_a = mask_ps[i]  # 对于该数据，mask的位置
                    # 存在无mask位置的情况
                    if len(mp_a) == 0:
                        continue

                    num_padding = len(input_mask[i]) - torch.sum(input_mask[i])  # 当前数据中padding位置的个数
                    candi_ids = pred_ids[i, mp_a[:], :].cpu().numpy().tolist()  # 对于该数据，mask位置的候选集
                    candi_comb = list(itertools.product(*candi_ids))  # 对于该数据，所有mask位置候选集的可能组合
                    num_candi_all.append(len(candi_comb))

                    input_ids_a = input_ids[i] # 当前数据
                    input_ids_aa = input_ids_a.unsqueeze(0).repeat(len(candi_comb), 1) # 当前数据的所有生成候选数据

                    # 为所有候选数据，完成替换
                    for j in range(len(input_ids_aa)): # 对于每条候选数据
                        input_ids_aa[j][mp_a] = torch.tensor(list(candi_comb[j])).cuda()
                        para_sents = self.dataset.tokenizer.convert_ids_to_tokens(
                            input_ids_aa[j][1: (self.args.max_seq_length - num_padding - 5) - 1].cpu().numpy().tolist()
                        ) #注意：只保存和输入相对应的输出
                        # 保存：[原始label，原始数据，mask位置，替换后数据]
                        outs.append([batch_labels[i].numpy().tolist(), self.dataset.tokenizer.tokenize(' '.join(ori_texts[i])),
                                     [mmm-1 for mmm in mp_a], para_sents])
                        # -1：输入的第0个位置是CLS

        with open('/pub/data/huangpei/TextFooler/prompt_results/outs_%s_%s_%.2f_%d_%d' % (self.args.task, self.args.target_model, self.args.mask_ratio, self.args.num_candi, self.args.topk), 'wb') as fw:
            pickle.dump(outs, fw)
        return outs


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
        # labels随便给的，为了调用transform_text函数
        dataloader, _ = self.dataset.transform_text0(text_data, labels=[0] * len(text_data), batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
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
        # labels随便给的，为了调用transform_text函数
        dataloader, _ = self.dataset.transform_text0(text_data, labels=[0] * len(text_data), batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids,_ in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

    # tiz
    # tiz: 加强的分类器
    # text：[[]]有多条数据，是word
    def text_pred_Enhance(self, orig_texts, text, sample_num=256, batch_size=128): # text是str

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



# def eval_bert(model, dataloader):
#     model.eval()
#
#     with torch.no_grad():
#         correct = 0
#         local_rank = torch.distributed.get_rank()
#         torch.cuda.set_device(local_rank)
#         device = torch.device("cuda", local_rank)
#         for step, (input_x, input_y) in enumerate(dataloader):
#             # input_x = torch.tensor(batch[0]).cuda(local_rank, non_blocking=True)
#             # input_y = torch.tensor(batch[1]).cuda(local_rank, non_blocking=True)
#
#             # input_x = torch.tensor(batch[0]).to(device)
#             # input_y = torch.tensor(batch[1]).to(device)
#             print('input_x', len(input_x), input_x)
#             print('input_y', len(input_y), input_y)
#             exit(0)
#
#             ori_probs = model(input_x)
#
#             ori_labels = torch.argmax(ori_probs, dim=1)
#
#             #ori_labels=torch.distributed.reduce(ori_labels.clone(),op=dist.ReduceOp.SUM)
#
#             input_y = torch.tensor(input_y).cuda()
#
#             print(ori_labels)
#             print(input_y)
#             exit(0)
#
#             correct += ori_labels.eq(input_y).cpu().sum()
#
#             torch.distributed.barrier()
#
#     return correct.item()/len(dataloader)




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


def attack(text_id, text_ls, true_label, cos_sim, predictor, stop_words_set, word2idx, idx2word, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # ---替换同义词获得方式--- #
        # origion
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)  # list of len(words_perturb_idx), each is a list with syn word
        # tiz
        # perturb_idx = [idx for idx, word in words_perturb]
        # synonym_word_ids = get_syn_words(text_id, perturb_idx, text_ls)
        # synonym_words = [[inv_full_dict[j] for j in k] for k in synonym_word_ids]

        synonyms_all = []
        for idx, word in words_perturb:
            # synonyms = synonym_words.pop(0)  # tiz
            if word in word2idx:
                synonyms = synonym_words.pop(0)  # origion
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries

def main():
    # 不合法数据
    wrong_clas_id = []  # 保存错误预测的数据id
    wrong_clas = 0  # 记录错误预测数据个数
    # too_long_id = []  # 保存太长被过滤的数据id

    # attack_list = []  # 记录待攻击样本id（整个数据集-错误分类的-长度不合法的）

    failed_list = []  # 记录攻击失败数据id
    failed_time = []  # 记录攻击失败时间
    failed_input_list = []  # 记录攻击失败的数据及其实际标签

    input_list = []  # 记录成功攻击的输入数据
    output_list = []  # 记录成功攻击的对抗样本
    success = []  # 记录成功攻击的数据id
    change_list = []  # 记录成功攻击的替换比例
    true_label_list = []  # 记录成功攻击的数据真实label
    success_count = 0  # # 记录成功攻击数据个数
    num_change_list = []  # 记录成功攻击的替换词个数
    success_time = []  # 记录成功攻击的时间


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack
    # origion
    # texts, labels = dataloader.read_corpus(args.dataset_path)
    # tiz
    texts = dataset.test_seqs2

    texts = [[inv_full_dict[word] for word in text]for text in texts]  # 网络输入是词语
    labels = dataset.test_y

    data = list(zip(texts, labels))
    # data = data[:args.data_size]  # choose how many samples for adversary

    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.max_seq_length, args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
        print('Save cosine similarity matrix...')  # tiz
        np.save('cos_sim_counter_fitting.npy', cos_sim)
    print("Cos sim import finished!")


    # sim_order = np.argsort(-cos_sim[word2idx['movie'], :])
    # sim_order = sim_order[:20]
    # sim_word = [idx2word[id] for id in sim_order]
    # print(sim_word)


    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    for idx, (text, true_label) in enumerate(data):
        print('text id:', idx)
        if idx % 20 == 0:
            print('{} samples out oqf {} have been finished!'.format(idx, args.data_size))
        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, \
            new_label, num_queries = random_attack(text, true_label, predictor, args.perturb_ratio, stop_words_set,
                                                    word2idx, idx2word, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)
        else:
            time_start = time.time()
            new_text, num_changed, orig_label, \
            new_label, num_queries = attack(idx, text, true_label, cos_sim, predictor, stop_words_set,
                                            word2idx, idx2word, sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)
            time_end = time.time()
            adv_time = time_end - time_start

        if true_label != orig_label:
            orig_failures += 1
            wrong_clas += 1
            wrong_clas_id.append(idx)
            print('Wrong classified!')
            continue
        else:
            nums_queries.append(num_queries)

        if true_label == new_label:
            print('failed! time:', adv_time)
            failed_list.append(idx)
            failed_time.append(adv_time)
            failed_input_list.append([text, true_label])
            continue

        if true_label != new_label:
            modify_ratio = 1.0 * num_changed / len(text)
            # print(idx, text)
            # print(num_changed, new_text)

            success_count += 1
            true_label_list.append(true_label)
            input_list.append([text, true_label])
            output_list.append(new_text)
            success.append(idx)
            change_list.append(modify_ratio)
            num_change_list.append(num_changed)  #
            success_time.append(adv_time)  #

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
                                                                     (1-orig_failures/1000)*100,
                                                                     (1-success_count/1000)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries))
    print(message)
    log_file.write(message)

    # with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
    #     for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
    #         ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

    with open(args.output_dir + '/adversaries_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
        pickle.dump((wrong_clas_id), f)
    with open(args.output_dir + '/adversaries_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
        pickle.dump((failed_list, failed_input_list, failed_time), f)
    with open(args.output_dir + '/adversaries_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
        pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)


if __name__ == "__main__":
    main()