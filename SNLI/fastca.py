# /usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pickle
import time
from random import choice

from model_nli import Model


with open('dataset/nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
with open('dataset/word_candidates_sense.pkl','rb') as fp:
    word_candidate=pickle.load(fp)
with open('dataset/all_seqs.pkl', 'rb') as fh:
    train, valid, test = pickle.load(fh)
with open('dataset/pos_tags_test.pkl','rb') as fp:
    test_pos_tags=pickle.load(fp)

# with open('dataset/pos_tags.pkl','rb') as fp:
#     train_pos_tags=pickle.load(fp)
# train = False
# if train:
#     pos_tags = train_pos_tags
# else:
#     pos_tags = test_pos_tags

pos_tags = test_pos_tags

test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]
train_s1 = [t[1:-1] for t in train['s1']]
train_s2 = [t[1:-1] for t in train['s2']]


train = False
sample_num = 3000
target_model = 'bdlstm'  # bdlstm bert
if train:
    fastca_in_dir = 'dataset/fastca_ins/%s/train' % target_model
else:
    fastca_in_dir = 'dataset/fastca_ins/%s/test' % target_model

def pre_for_fastCA_hownet(idx, s2):
    # fastca_ana = open(args.ca_dir + '/pertub_pos.txt', 'w')  # 统计所有数据的可替换位置个数
    # 1. 加载数据及候选替换集
    pos_tag = pos_tags[idx]  # 当前文本所有单词词性
    # 2. 获取待测试文本各个位置的同义词个数
    syn_words_num = [1] * len(s2)  # 保存该文本各个位置同义词个数（包含自己）
    s2_syns = [[t] for t in s2]  # 保存该文本各个位置同义词（包含自己。获取CA后，用其映射回词语）
    pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    pertub_psts = []  # 保存该文本的所有可替换位置
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
        if len(neigbhours) >0:
            pertub_psts.append(i)
        s2_syns[i] += neigbhours
        syn_words_num[i] += len(neigbhours)
    # 2. 写入文件
    # 若无可替换位置，不写
    if len(pertub_psts) == 0:
        return None
    # 写入该文本的替换位置和替换数量
    print('%d data has %d in %d pertub positions.' % (idx, len(pertub_psts), len(s2)))
    # print('%d data has %d in %d pertub positions.' % (idx, len(pertub_psts), len(s2)), file=fastca_ana)
    # print(pertub_psts)
    # print(pertub_psts, file=fastca_ana)
    # fastca_ana.flush()
    fastca_in_file = fastca_in_dir + '/' + str(idx) + '_fastca_in.txt' # fastca所需文件
    t = '2'  # 2-路覆盖
    with open(fastca_in_file, 'w') as w:
        w.write(t + '\n')  # 第一行：写入t
        w.write(str(len(s2)) + '\n')  # 第二行：写入文本长度（词语个数）
        syn_words_num = [str(sw) for sw in syn_words_num]
        w.write(' '.join(syn_words_num) + '\n')  # 第三行：写入所有位置同义词个数
        w.close()
    return s2_syns


def get_all_fastcaInfo():
    if train:
        # 取出不鲁棒数据
        train_robust_file = 'dataset/robust_samp%d_train25_%s.txt' % (sample_num, target_model)
        train_robust = open(train_robust_file, 'r').read().splitlines()
        if target_model == 'bdlstm':
            train_robust = [1 if float(r.split(' ')[2]) < 100 else 0 for r in train_robust]  # 只有这里是整数，小于100
        else:
            train_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in train_robust]
        non_robust_id = np.nonzero(np.array(train_robust))[0]
        # 不鲁棒数据写入文件
        with open('dataset/non_robust_file.txt','w') as fp:
            print('Non robust file:', non_robust_id)
            print('Non robust file:', non_robust_id, file=fp)
            fp.flush()
        # 获得一条数据的fastca input数据
        for idx in non_robust_id:
            s2 = train_s2[idx]
            _ = pre_for_fastCA_hownet(idx, s2)
    else:
        # 取出不鲁棒数据
        test_robust_file = 'dataset/robust_samp%d_test25_%s.txt' % (sample_num, target_model)
        test_robust = open(test_robust_file, 'r').read().splitlines()
        if target_model == 'bdlstm':
            test_robust = [1 if float(r.split(' ')[2]) < 100 else 0 for r in test_robust]
        else:
            test_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in test_robust]
        non_robust_id = np.nonzero(np.array(test_robust))[0]
        with open('dataset/non_robust_file.txt', 'w') as fp:
            print('Non robust file:', non_robust_id)
            print('Non robust file:', non_robust_id, file=fp)
            fp.flush()
        # 获得fastca input
        # 获得一条数据的fastca input数据
        for idx in non_robust_id:
                s2 = test_s2[idx]
                _ = pre_for_fastCA_hownet(idx, s2)

    
if __name__ == '__main__':
    get_all_fastcaInfo()