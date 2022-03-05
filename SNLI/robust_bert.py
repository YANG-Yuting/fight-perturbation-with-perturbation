import numpy as np
import pickle
import time
from random import choice
from functools import reduce

from encap_snli_bert import Model

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

with open('dataset/nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
with open('dataset/word_candidates_sense_top5.pkl','rb') as fp:
    word_candidate=pickle.load(fp)
with open('dataset/all_seqs.pkl', 'rb') as fh:
    train, valid, test = pickle.load(fh)
with open('dataset/pos_tags_test.pkl','rb') as fp:
    test_pos_tags=pickle.load(fp)
with open('dataset/pos_tags.pkl','rb') as fp:
    train_pos_tags = pickle.load(fp)  # 137341条
# print('the length of test cases is:', len(test_s1))
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}

pos_tags = test_pos_tags

test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]
train_s1 = [t[1:-1] for t in train['s1']]
train_s2 = [t[1:-1] for t in train['s2']]  # 549367条


model = Model(inv_vocab)  # bert

sample_num = 5000

pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


"""获得一条文本的同义词空间大小，即所有可替换词数"""
def get_candidate_space(idx, s2):
    pos_tag = pos_tags[idx]
    candi_num = []
    for j in range(len(s2)):  # 对于每个词语
        word = s2[j]
        pos = pos_tag[j][1]  # 当前词语词性
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
        neigbhours = word_candidate[word][pos]  # 候选集
        candi_num.append(len(neigbhours))  # 保存当前词语的同义词个数
    candi_space = reduce(lambda x, y:x*y, candi_num)
    return candi_space


"""为一条文本采样"""
def gen_sample_aText(idx, s2):  # text一条文本，list
    # 获得同义词空间随机采样样本
    pos_tag = pos_tags[idx]
    sample_texts = []
    for s in range(sample_num):
        sample_text = s2.copy()
        for j in range(len(sample_text)):  # 对于每个词语
            word = sample_text[j]
            pos = pos_tag[j][1]  # 当前词语词性
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
            neigbhours = word_candidate[word][pos]  # 候选集
            neigbhours.append(word)  # 候选集包含自己
            sample_text[j] = choice(neigbhours)  # 候选集中随机选择一个
        sample_texts.append(sample_text)
    return sample_texts


"""计算一条数据的鲁棒性"""
def robust_aText(idx, s1, s2, true_label):
    # 1.确定采样个数
    # 若可替换同义词空间小于sample_num，则只需采样#同义词空间次
    # candidate_space = get_candidate_space(idx, s2)
    # sample_num = min(candidate_space, ori_sample_num)
    # print('Sample num:', sample_num)
    # 1. 获得采样数据
    sample_s2s = gen_sample_aText(idx, s2)
    # 2. 测试采样数据鲁棒性
    # 判断原始文本是否预测正确
    ori_label = np.argmax(model.pred([s1], [s2])[0])
    predict = (ori_label == true_label)  # 是否正确预测

    # 预测采样文本
    sample_labels = np.array([true_label] * len(sample_s2s))
    sample_s1s = [s1] * len(sample_s2s)
    sample_pred_labels = np.argmax(model.pred(sample_s1s, sample_s2s), axis=1)
    robust_value = float(sum(sample_labels == sample_pred_labels)) / float(sample_labels.shape[0])

    return predict, robust_value


def robust():
    # 测试集遍历数据获得鲁棒性
    if_train = False
    if if_train:
        print('For train set...')
        robust_file = 'dataset/robust_samp%d_train25_bert.txt' % sample_num
        f = open(robust_file, 'w')
        eval_num = 2500  # 评估数据个数
        for idx in range(eval_num):
            time_start = time.time()
            s1 = train_s1[idx]
            s2 = train_s2[idx]
            true_label = train['label'][idx]
            predict, robust_value = robust_aText(idx, s1, s2, true_label)
            time_end = time.time()
            print('Time:', time_end - time_start)
            print('%d %d %f' % (idx, predict, robust_value))
            print('%d %d %f' % (idx, predict, robust_value), file=f)
            f.flush()
    else:
        print('For test set...')
        pred_labels = np.argmax(model.pred(test_s1[:1000], test_s2[:1000]), axis=1)  # 1是行
        predicts = (pred_labels == test['label'][:1000])  # 是否正确预测
        test_acc = np.sum(predicts) / float(len(test_s1))
        print('accuracy test :', test_acc)
        exit(0)

        robust_file = 'dataset/robust_samp%d_test_bert.txt' % sample_num
        f = open(robust_file, 'w')
        # eval_num = int(0.25 * len(test_s1))  # 评估数据个数
        eval_num = len(test_s1)
        for idx in range(eval_num):
            time_start = time.time()
            s1 = test_s1[idx]
            s2 = test_s2[idx]
            true_label = test['label'][idx]
            predict, robust_value = robust_aText(idx, s1, s2, true_label)
            time_end = time.time()
            print('Time:', time_end - time_start)
            print('%d %d %f' % (idx, predict, robust_value))
            print('%d %d %f' % (idx, predict, robust_value), file=f)
            f.flush()

def ana_robust():
    train_robust_file = 'dataset/robust_samp%d_train25_bert.txt' % sample_num
    # test_robust_file =  'dataset/robust_samp%d_test25_bert.txt' % sample_num
    train_robust = open(train_robust_file, 'r').read().splitlines()
    # test_robust = open(test_robust_file, 'r').read().splitlines()

    train_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in train_robust]
    lt1 = sum(train_robust)
    print('snli(train): %d/%d(%f) data has robust value less than 1 for bert' % (lt1, len(train_robust), float(lt1)/float(len(train_robust))))

    # test_robust = [1 if float(r.split(' ')[2]) < 1 else 0 for r in test_robust]
    # lt1 = sum(test_robust)
    # print('snli(test): %d/%d(%f) data has robust value less than 1 for bert' % (lt1, len(test_robust), float(lt1)/float(len(test_robust))))


if __name__ == '__main__':
    robust()
    # ana_robust()
