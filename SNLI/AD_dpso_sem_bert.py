import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import pickle
import time

from attack_dpso_sem import PSOAttack
from encap_snli_bert import Model

from torch.autograd import Variable
import torch
import torch.nn as nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

with open('dataset/nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
with open('dataset/word_candidates_sense_top5.pkl','rb') as fp:
    word_candidate = pickle.load(fp)
with open('dataset/all_seqs.pkl', 'rb') as fh:
    _, _, test = pickle.load(fh)
with open('dataset/pos_tags_test.pkl', 'rb') as fp:  # 测试集
    pos_tags = pickle.load(fp)

test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]

vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}
model = Model(inv_vocab)

adversary = PSOAttack(model, word_candidate, pop_size=60, max_iters=20)  # 少了dataset参数初始化PSOAttack类
print('the length of test:', len(test_s1))
# TEST_SIZE = 5000
# test_idxs = np.random.choice(len(test_s1), size=TEST_SIZE, replace=False) # 不打乱数据
# 不合法数据
wrong_clas_id = []  # 保存错误预测的数据id
wrong_clas = 0  # 记录错误预测数据个数
too_long_id = []  # 保存太长被过滤的数据id

attack_list = []  # 记录待攻击样本id（整个数据集-错误分类的-长度不合法的）

failed_list = []  # 记录攻击失败数据id
failed_time = []  # 记录攻击失败时间
failed_input_list = []  # 记录攻击失败的数据及其实际标签

modift_too_much_id = []  # 记录攻击成功但替换比例过多的数据id
modift_too_much_changes = []  # 记录攻击成功但替换比例过多，替换个数
modift_too_much_input_list = []  # 记录攻击成功但替换比例过多的原始数据及标签
modift_too_much_output_list = []  # ...对抗样本...
modift_too_much_time = []  # ...攻击时间...



input_list = []  # 记录成功攻击的输入数据
output_list = []  # 记录成功攻击的对抗样本
success = []  # 记录成功攻击的数据id
change_list = []  # 记录成功攻击的替换比例
target_list = []  # 记录成功攻击的目标label（真实label的相反，1--0、0--1）
true_label_list = []  # 记录成功攻击的数据真实label
success_count = 0  # # 记录成功攻击数据个数
num_change_list = []  # 记录成功攻击的替换词个数
success_time = []  # 记录成功攻击的时间



SUCCESS_THRESHOLD = 0.25

preditc_true_num = 0
for i in range(len(test_s1)):
# while len(test_list) < len(train_s1) * 0.1:  # 获得10%的待扰动数据（正确分类且长度大于10）注意：这里的训练数据是原始数据的前25%
# i = -1
# while preditc_true_num <= 1000:
#     i += 1
    print('text id:', i)
    print('predict true num: ', preditc_true_num)

    s1 = test_s1[i]
    s2 = test_s2[i]
    pos_tag = pos_tags[i]
    ori_pred = np.argmax(model.pred([s1], [s2])[0])
    true_label = test['label'][i]
    x_len = np.sum(np.sign(s2))
    if ori_pred != true_label:
        wrong_clas_id.append(i)
        wrong_clas += 1
        print('Wrong classified')
    # elif x_len < 10:  # 过滤长度小于10的
    #     print('Skipping too short input')
    else:
        preditc_true_num += 1
        attack_list.append(i)
        if true_label == 2:
            target = 0
        elif true_label == 0:
            target = 2
        else:
            target = 0 if np.random.uniform() < 0.5 else 2
        time_start = time.time()
        attack_result = adversary.attack(np.array(s1), np.array(s2), target, pos_tag)
        time_end = time.time()
        adv_time = time_end - time_start
        if attack_result is None:
            print('failed! time:', adv_time)
            failed_list.append(i)
            failed_time.append(adv_time)
            failed_input_list.append([s1, s2, true_label])
        else:
            num_changes = np.sum(np.array(s2) != np.array(attack_result))
            x_len = np.sum(np.sign(s2))

            print('%d - %d changed.' % (i + 1, int(num_changes)))
            modify_ratio = num_changes / x_len
            if modify_ratio > 0.25:  # 过滤替换比例大于25%的
                modift_too_much_id.append(i)
                modift_too_much_changes.append(num_changes)
                modift_too_much_input_list.append([s1, s2, true_label])
                modift_too_much_output_list.append(attack_result)
                modift_too_much_time.append(adv_time)
                print('Modify more than 0.25:', modify_ratio)
            else:
                print('Successfully attack!')
                print(modify_ratio, )
                success_count += 1
                true_label_list.append(true_label)
                input_list.append([s1, s2, true_label])
                output_list.append(attack_result)
                success.append(i)
                target_list.append(target)
                change_list.append(modify_ratio)
                num_change_list.append(num_changes)  #
                success_time.append(adv_time)  #


target_model = 'bert'
task = 'snli'

with open('dataset/AD_dpso_sem_bert_ifvalid_10k.pkl', 'wb') as f:
    pickle.dump((wrong_clas_id, too_long_id, attack_list), f)
with open('dataset/AD_dpso_sem_bert_fail_10k.pkl', 'wb') as f:
    pickle.dump((failed_list, failed_input_list, failed_time), f)
with open('dataset/AD_dpso_sem_bert_modtoomuch_10k.pkl', 'wb') as f:
    pickle.dump((modift_too_much_id, modift_too_much_changes, modift_too_much_input_list, modift_too_much_output_list, modift_too_much_time), f)
with open('dataset/AD_dpso_sem_bert_success_10k.pkl', 'wb') as f:
    pickle.dump((input_list, true_label_list, output_list, success, change_list, target_list, num_change_list, success_time), f)


print('All %d data, Wrong classified: %d' % (len(test_s1), wrong_clas))
print('Attack in %d data, success rate: %f' % (len(attack_list), float(len(output_list))/float(len(attack_list))))
