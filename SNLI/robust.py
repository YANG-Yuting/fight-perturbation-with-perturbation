import numpy as np
import pickle
import time
from random import choice
from model_nli import Model
from encap_snli_bert import Model as Model1
from config import args
import sys
sys.path.append('..')
from train_classifier import gen_sample_multiTexts
import os

def robust():

    # 1. 加载数据集
    # 测试集（取前1000）
    test_s1 = args.test_s1[:1000]
    test_s2 = args.test_s2[:1000]
    true_labels = args.true_labels[:1000]
    # id转str
    # test_s1 = [[args.inv_full_dict[w] for w in t] for t in test_s1]
    # test_s2 = [[args.inv_full_dict[w] for w in t] for t in test_s2]

    # 2.加载模型
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args)  # wordLSTM
    elif args.target_model == 'bert':
        model = Model1(args)  # bert

    # 3.获得测试集鲁棒性
    print('For test set...')
    robust_file = '/pub/data/huangpei/TextFooler/data/adversary_training_corpora/new_robust_score/robust_snli_%s_samp%d_chg%s.txt' % (args.target_model, args.sample_num,args.change_ratio)
    f = open(robust_file, 'w')
    # 测试集可同时处理
    # 预测原始文本

    pred_labels = np.argmax(model.pred_org(None, (test_s1, test_s2)), axis=1) # 1是行
    predicts = (pred_labels == true_labels)  # 是否正确预测
    test_acc = np.sum(predicts)/float(len(test_s1))
    print('accuracy test :', test_acc)

    # 采样（对s2）
    sample_s2s = gen_sample_multiTexts(args, None, test_s2, args.sample_num, args.change_ratio)
    sample_labels = [l for l in true_labels for i in range(args.sample_num)]  # 每个lable复制sample_num 次
    # 获得采样文本预测准确率，即鲁棒性
    sample_s1s = [s1 for s1 in test_s1 for i in range(args.sample_num)]
    # 预测
    sample_probs = model.pred_org(None, (sample_s1s, sample_s2s))
    s_results = (np.argmax(sample_probs, axis=1) == sample_labels).reshape(len(true_labels), args.sample_num)
    R_scores = np.sum(s_results, axis=1) / float(args.sample_num)  # 每个训练点的鲁棒打分
    # 写入文件
    for i in range(len(true_labels)):
        predict = predicts[i]
        R_score = R_scores[i]
        print('%d\t%d\t%f' % (i, predict, R_score))
        print('%d\t%d\t%f' % (i, predict, R_score), file=f)
        f.flush()

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
    robust()
    # ana_robust()
