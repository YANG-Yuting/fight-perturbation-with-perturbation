from __future__ import print_function
import argparse
import torch
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
import os
import torch.optim as optim
import sys
import random
from train_classifier import Model, eval_model, train_model, train_model1
from attack_classification import NLI_infer_BERT, eval_bert

import dataloader


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True # 不全部占满显存, 按需分配
# sess = tf.Session(config=config)
# KTF.set_session(sess)





"""加载模型"""
def load_model():
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

    print("Model built!")

    return model


def adv_train():
    # 1. 加载原始模型
    model = load_model()

    # 2. 加载测试集(用于测试模型准确率)
    test_x = dataset.test_seqs2
    test_x = [[inv_full_dict[w] for w in x] for x in test_x]  # 网络输入是词语
    test_y = dataset.test_y
    print('Test set:', len(test_x))
    # 原始测试集准确率

    # orig_probs_1 = model.text_pred([test_x[0]])  # tensor, [1,2]
    # orig_probs_ = model.text_pred(test_x)  # tensor, [1,2]
    # pred_labels = torch.argmax(orig_probs_, dim=1).cpu().numpy()
    # print(np.sum(test_y == pred_labels), float(np.sum(test_y == pred_labels))/float(len(test_y)))
    # with open('tttest.txt', 'w') as fp:
    #     for i in range(len(pred_labels)):
    #         print(pred_labels[i])
    #         print(pred_labels[i], file=fp)

    test_x, test_y = dataloader.create_batches(test_x, test_y, args.max_seq_length,args.batch_size, model.word2id, )
    test_acc = eval_model(model, test_x, test_y)
    print('Original test acc: {:.1%}'.format(test_acc))

    # 3. 加载训练集（包括 原始文本 + 其中前10%的原始文本获得的对抗样本）
    # 原始训练集
    train_x = dataset.train_seqs2
    train_x = [[inv_full_dict[w] for w in x] for x in train_x]  # 网络输入是词语
    train_y = dataset.train_y
    train_y = list(train_y)
    print('Original train set:', len(train_x))
    orig_train_x = train_x.copy()
    orig_train_y = train_y.copy()
    orig_train_x, orig_train_y = dataloader.create_batches(orig_train_x, orig_train_y, args.max_seq_length, args.batch_size, model.word2id, )
    train_acc = eval_model(model, orig_train_x, orig_train_y)
    sys.stdout.write("Originally, train acc: {:.6f}\n".format(train_acc))


    if args.mode == 'adversary_example':
        # 对抗样本
        with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s.pkl' % (args.task, args.target_model), 'rb') as f:
            input_list, test_list, true_label_list, output_list, success, change_list, target_list = pickle.load(f)
        # with open('data/adversary_training_corpora/%s/AD_dpso_sem_%s_all.pkl' % (args.task, args.target_model),
        #           'rb') as f:
        #     input_list, true_label_list, output_list, success, change_list, target_list = pickle.load(f)

        print('Adv examples:', len(output_list))
        output_list = [[inv_full_dict[w] for w in x] for x in output_list]  # 网络输入是词语

        if args.select_nonrobust:
            # 筛选鲁棒性低的对抗样本
            # 鲁棒性低于1的训练数据id
            train_robust_file = 'data/adversary_training_corpora/%s/robust_train_%s.txt' % (args.task, args.target_model)
            train_robust = open(train_robust_file, 'r').read().splitlines()
            train_robust = np.array([float(r.split(' ')[2]) for r in train_robust])
            # id_gt_09 = np.where(train_robust > 0.9)
            # id_lt_1 = np.where(train_robust < 1)
            # nonrobust_id = np.intersect1d(id_gt_09, id_lt_1)
            nonrobust_id = np.where(train_robust < 1)
            # 成功攻击且鲁棒性小于1的训练数据id
            nonrobust_adv_id = np.intersect1d(nonrobust_id, np.array(success))
            # 转化成对抗样本索引
            nonrobust_adv_idx = np.isin(success, nonrobust_adv_id, invert=False)
            output_list = np.array(output_list)[nonrobust_adv_idx].tolist()
            true_label_list = np.array(true_label_list)[nonrobust_adv_idx].tolist()
            print('Non-robust adv examples:', len(output_list))

        # 测试对抗样本分类准确率
        output_list_1 = output_list.copy()
        output_list_1, true_label_list_1 = dataloader.create_batches(output_list_1, true_label_list, args.max_seq_length,args.batch_size,
                                                               model.word2id, perm=None, sort=False,weight=None)
        adv_acc = eval_model(model, output_list_1, true_label_list_1)
        # orig_probs_ = model.text_pred(output_list)  # tensor, [1,2]
        # pred_labels = torch.argmax(orig_probs_, dim=1)
        # if torch.sum(pred_labels==true_label_list)>0:
        #     print("gg")

        print('Adv acc:', (adv_acc, len(output_list)))


        # 加起来
        train_x.extend(output_list)
        train_y.extend(true_label_list)
        print('New train set:', len(train_x))

    elif args.mode == 'cover_array':
        output_list, true_label_list, weight_list = get_ca_arr(model)
        print('Cover array:', len(output_list))
        output_list = [[inv_full_dict[w] for w in x] for x in output_list]  # 网络输入是词语

        # random.seed(333)
        # random.shuffle(output_list)
        # random.seed(333)
        # random.shuffle(true_label_list)
        # output_list = output_list[:700]
        # true_label_list = true_label_list[:700]

        # output_list_1 = output_list.copy()
        # true_label_list_1 = true_label_list.copy()
        # output_list_1, true_label_list_1 = dataloader.create_batches(output_list_1, true_label_list_1, args.batch_size, model.word2id, )
        #
        # arr_acc = eval_model(model, output_list_1, true_label_list_1)
        # print('Cover array acc originally:' , arr_acc)


        # # 加起来
        # # x
        # train_x.extend(output_list)
        # # weight
        # fin_weight_list = [1] * len(train_y)
        # fin_weight_list.extend(weight_list)
        # # y
        # train_y.extend(true_label_list)
        # print('New train set:', len(train_x))

        # 加起来
        # x
        train_x = output_list + train_x
        # weight
        fin_weight_list = [1.0] * len(train_y)
        fin_weight_list = weight_list + fin_weight_list
        # y
        train_y = true_label_list + train_y
        print('New train set:', len(train_y))


    # 4. 重新训练
    print('Train...')
    # random.seed(333)
    # random.shuffle(train_x)
    # random.seed(333)
    # random.shuffle(train_y)
    if args.mode == 'cover_array':
        random.seed(333)
        # random.shuffle(fin_weight_list)
        train_x, train_y, train_weight = dataloader.create_batches(train_x, train_y, args.max_seq_length, args.batch_size, model.word2id, perm=None, sort=False, weight=fin_weight_list)
    else:
        train_x, train_y = dataloader.create_batches(train_x, train_y, args.max_seq_length, args.batch_size, model.word2id, perm=None, sort=False)

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
    best_test = 0
    for epoch in range(args.max_epoch):
        train_acc = eval_model(model, train_x, train_y)
        sys.stdout.write("train acc: {:.6f}\n".format(train_acc))
        # best_test = train_model(epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)
        if args.mode == 'cover_array':
            # best_test = train_model1(epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path, train_weight)
            # 无浓度
            best_test = train_model(epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)

        else:
            best_test = train_model(epoch, model, optimizer, train_x, train_y, test_x, test_y, best_test, args.save_path)
        if args.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

    sys.stdout.write("After adv training, test acc: {:.6f}\n".format(best_test))
    train_acc = eval_model(model, train_x, train_y)
    sys.stdout.write("After adv training, train acc: {:.6f}\n".format(train_acc))

    # if args.mode == 'cover_array':
    #     arr_acc = eval_model(model, output_list_1, true_label_list_1)
    #     print('Cover array acc after adv training:', arr_acc)

    # 4. 测试新模型的鲁棒性

def get_ca_arr(model):
    fastca_outs_dir = 'data/adversary_training_corpora/mr/fastca_hownet/wordLSTM/train/fastca_outs'
    fastca_outs_files = os.listdir(fastca_outs_dir)
    fastca_outs_files.sort()
    train_x = dataset.train_seqs2
    train_y = dataset.train_y
    train_y = list(train_y)

    # 获得不鲁棒的训练数据
    train_robust_file = 'data/adversary_training_corpora/%s/robust_train_%s.txt' % (args.task, args.target_model)
    train_robust = open(train_robust_file, 'r').read().splitlines()
    train_robust = np.array([float(r.split(' ')[2]) for r in train_robust])
    nonrobust_id = np.where(train_robust < 1)[0].tolist()

    # 对不鲁棒训练数据获得覆盖数组替换文本
    all_replace_text = []
    weight_list = []
    true_label_list = []
    for out_file in fastca_outs_files:
        id = int(out_file.split('_')[0])
        if id not in nonrobust_id:
            continue
        pos_tag = pos_tags[id]
        ori_data = train_x[id]  # 原始数据
        true_label = train_y[id]

        # 只取出预测正确的
        ori_data_1 = ori_data.copy()
        ori_data_1 = [inv_full_dict[id] for id in ori_data_1]  # 转为str（网络输入需要词语）
        # ori_probs = model.module.text_pred([text1])
        ori_probs = model.text_pred([ori_data_1])
        ori_probs = ori_probs.squeeze()
        ori_label = torch.argmax(ori_probs).item()
        if ori_label != true_label:
            continue

        # 获得原始数据的同义词表
        text_syns = [[t] for t in ori_data]  # 保存该文本各个位置同义词（包含自己。获取CA后，用其映射回词语）
        for i in range(len(ori_data)):
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
            neigbhours = word_candidate[ori_data[i]][pos]  # 获得 当前词语 当前词性 的 替换词语候选集
            if len(neigbhours) > 0:
                text_syns[i] += neigbhours
        text_syn_flatten = [w for s in text_syns for w in s]  # 将该文本所有位置的同义词（包含自己）按顺序拉伸成一维列表
        # 转化覆盖数组
        caarrs = open(fastca_outs_dir + '/' + out_file, 'r').readlines()
        # caarrs = [[int(w) for w in data.split(' ')[:len(data)-1]] for data in caarrs]
        for text in caarrs:
            a_replace_text = []
            text = text.strip().split(' ')
            for j in text:
                a_replace_text.append(text_syn_flatten[int(j)])
            all_replace_text.append(a_replace_text)
        #  一条数据对应覆盖数组的权重
        weight = float(args.alpha) / float(len(caarrs))
        weight_list.extend([weight]*len(caarrs))
        true_label_list.extend([true_label]*len(caarrs))


    return all_replace_text, true_label_list, weight_list


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default='')
    parser.add_argument("--task", type=str, default='mr', help="task name: mr/imdb")
    parser.add_argument("--nclasses", type=int, default=2, help="How many classes for classification.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="max sequence length for BERT target model")
    parser.add_argument("--target_model", type=str, default='wordLSTM', help="For mr/imdb: wordLSTM or bert")
    parser.add_argument("--select_nonrobust", type=bool, default=False)
    parser.add_argument("--target_model_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='', help="n")
    parser.add_argument("--word_embeddings_path", type=str, default='./glove.6B/glove.6B.200d.txt',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--batch_size", "--batch", type=int, default=128)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--d", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0)
    parser.add_argument("--cv", type=int, default=0)

    args = parser.parse_args()
    # args.target_model_path = ('models/%s/%s' % (args.target_model, args.task))

    nclasses_dict = {'imdb': 2, 'mr': 2}
    args.nclasses = nclasses_dict[args.task]
    seq_len_list = {'imdb': 250, 'mr': 128}
    args.max_seq_length = seq_len_list[args.task]
    sizes = {'imdb': 50000, 'mr': 20000}
    max_vocab_size = sizes[args.task]
    with open('data/adversary_training_corpora/%s/dataset_%d.pkl' % (args.task, max_vocab_size), 'rb') as f:
        dataset = pickle.load(f)
    # with open('dataset/word_candidates_sense_top5.pkl','rb') as fp:
    #     word_candidate=pickle.load(fp)
    with open('data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
        word_candidate = pickle.load(fp)
    with open('data/adversary_training_corpora/%s/pos_tags.pkl' % args.task, 'rb') as fp:  # 针对训练集获得对抗样本
        pos_tags = pickle.load(fp)

    inv_full_dict = dataset.inv_full_dict
    full_dict = dataset.full_dict
    pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    adv_train()
    # get_ca_arr()

    # file1 = open('tttest.txt','r').read().splitlines()
    # file2 = open('data/adversary_training_corpora/mr/robust_test_wordLSTM.txt', 'r').read().splitlines()
    # for i in range(len(file1)):
    #     if int(file1[i]) != int(file2[i].split(' ')[1]):
    #         print(i)
    #         exit(0)

