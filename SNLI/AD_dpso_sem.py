import numpy as np
import pickle
import time
import os
from attack_dpso_sem import PSOAttack
from model_nli import Model as bdlstm_model
from encap_snli_bert import Model as bert_model
from config import args
import torch
"""
max len in s1 57
max len in s2 30
"""

def main():
    """Get data to attack"""
    if args.train_set:
    # train set
        s1s,s2s,labels = args.train_s1, args.train_s2, args.train_labels
        s1s = [[args.full_dict[w] for w in t] for t in s1s]
        s2s = [[args.full_dict[w] for w in t] for t in s2s]
        data = list(zip(s1s,s2s,labels))
    else:
    # test set
        s1s,s2s,labels = args.test_s1, args.test_s2, args.test_labels
        s1s = [[args.full_dict[w] for w in t] for t in s1s]
        s2s = [[args.full_dict[w] for w in t] for t in s2s]
        data = list(zip(s1s,s2s,labels))
        data = data[:100]
    print("Data import finished!")
    print('Attaked data size:', len(data))

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = bdlstm_model(args)
        print(args.target_model)
    elif args.target_model == 'bert':
        model = bert_model(args)
        print(args.target_model)
    print("Model built!")
    model.eval()
    predictor = model.text_pred()

    adversary = PSOAttack(args, predictor, max_iters=20, pop_size=60)

    wrong_clas_id = []
    wrong_clas = 0
    attack_list = []

    failed_list = []
    failed_time = []
    failed_input_list = []

    input_list = []
    output_list = []
    success = []
    change_list = []
    true_label_list = []
    success_count = 0
    num_change_list = []
    success_time = []

    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    new_labels = []
    predict_true = 0

    print('Start attacking!')
    for idx, (ori_s1, ori_s2, true_label) in enumerate(data):
        print('text id:', idx)
        ori_s1_str = [args.inv_full_dict[w] for w in ori_s1]
        ori_s2_str = [args.inv_full_dict[w] for w in ori_s2]
        orig_probs = predictor([ori_s2_str], ([ori_s1_str], [ori_s2_str])).squeeze()
        orig_label = torch.argmax(orig_probs)
        if orig_label != true_label:
            wrong_clas += 1
            wrong_clas_id.append(idx)
            print('wrong classifed ..')
            print('--------------------------')
            continue

        predict_true += 1
        attack_list.append(idx)
        if true_label == 2:
            target = 0
        elif true_label == 0:
            target = 2
        else:
            target = 0 if np.random.uniform() < 0.5 else 2
        time_start = time.time()
        new_text = adversary.attack(np.array(ori_s1), np.array(ori_s2), target)
        time_end = time.time()
        adv_time = time_end - time_start

        if new_text is None:
            print('failed! time:', adv_time)
            failed_list.append(idx)
            failed_time.append(adv_time)
            failed_input_list.append([ori_s1, ori_s2, true_label])
            continue

        print('-------------')
        # new_text = new_text.tolist()
        num_changed = np.sum(np.array(ori_s2) != np.array(new_text))
        print('%d changed.' % int(num_changed))
        modify_ratio = float(num_changed) / float(len(ori_s2))
        if modify_ratio > 0.25:
            continue
        new_text_str = [args.inv_full_dict[w] for w in new_text]
        probs = predictor([ori_s2_str], ([ori_s1_str], [new_text_str])).squeeze()
        pred_label = torch.argmax(probs)
        if pred_label != target:
            continue
        print('Success! time:', adv_time)
        print('Changed num:', num_changed)
        success_count += 1
        true_label_list.append(true_label)
        input_list.append([ori_s1, ori_s2, true_label])
        output_list.append(new_text)
        success.append(idx)
        change_list.append(modify_ratio)
        num_change_list.append(num_changed)  #
        success_time.append(adv_time)  #
    print(float(success_count)/float(predict_true), 1.0-(success_count+wrong_clas)/float(len(data)))


    w_dir = args.output_dir + '/sempso'
    if args.train_set:
        w_dir += '/train_set/'
    else:
        w_dir += '/test_set/'
    w_dir += args.kind
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)

    args.task = 'snli'
    if args.train_set:
        with open(w_dir + '/sem_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((wrong_clas_id), f)
        with open(w_dir + '/sem_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((failed_list, failed_input_list, failed_time), f)
        with open(w_dir + '/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)
    else:
        with open(w_dir + '/sem_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((wrong_clas_id), f)
        with open(w_dir + '/sem_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((failed_list, failed_input_list, failed_time), f)
        with open(w_dir + '/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main()

