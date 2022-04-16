"""
attack for mr and imdb, models are wordLSTM and bert-MIT
"""

from __future__ import division
import pickle
import torch
import time
import numpy as np

from attack_dpso_sem import PSOAttack
from train_classifier import Model
from attack_classification import NLI_infer_BERT
import os
from config import args

def main():
    """Get data to attack"""
    if args.train_set:
    # train set
        texts = args.datasets.train_seqs2
        labels = args.datasets.train_y
        data = list(zip(texts, labels))
    else:
    # test set
        texts = args.datasets.test_seqs2
        labels = args.datasets.test_y
        data = list(zip(texts, labels))
        data = data[:200]
    print("Data import finished!")
    print('Attaked data size', len(data))

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args, args.max_seq_length, args.embedding, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        print('Load model from: %s' % args.target_model_path)
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args, args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
        print('Load model from: %s' % args.target_model_path)
    model.eval()
    predictor = model.text_pred()

    print('Whether use sym candidates: ', args.sym)


    pop_size = 60
    adversary = PSOAttack(args, predictor, args.word_candidate, args.datasets, max_iters=20, pop_size=pop_size)

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

    predict_true = 0

    print('Start attacking!')

    for idx, (text_ids, true_label) in enumerate(data):
        # if predict_true > 200:
        #     break
        print('text id: ', idx)

        text_str = [args.inv_full_dict[word] for word in text_ids]
        orig_probs = predictor([text_str], [text_str]).squeeze() # (2,)
        orig_label = torch.argmax(orig_probs)
        if orig_label != true_label:
            wrong_clas += 1
            wrong_clas_id.append(idx)
            print('wrong classifed ..')
            print('--------------------------')
            continue

        predict_true += 1
        attack_list.append(idx)
        target = 1 if orig_label == 0 else 0
        time_start = time.time()
        new_text = adversary.attack(np.array(text_ids), np.array(target))
        time_end = time.time()
        adv_time = time_end - time_start

        if new_text is None:
            print('failed! time:', adv_time)
            failed_list.append(idx)
            failed_time.append(adv_time)
            failed_input_list.append([text_str, true_label])
            continue

        print('-------------')
        new_text = new_text.tolist()
        num_changed = np.sum(np.array(text_ids) != np.array(new_text))
        print('%d changed.' % int(num_changed))
        modify_ratio = float(num_changed) / float(len(text_ids))
        if modify_ratio > 0.25:
            continue
        new_text = [args.inv_full_dict[ii] for ii in new_text]
        probs = predictor([text_str], [new_text]).squeeze()
        pred_label = torch.argmax(probs)
        if pred_label != target:
            continue
        print('Success! time:', adv_time)
        print('Changed num:', num_changed)

        success_count += 1
        true_label_list.append(true_label)
        input_list.append([text_str, true_label])
        output_list.append(new_text)
        success.append(idx)
        change_list.append(modify_ratio)
        num_change_list.append(num_changed)  #
        success_time.append(adv_time)  #
        print('Num of data: %d, num of predict true: %d, num of successfule attacl:%d' % (idx+1, predict_true, success_count))

    print(float(success_count)/float(predict_true), 1.0-(success_count+wrong_clas)/float(idx+1))

    w_dir = args.output_dir + '/sempso'
    if args.train_set:
        w_dir += '/train_set/'
    else:
        w_dir += '/test_set/'
    w_dir += args.kind
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main()






