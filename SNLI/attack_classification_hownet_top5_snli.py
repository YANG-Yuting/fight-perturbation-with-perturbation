import argparse
import numpy as np
import criteria
import pickle
import time
import tensorflow_hub as hub
import tensorflow as tf

from model_nli import Model as bdlstm_model
from encap_snli_bert import Model as bert_model
import torch
from config import args
import os
from gen_pos_tag import pos_tagger

def get_syn_words(idx, perturb_idx, text_str):
    neigbhours_list = []
    text_str = [word.replace('\x85', '') for word in text_str]
    if ' '.join(text_str) in args.candidate_bags.keys():  # seen data (from train or test dataset)
        candidate_bag = args.candidate_bags[' '.join(text_str)]
        for j in perturb_idx:
            neghs = candidate_bag[text_str[j]].copy()
            neghs.remove(text_str[j])
            neigbhours_list.append(neghs)
    else:
        pos_tag = pos_tagger.tag(text_str)
        text_ids = []
        for word_str in text_str:
            if word_str in args.full_dict.keys():
                text_ids.append(args.full_dict[word_str])  # id
            else:
                text_ids.append(word_str)  # str
        for j in perturb_idx:
            word = text_ids[j]
            pos = pos_tag[j][1]  # 当前词语词性
            if isinstance(word, int) and pos in args.pos_list:
                if pos.startswith('JJ'):
                    pos = 'adj'
                elif pos.startswith('NN'):
                    pos = 'noun'
                elif pos.startswith('RB'):
                    pos = 'adv'
                elif pos.startswith('VB'):
                    pos = 'verb'
                neigbhours_list.append(args.word_candidate[word][pos].copy())  # 候选集
            else:
                neigbhours_list.append([])
        neigbhours_list = [[args.inv_full_dict[i] if isinstance(i, int) else i for i in position] for position in neigbhours_list]  # id转为str

    return neigbhours_list


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


def attack(text_id, s1, s2, true_label, predictor, stop_words_set, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([s2], ([s1], [s2])).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(s2)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        # s2_str = [args.inv_vocab[i] for i in s2_id]
        pos_ls = criteria.get_pos(s2)

        # get importance score
        leave_1_texts = [s2[:ii] + ['<oov>'] + s2[min(ii + 1, len_text):] for ii in range(len_text)]
        # leave_1_ids = [[args.vocab[i] for i in tex] for tex in leave_1_texts]
        leave_1_probs = predictor([s2]*len_text, ([s1]*len_text, leave_1_texts))
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)  #[103]

        a = (leave_1_probs_argmax != orig_label).float()
        b = leave_1_probs.max(dim=-1)[0]
        c = torch.index_select(orig_probs, 0, leave_1_probs_argmax)
        d = b-c
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and s2[idx] not in stop_words_set:
                    words_perturb.append((idx, s2[idx]))
            except:
                print(idx, len(s2), import_scores.shape, s2, len(leave_1_texts))


        perturb_idx = [idx for idx, word in words_perturb]
        synonym_words = get_syn_words(text_id, perturb_idx, s2)

        synonyms_all = []
        for idx, word in words_perturb:
            synonyms = synonym_words.pop(0)  # tiz
            if word in args.full_dict.keys():
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = s2[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            # new_probs = predictor(new_texts, batch_size=batch_size)
            # new_ids = [[args.vocab[i] for i in tex] for tex in new_texts]
            new_probs = predictor([s2]*len(new_texts), ([s1]*len(new_texts), new_texts))

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
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy((semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        # text_prime_ids = [args.vocab[i] for i in text_prime]
        attack_label = torch.argmax(predictor([s2], ([s1], [text_prime])))

        return ' '.join(text_prime), num_changed, orig_label,attack_label, num_queries

def main():
    use = USE(args.USE_cache_path)

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



    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    new_labels = []

    """Get data to attack"""
    if args.train_set:
    # train set
        s1s,s2s,labels = args.train_s1, args.train_s2, args.train_labels
        data = list(zip(s1s,s2s,labels))
    else:
    # test set
        s1s,s2s,labels = args.test_s1, args.test_s2, args.test_labels
        data = list(zip(s1s,s2s,labels))
    print("Data import finished!")
    print('Attaked data size:', len(data))

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = bdlstm_model(args)  # 模型初始化的时候回自动加载断点等
        print(args.target_model)
    elif args.target_model == 'bert':
        model = bert_model(args)  # 模型初始化的时候回自动加载断点等
        print(args.target_model)
    print("Model built!")

    predictor = model.text_pred()


    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    predict_true = 0
    for idx, (ori_s1, ori_s2, true_label) in enumerate(data):
        print('text id:', idx)

        time_start = time.time()
        new_text, num_changed, orig_label, new_label, num_queries = attack(idx, ori_s1,ori_s2, true_label, predictor,stop_words_set, sim_predictor=use,sim_score_threshold=args.sim_score_threshold,import_score_threshold=args.import_score_threshold,sim_score_window=args.sim_score_window,synonym_num=args.synonym_num,batch_size=args.batch_size)
        time_end = time.time()
        adv_time = time_end - time_start

        if true_label != orig_label:
            orig_failures += 1
            wrong_clas += 1
            wrong_clas_id.append(idx)
            print('Wrong classified!')
            continue
        else:
            predict_true += 1
            nums_queries.append(num_queries)

        if true_label == new_label:
            print('failed! time:', adv_time)
            failed_list.append(idx)
            failed_time.append(adv_time)
            failed_input_list.append([ori_s1, ori_s2, true_label])
            continue

        if true_label != new_label:
            modify_ratio = 1.0 * num_changed / float(len(ori_s2))
            if modify_ratio > 0.25:
                continue
            # 对抗样本再输入模型判断
            probs = predictor([ori_s2], ([ori_s1], [new_text.split(' ')])).squeeze()
            pred_label = torch.argmax(probs)
            if true_label == pred_label:
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
    print(predict_true, success_count, float(predict_true)/float(len(data)), float(success_count)/float(predict_true), 1.0-(success_count+wrong_clas)/float(len(data)))
    # with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
    #     for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
    #         ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

    w_dir = args.output_dir
    if args.train_set:
        w_dir += '/train_set/'
    else:
        w_dir += '/test_set/'
    w_dir += args.kind
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)

    args.task = 'snli'
    if args.train_set:
        with open(w_dir + '/tf_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((wrong_clas_id), f)
        with open(w_dir + '/tf_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((failed_list, failed_input_list, failed_time), f)
        with open(w_dir + '/tf_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)
    else:
        with open(w_dir + '/tf_%s_%s_ifvalid.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((wrong_clas_id), f)
        with open(w_dir + '/tf_%s_%s_fail.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((failed_list, failed_input_list, failed_time), f)
        with open(w_dir + '/tf_%s_%s_success.pkl' % (args.task, args.target_model), 'wb') as f:
            pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main()

