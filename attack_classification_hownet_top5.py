import json

import numpy as np
import criteria
import random
import tensorflow as tf
import tensorflow_hub as hub
import time
import torch
import pickle

from train_classifier import Model
from attack_classification import NLI_infer_BERT
from gen_pos_tag import pos_tagger
from train_classifier import *
import os
from config import args

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

def get_syn_words(idx, perturb_idx, text_str):
    neigbhours_list = []
    text_str = [word.replace('\x85', '') for word in text_str]
    if ' '.join(text_str) in args.candidate_bags.keys():  # seen data (from train or test dataset)
        # 获得候选集
        candidate_bag = args.candidate_bags[' '.join(text_str)]
        for j in perturb_idx:
            neghs = candidate_bag[text_str[j]].copy()
            neghs.remove(text_str[j])  # 同义词中删掉自己
            neigbhours_list.append(neghs)
    else:
        pos_tag = pos_tagger.tag(text_str)
        # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
        text_ids = []
        for word_str in text_str:
            if word_str in args.full_dict.keys():
                text_ids.append(args.full_dict[word_str])  # id
            else:
                text_ids.append(word_str)  # str
        # 对于所有扰动位置
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

def attack(text_id, text_ls, true_label, cos_sim, predictor, stop_words_set, word2idx, idx2word, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], [text_ls]).squeeze()
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
        # start_time = time.clock()
        leave_1_probs = predictor([text_ls for i in range(len(leave_1_texts))], leave_1_texts)   # [103,2]

        # use_time = (time.clock() - start_time)
        # print("Predict for replaced texts for a text: " , use_time)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)  # [103]
        a = (leave_1_probs_argmax != orig_label).float()
        b = leave_1_probs.max(dim=-1)[0]
        c = torch.index_select(orig_probs, 0, leave_1_probs_argmax)
        d = b - c
        if len(leave_1_probs.shape) == 1: # 说明该文本只有一个单词，增加一维
            leave_1_probs = leave_1_probs.unsqueeze(0)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + a * d).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))


        perturb_idx = [idx for idx, word in words_perturb]
        synonym_words = get_syn_words(text_id, perturb_idx, text_ls)

        synonyms_all = []
        for idx, word in words_perturb:
            synonyms = synonym_words.pop(0)
            if word in word2idx:
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            # start_time = time.clock()
            new_probs = predictor([text_ls for i in range(len(new_texts))], new_texts)
            # use_time = (time.clock() - start_time)
            # print("Predict for candidate texts for a perturbed position in a text: " , use_time)

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


        # start_time = time.clock()
        attack_label = torch.argmax(predictor([text_ls for i in range(len(text_prime))], [text_prime]))
        # use_time = (time.clock() - start_time)
        # print("Predict for the final selected candidate text: " , use_time)

        return ' '.join(text_prime), num_changed, orig_label, attack_label, num_queries


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
            new_probs = predictor(new_texts)

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

    print('Whether use sym candidates: ', args.sym)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    """Get data to attack"""
    if args.train_set:
    # train set
        texts = args.datasets.train_seqs2
        texts = [[args.inv_full_dict[word] for word in text]for text in texts]  # 网络输入是词语
        labels = args.datasets.train_y
        data = list(zip(texts, labels))
    else:
    # test set
        texts = args.datasets.test_seqs2
        texts = [[args.inv_full_dict[word] for word in text]for text in texts]  # 网络输入是词语
        labels = args.datasets.test_y
        data = list(zip(texts, labels))
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


    cos_sim = None
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



    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    predict_true = 0

    outs = {}

    for idx, (text, true_label) in enumerate(data):
        # if predict_true > 200:  # 攻击前200个正确预测的
        #     break
        text = [word for word in text if word != '\x85']
        print('text id:', idx)
        if idx % 20 == 0:
            print('{} samples out of {} have been finished!'.format(idx, len(labels)))
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
            predict_true += 1
            nums_queries.append(num_queries)

        if true_label == new_label:
            print('failed! time:', adv_time)
            failed_list.append(idx)
            failed_time.append(adv_time)
            failed_input_list.append([text, true_label])
            continue

        if true_label != new_label:
            modify_ratio = 1.0 * num_changed / len(text)
            if modify_ratio > 0.25:
                continue
            # 对抗样本再输入模型判断
            probs = predictor([text], [new_text.split(' ')]).squeeze()
            pred_label = torch.argmax(probs)
            if true_label == pred_label:
                continue
            print('Success! time:', adv_time)
            print('Changed num:', num_changed)
            success_count += 1
            true_label_list.append(true_label)
            input_list.append([text, true_label])
            output_list.append(new_text)
            success.append(idx)
            change_list.append(modify_ratio)
            num_change_list.append(num_changed)  #
            success_time.append(adv_time)  #
            outs[idx] = {'label': true_label, 'text': ' '.join(text), 'modify_ratio': modify_ratio, 'adv_texts': new_text}

    # with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
    #     for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
    #         ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))
    print(predict_true, success_count, float(predict_true)/float(len(data)), float(success_count)/float(predict_true), 1.0-(success_count+wrong_clas)/float(len(data)))

    w_dir = args.output_dir
    if args.train_set:
        w_dir += '/train_set/'
    else:
        w_dir += '/test_set/'
    w_dir += args.kind
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)

    with open(w_dir + '/tf_%s_%s_ifvalid_sym%s.pkl' % (args.task, args.target_model, args.sym), 'wb') as f:
        pickle.dump((wrong_clas_id), f)
    with open(w_dir + '/tf_%s_%s_fail_sym%s.pkl' % (args.task, args.target_model, args.sym), 'wb') as f:
        pickle.dump((failed_list, failed_input_list, failed_time), f)
    with open(w_dir + '/tf_%s_%s_success_sym%s.pkl' % (args.task, args.target_model, args.sym), 'wb') as f:
        pickle.dump((input_list, true_label_list, output_list, success, change_list, num_change_list, success_time), f)
    with open(w_dir + '/tf_%s_%s_adv_sym%s.json' % (args.task, args.target_model, args.sym), 'w') as f:
        json.dump(outs, f, indent=4)

    return

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main()

