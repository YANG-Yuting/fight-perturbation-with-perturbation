import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

task = 'fake'
vocab_size = 50000
dataset_path = 'data/adversary_training_corpora/%s' % task
word_emb_path = dataset_path + ('/embeddings_glove_%d.pkl.npy' % vocab_size)
embedding_matrix = np.load(word_emb_path)
embedding_matrix = embedding_matrix.T  # (50001, 200)

with open(dataset_path + ('/dataset_%d.pkl' % vocab_size), 'rb') as fh:
    dataset = pickle.load(fh)
with open(dataset_path + '/word_candidates_sense_all.pkl', 'rb') as fp:
    word_candidate = pickle.load(fp)
inv_full_dict = dataset.inv_full_dict
full_dict = dataset.full_dict

pos_list = ['noun', 'verb', 'adj', 'adv']
k = 5

for word_id in word_candidate.keys():
    print(word_id)
    all_syns = word_candidate[word_id]
    new_all_syns = all_syns.copy()
    for pos in pos_list:
        syn_pos = new_all_syns[pos]
        if len(syn_pos) == 0:
            continue
        else:
            ori_word_emb = embedding_matrix[word_id]  # 原始单词向量
            syn_pos_word_emb = embedding_matrix[syn_pos]
            sim = []
            for i in range(len(syn_pos)):
                a_sim = cosine_similarity([ori_word_emb, syn_pos_word_emb[i]])[0][1]
                sim.append([a_sim, syn_pos[i]])
            sim.sort(reverse=True)
            top_k = sim[:k]
            top_k_syn = [s[1] for s in top_k]
            word_candidate[word_id][pos] = top_k_syn

f = open('data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % task, 'wb')
pickle.dump(word_candidate, f)
