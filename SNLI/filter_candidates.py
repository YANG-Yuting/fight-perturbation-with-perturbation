import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('dataset/word_candidates_sense.pkl','rb') as fp:
    word_candidate = pickle.load(fp)
with open('dataset/word_vec.pkl', 'rb') as fp:
    word_vec = pickle.load(fp)  # dict


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
            ori_word_emb = word_vec[word_id]  # 原始单词向量 array
            syn_pos_word_emb = []
            for syn in syn_pos:
                syn_vec = word_vec[syn]
                syn_pos_word_emb.append(syn_vec)
            syn_pos_word_emb = np.array(syn_pos_word_emb)
            sim = []
            for i in range(len(syn_pos)):
                a_sim = cosine_similarity([ori_word_emb, syn_pos_word_emb[i]])[0][1]
                sim.append([a_sim, syn_pos[i]])
            sim.sort(reverse=True)
            top_k = sim[:k]
            top_k_syn = [s[1] for s in top_k]
            word_candidate[word_id][pos] = top_k_syn

f = open('dataset/word_candidates_sense_top5.pkl', 'wb')
pickle.dump(word_candidate, f)
