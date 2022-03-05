import pickle
task = 'imdb'
with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % task, 'rb') as fh:
    word_candidates = pickle.load(fh)
non_sym_count = 0
for word in word_candidates.keys():
    candii =word_candidates[word]
    for pos in candii.keys():
        neighbors = candii[pos]
        if len(neighbors) > 0:
            for nei in neighbors:
                nei_candii = word_candidates[nei]
                if word not in nei_candii[pos]:
                    word_candidates[nei][pos].append(word)
                    non_sym_count += 1
print('Find non-sym word: ', non_sym_count)

f = open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/word_candidates_sense_top5_sym.pkl' % task, 'wb')
pickle.dump(word_candidates, f)
