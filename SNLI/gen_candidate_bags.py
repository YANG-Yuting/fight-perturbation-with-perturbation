import numpy as np
from numpy.random import choice
import pickle


train_set = True
sym = True


with open('/pub/data/huangpei/TextFooler/SNLI/dataset/nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
with open('/pub/data/huangpei/TextFooler/SNLI/dataset/all_seqs.pkl', 'rb') as fh:
    train, _, test = pickle.load(fh)
with open('/pub/data/huangpei/TextFooler/SNLI/dataset/pos_tags_test.pkl','rb') as fp:
    pos_tags = pickle.load(fp)
true_labels = test['label']
np.random.seed(3333)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}
# tiz add oov
vocab['<oov>'] = 42391
inv_vocab[42391] = '<oov>'
pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
test_s1 = [[inv_vocab[i] for i in t[1:-1]] for t in test['s1']]
test_s2 = [[inv_vocab[i] for i in t[1:-1]] for t in test['s2']]
if sym:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/word_candidates_sense_top5_sym.pkl', 'rb') as fp:
        word_candidate = pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/word_candidates_sense_top5.pkl', 'rb') as fp:
        word_candidate = pickle.load(fp)


"""获得数据的各个位置同义词，写入文件。注：以下均针对s2"""
def gen_texts_candidates():
    if train_set:
        train_s2 = [[inv_vocab[i] for i in t[1:-1]] for t in train['s2']]
        data = train_s2
        if sym:
            outfile = '/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_train_sym.pkl'
        else:
            outfile = '/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_train.pkl'
        pos_tags_file = '/pub/data/huangpei/TextFooler/SNLI/dataset/pos_tags.pkl'
    else:
        data = test_s2
        if sym:
            outfile = '/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_test_sym.pkl'
        else:
            outfile = '/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_test.pkl'
        pos_tags_file = '/pub/data/huangpei/TextFooler/SNLI/dataset/pos_tags_test.pkl'

    # load pos_tags
    with open(pos_tags_file, 'rb') as fp:
        pos_tags = pickle.load(fp)

    # get candidate
    candidate_bags = {}
    for text_str in data:
        # 获得词性
        pos_tag = pos_tags[' '.join(text_str)].copy()
        # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
        text_ids = []
        for word in text_str:
            if word in vocab.keys():
                text_ids.append(vocab[word])  # id
            else:
                text_ids.append(word)  # str
        # 获得候选集
        candidate_bag = {}
        for j in range(len(text_ids)):  # 对于每个位置
            word = text_ids[j]
            pos = pos_tag[j][1]
            neigbhours = [word]
            if isinstance(word, int) and pos in pos_list and word < len(word_candidate):
                if pos.startswith('JJ'):
                    pos = 'adj'
                elif pos.startswith('NN'):
                    pos = 'noun'
                elif pos.startswith('RB'):
                    pos = 'adv'
                elif pos.startswith('VB'):
                    pos = 'verb'
                neigbhours.extend(word_candidate[word][pos])  # 候选集
            # 转str
            neigbhours = [inv_vocab[n] if isinstance(n, int) else n for n in neigbhours]
            candidate_bag[inv_vocab[word] if isinstance(word, int) else word] = neigbhours

        candidate_bags[' '.join(text_str)] = candidate_bag # 保存的候选集里保留特殊符号

    # write to file
    f = open(outfile, 'wb')
    pickle.dump(candidate_bags, f)


if __name__ == '__main__':
    gen_texts_candidates()

