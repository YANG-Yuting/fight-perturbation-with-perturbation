import pickle
from sklearn.feature_extraction.text import CountVectorizer
from config import args
import numpy as np
if __name__ == '__main__':
    # 50w数据直接做，内存不够。数据分三部分做
    # # train
    # null_idx = [i for i in range(len(args.train['s2'])) if len(args.train['s2'][i]) <= 2]
    # args.train['s2'] = np.delete(args.train['s2'], null_idx)
    # args.train['label'] = np.delete(args.train['label'], null_idx)
    # # 删除首尾<s>、</s>；id转str
    # args.train_s2 = [[args.inv_vocab[w] for w in t[1:-1]] for t in args.train['s2']]
    # print('the length of train cases is:', len(args.train_s2))
    #
    # train_x = args.train_s2[400000:]
    # train_y = args.train['label']
    #
    # corpus = []
    # for text in train_x:
    #     text = ' '.join(text)
    #     corpus.append(text)
    # cv = CountVectorizer(ngram_range=(1, 2))  # 创建词袋数据结构
    # cv_fit = cv.fit_transform(corpus)
    # tf_vocabulary = {}
    # vocab = cv.get_feature_names()  # 所有n gram
    # # 统计n gram总数
    # num_1gram, num_2gram = 0.0, 0.0
    # for vo in vocab:
    #     if len(vo.split(' ')) == 1:
    #         num_1gram += 1
    #     elif len(vo.split(' ')) == 2:
    #         num_2gram += 1
    # # 统计各n gram出现次数
    # i=0
    # for dd in cv_fit:
    #     i+=1
    #     print(i)
    #     for j in range(dd.nnz):
    #         word_str = vocab[dd.indices[j]]
    #         if word_str in tf_vocabulary.keys():
    #             tf_vocabulary[word_str] += dd.data[j]
    #         else:
    #             tf_vocabulary[word_str] = dd.data[j]
    # # 保留出现次数
    # print('num of 1 and 2 gram:', (num_1gram, num_2gram))
    # with open('dataset/tf_vocabulary_3.pkl', 'wb') as fw:
    #     pickle.dump(tf_vocabulary, fw)


    # 合并三个文件
    tf_vocabulary = {}
    with open('dataset/tf_vocabulary_1.pkl', 'rb') as fw:
        tf1 = pickle.load(fw)
    with open('dataset/tf_vocabulary_2.pkl', 'rb') as fw:
        tf2 = pickle.load(fw)
    with open('dataset/tf_vocabulary_3.pkl', 'rb') as fw:
        tf3 = pickle.load(fw)
    for word_str in tf1.keys():
        if word_str in tf_vocabulary.keys():
            tf_vocabulary[word_str] += tf1[word_str]
        else:
            tf_vocabulary[word_str] = tf1[word_str]
    for word_str in tf2.keys():
        if word_str in tf_vocabulary.keys():
            tf_vocabulary[word_str] += tf2[word_str]
        else:
            tf_vocabulary[word_str] = tf2[word_str]
    for word_str in tf3.keys():
        if word_str in tf_vocabulary.keys():
            tf_vocabulary[word_str] += tf3[word_str]
        else:
            tf_vocabulary[word_str] = tf3[word_str]

    # 将出现次数转换为频率
    num_1gram = 18700 + 18588 + 16725
    num_2gram = 187218 + 186888 + 152821
    for g in tf_vocabulary.keys():
        if len(g.split(' ')) == 1:
            tf_vocabulary[g] = float(tf_vocabulary[g])/num_1gram
        elif len(g.split(' ')) == 2:
            tf_vocabulary[g] = float(tf_vocabulary[g])/num_2gram
    print(len(tf_vocabulary.keys()))
    with open('dataset/tf_vocabulary.pkl', 'wb') as fw:
        pickle.dump(tf_vocabulary, fw)