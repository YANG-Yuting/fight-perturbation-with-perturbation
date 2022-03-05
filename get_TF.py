from sklearn.feature_extraction.text import TfidfVectorizer
import dataloader
import json
from collections import defaultdict
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import tqdm
if __name__ == '__main__':
    task = 'imdb'
    # train_x, train_y = dataloader.read_corpus('data/adversary_training_corpora/mr/train.txt', clean=False, FAKE=False, shuffle=True)
    if task == 'mr':
        with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/mr/dataset_20000.pkl', 'rb') as f:
            datasets = pickle.load(f)
    elif task == 'imdb':
        with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/imdb/dataset_50000.pkl', 'rb') as f:
            datasets = pickle.load(f)
    train_x = datasets.train_seqs2
    train_x = [[datasets.inv_full_dict[word] for word in text] for text in train_x]
    train_y = datasets.train_y

    corpus = []
    for text in train_x:
        text = ' '.join(text)
        corpus.append(text)

    # vectorizer = TfidfVectorizer(ngram_range=(2,2))
    # tfidftransformer = vectorizer.fit_transform(corpus)

    # cv = CountVectorizer(ngram_range=(1,2))  # 创建词袋数据结构
    # cv_fit = cv.fit_transform(corpus)
    # a = cv.get_feature_names()  # ['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典
    # b = cv.vocabulary_
    # c = b['go to']
    # d = b['go']

    #
    # tf_vocabulary = pickle.load(open('data/adversary_training_corpora/mr/tf_vocabulary.pkl', "rb"))
    # print(tf_vocabulary['go to'])

    # corpus = ["dogs cat fish", "dog cat cat", "fish bird", 'bird']

    # new 1-2 gram
    cv = CountVectorizer(ngram_range=(1, 2))  # 创建词袋数据结构
    cv_fit = cv.fit_transform(corpus)
    tf_vocabulary = {}
    vocab = cv.get_feature_names()  # 所有n gram
    tf_matrix = cv_fit.toarray()
    # 统计n gram总数
    num_1gram, num_2gram = 0.0, 0.0
    for vo in vocab:
        if len(vo.split(' ')) == 1:
            num_1gram += 1
        elif len(vo.split(' ')) == 2:
            num_2gram += 1
    # 统计各n gram出现次数
    i=0
    for text in tf_matrix:
        i+=1
        print(i)
        for word in range(len(text)):
            word_str = vocab[word]
            if word_str in tf_vocabulary.keys():
                tf_vocabulary[word_str] += text[word]
            else:
                tf_vocabulary[word_str] = text[word]
    # 将出现次数转换为频率
    for g in tf_vocabulary.keys():
        if len(g.split(' ')) == 1:
            tf_vocabulary[g] = float(tf_vocabulary[g])/num_1gram
        elif len(g.split(' ')) == 2:
            tf_vocabulary[g] = float(tf_vocabulary[g])/num_2gram

    # print(tf_vocabulary)
    with open('data/adversary_training_corpora/%s/tf_vocabulary.pkl' % task, 'wb') as fw:
        pickle.dump(tf_vocabulary, fw)

