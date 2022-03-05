import os
#import nltk
import re
from collections import Counter
#from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import string
from nltk.corpus import stopwords


vocab = Counter()
with open('data/adversary_training_corpora/fake/train_tok.csv', "r", encoding="utf-8") as fp:
    for line in fp:
        line = line.strip().lower()
        ## 文档按空格分词
        tokens = line.split()
        ## 准备标点符号过滤正则
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        ## 移除每个单词的标点
        tokens = [re_punc.sub('', w) for w in tokens]
        ## 移除全部非字母组成的token
        tokens = [word for word in tokens if word.isalpha()]
        ## 移除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        ## 移除长度小于等于1的token
        tokens = [w for w in tokens if len(w) > 1]
        vocab.update(tokens)

print(len(vocab))
vocab = vocab.most_common(50000)
vocab = list(vocab)
vocab = [v[0] for v in vocab]
with open('data/adversary_training_corpora/fake/fake.vocab', 'w') as wr:
    wr.write('\n'.join(vocab))
print('wrote finished to data/adversary_training_corpora/fake/fake.vocab')