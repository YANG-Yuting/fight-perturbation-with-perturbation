import pickle
from nltk.tag import StanfordPOSTagger
import time
from tqdm import tqdm

jar = '/pub/data/huangpei/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
model = '/pub/data/huangpei/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')


if __name__ == '__main__':
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/all_seqs.pkl', 'rb') as fh:
        train,valid,test = pickle.load(fh)
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/nli_tokenizer.pkl', 'rb') as fh:
        tokenizer = pickle.load(fh)
    dict = {w: i for (w, i) in tokenizer.word_index.items()}
    inv_dict = {i: w for (w, i) in dict.items()}
    trains=[t[1:-1] for t in train['s2']]
    tests=[t[1:-1] for t in test['s2']]

    # POS
    test_text = [[inv_dict[t] for t in tt] for tt in tests]
    all_pos_tags = {}
    a = pos_tagger.tag_sents(test_text)
    for i in range(len(test_text)):
        text = ' '.join(test_text[i])
        all_pos_tags[text] = a[i]
        if text != ' '.join([b[0] for b in a[i]]):
            print(i)
            exit(0)

    f = open('/pub/data/huangpei/TextFooler/SNLI/dataset/pos_tags_test.pkl', 'wb')
    pickle.dump(all_pos_tags, f)
    f.close()
