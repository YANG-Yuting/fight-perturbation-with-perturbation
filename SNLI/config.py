class DianpingConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-chinese"
        self.max_sent_lens = 64
class SSTConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-uncased"
        self.max_sent_lens = 32
class SNLIConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-uncased"
        self.max_sent_lens = 128 #
class IMDBConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-uncased"
        self.max_sent_lens = 254
class LCQMCConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-chinese"
        self.max_sent_lens = 64

import pickle
import argparse
import numpy as np
parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--task", type=str, default='snli', help="task name: mr/imdb")
# parser.add_argument("--nclasses",type=int,default=3,help="How many classes for classification.")
parser.add_argument("--target_model",type=str,default='bdlstm',help="Target models for text classification: fasttext, charcnn, word level lstm ""For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--target_model_path",type=str,default='/pub/data/huangpei/TextFooler/SNLI/lstm',help="Target models for text classification: fasttext, charcnn, word level lstm ""For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--word_embeddings_path",type=str,default='/pub/data/huangpei/TextFooler/glove.6B/glove.6B.200d.txt',help="path to the word embeddings for the target model")
parser.add_argument("--counter_fitting_embeddings_path", type=str, default='/pub/data/huangpei/TextFooler/data/counter-fitted-vectors.txt', help="path to the counter-fitting embeddings we used to find synonyms")
parser.add_argument("--counter_fitting_cos_sim_path",type=str,default='/pub/data/huangpei/TextFooler/data/cos_sim_counter_fitting.npy',help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
parser.add_argument("--USE_cache_path",type=str,default='',help="Path to the USE encoder cache.")
parser.add_argument("--output_dir",type=str,default='/pub/data/huangpei/TextFooler/adv_results',help="The output directory where the attack results will be written.")

## Model hyperparameters
parser.add_argument("--sim_score_window",default=15,type=int,help="Text length or token number to compute the semantic similarity score")
parser.add_argument("--import_score_threshold",default=-1.,type=float,help="Required mininum importance score.")
parser.add_argument("--sim_score_threshold",default=0, type=float,help="Required minimum semantic similarity score.")
parser.add_argument("--synonym_num",default=50,type=int,help="Number of synonyms to extract")
parser.add_argument("--batch_size",default=128,type=int,help="Batch size to get prediction")
parser.add_argument("--perturb_ratio",default=0., type=float,help="Whether use random perturbation for ablation study")
parser.add_argument("--max_seq_length",default=100,type=int,help="max sequence length for BERT target model")
parser.add_argument("--save_path", type=str, default='/pub/data/huangpei/TextFooler/SNLI/savedir_adv', help='path to store trained model')

parser.add_argument("--sample_num", type=int, default=256)
parser.add_argument("--change_ratio", type=float, default=1.0, help='the percentage of changed words in a text while sampling')
parser.add_argument("--gpu_id", type=str, default='1')

parser.add_argument("--sym", type=bool, default=True, help="if use the symmetric candidate")
parser.add_argument("--train_set", action='store_true', help='whether attack train set')
parser.add_argument("--attack_robot", action='store_true', help='whether attack the enhanced model')
parser.add_argument("--kind", type=str, default='org', help='the model to be attacked')
parser.add_argument("--mode", type=str, default='eval', help='train, eval')
args = parser.parse_args()

with open('/pub/data/huangpei/TextFooler/SNLI/dataset/nli_tokenizer.pkl', 'rb') as fh:
    args.tokenizer = pickle.load(fh)
if args.sym:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/word_candidates_sense_top5_sym.pkl','rb') as fp:
        args.word_candidate=pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/word_candidates_sense_top5.pkl','rb') as fp:
        args.word_candidate=pickle.load(fp)
with open('/pub/data/huangpei/TextFooler/SNLI/dataset/all_seqs.pkl', 'rb') as fh:
    args.train, _, args.test = pickle.load(fh)

if args.train_set:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/pos_tags.pkl','rb') as fp:
        args.pos_tags = pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/pos_tags_test.pkl', 'rb') as fp:
        args.pos_tags = pickle.load(fp)

if args.sym:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_test_sym.pkl', 'rb') as fp:
        candidate_bags_test = pickle.load(fp)
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_train_sym.pkl', 'rb') as fp:
        candidate_bags_train = pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_test.pkl', 'rb') as fp:
        candidate_bags_test = pickle.load(fp)
    with open('/pub/data/huangpei/TextFooler/SNLI/dataset/candidates_train.pkl', 'rb') as fp:
        candidate_bags_train = pickle.load(fp)

args.candidate_bags = {**candidate_bags_train, **candidate_bags_test}

with open('/pub/data/huangpei/TextFooler/SNLI/dataset/tf_vocabulary.pkl', 'rb') as fp:
    args.tf_vocabulary = pickle.load(fp)
np.random.seed(3333)
args.full_dict = {w: i for (w, i) in args.tokenizer.word_index.items()}
args.inv_full_dict = {i: w for (w, i) in args.full_dict.items()}
# tiz add oov
args.full_dict['<oov>'] = 42391
args.inv_full_dict[42391] = '<oov>'
# tiz 20210827 过滤掉空的数据（7459: ['<s>','</s2>']，创建数据时把词汇表之外的删了，所以会出现空的数据。所幸只有一条），按batch预测的时候会报错，而且空数据也没有意义
# test set
null_idx = [i for i in range(len(args.test['s2'])) if len(args.test['s2'][i])<=2]
args.test['s1'] = np.delete(args.test['s1'], null_idx)
args.test['s2'] = np.delete(args.test['s2'], null_idx)
args.test['label'] = np.delete(args.test['label'], null_idx)
# 删除首尾<s>、</s>；id转str
args.test_s1 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.test['s1']]
args.test_s2 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.test['s2']]
# args.test_s1 = [t[1:-1] for t in args.test['s1']]
# args.test_s2 = [t[1:-1] for t in args.test['s2']]
args.test_labels = args.test['label']
# if args.train_set:
# train set
null_idx = [i for i in range(len(args.train['s2'])) if len(args.train['s2'][i]) <= 2]
args.train['s1'] = np.delete(args.train['s1'], null_idx)
args.train['s2'] = np.delete(args.train['s2'], null_idx)
args.train['label'] = np.delete(args.train['label'], null_idx)
# 删除首尾<s>、</s>；id转str
# args.train_s1 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.train['s1'][:100]]
# args.train_s2 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.train['s2'][:100]]
# args.train_labels = args.train['label'][:100]

args.train_s1 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.train['s1']]
args.train_s2 = [[args.inv_full_dict[w] for w in t[1:-1]] for t in args.train['s2']]
args.train_labels = args.train['label']

print('the length of test cases is:', len(args.test_s1)) # 删掉一个s2为空的之后，9824-->9823
print('the length of train cases is:', len(args.train_s1))
args.pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
