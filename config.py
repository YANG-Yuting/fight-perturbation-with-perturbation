import pickle
import argparse
import glove_utils

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", type=str, default='/pub/data/huangpei/TextFooler/glove.6B/glove.6B.200d.txt', help="word vectors")
parser.add_argument("--counter_fitting_embeddings_path",type=str,default='/pub/data/huangpei/TextFooler/data/counter-fitted-vectors.txt',help="path to the counter-fitting embeddings we used to find synonyms")
parser.add_argument("--counter_fitting_cos_sim_path",type=str,default='/pub/data/huangpei/TextFooler/data/cos_sim_counter_fitting.npy',help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
parser.add_argument("--USE_cache_path",type=str,default='',help="Path to the USE encoder cache.")
parser.add_argument("--output_dir",type=str,default='/pub/data/huangpei/TextFooler/adv_results',help="The output directory where the attack results will be written.")
parser.add_argument("--sim_score_window",default=15,type=int,help="Text length or token number to compute the semantic similarity score")
parser.add_argument("--import_score_threshold",default=-1.,type=float,help="Required mininum importance score.")
parser.add_argument("--sim_score_threshold",default=0, type=float,help="Required minimum semantic similarity score.") # 0.7
parser.add_argument("--synonym_num",default=50,type=int,help="Number of synonyms to extract")
parser.add_argument("--perturb_ratio",default=0.,type=float,help="Whether use random perturbation for ablation study")

parser.add_argument("--task", type=str, default='mr', help="task name: mr/imdb/fake")
parser.add_argument("--nclasses",type=int,default=2,help="How many classes for classification.")
parser.add_argument("--target_model",type=str,default='wordLSTM',help="Target models for text classification: fasttext, charcnn, word level lstm ""For NLI: InferSent, ESIM, bert-base-uncased")
parser.add_argument("--target_model_path",type=str,default='/pub/data/huangpei/TextFooler/models/wordLSTM/mr',help="pre-trained target model path")
parser.add_argument("--batch_size",default=128,type=int,help="Batch size to get prediction")

# for train_classifier.py
parser.add_argument("--cnn", action='store_true', help="whether to use cnn")
parser.add_argument("--lstm", action='store_true', help="whether to use lstm")
parser.add_argument("--data_path", type=str, default="/pub/data/huangpei/TextFooler/data/adversary_training_corpora/", help="where to load dataset, parent dir")
parser.add_argument("--max_epoch", type=int, default=70)
parser.add_argument("--d", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--lr_decay", type=float, default=0.0)
parser.add_argument("--cv", type=int, default=0)
parser.add_argument("--save_path", type=str, default='/pub/data/huangpei/TextFooler/models/wordLSTM/mr_new', help='path to store trained model')
parser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--mode", type=str, default='eval', help='train, eval')

# for robust.py
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument("--sample_num", type=int, default=1000)
parser.add_argument("--change_ratio", type=float, default=1.0, help='the percentage of changed words in a text while sampling')

parser.add_argument("--sym", action='store_true', help="if use the symmetric candidate")
parser.add_argument("--train_set", action='store_true',help='if attack train set')
parser.add_argument("--attack_robot", action='store_true',help='if attack robot classifier')
parser.add_argument("--kind", type=str, default='org', help='the model to be attacked')
parser.add_argument("--prompt_generate", action='store_true', help='use prompt and bert to generate candidate adversarial examples')
parser.add_argument("--prompt_level", type=str, default='word-level', help='word-level or sentence-level')
parser.add_argument("--mask_ratio", type=float, default=0.15, help='the ratio of words to be masked')
parser.add_argument("--sample_size", type=int, default=10, help='the num of candidate adversarial examples for each instance')
parser.add_argument("--topk", type=int, default=3, help='the num of candidate for each masked word')
parser.add_argument("--mask_mode", type=str, default='random', help='the way of where to mask')
parser.add_argument("--word_emb", action='store_true',help='whether add word embedding for the candidate word selection while prompting')
parser.add_argument("--gpt_generate", action='store_true',help='use gpt to write a following sentence')


args = parser.parse_args()

seq_len_list = {'imdb': 256, 'mr': 128, 'fake': 512}
args.max_seq_length = seq_len_list[args.task]
sizes = {'imdb': 50000, 'mr': 20000, 'fake': 50000}
args.max_vocab_size = sizes[args.task]

# with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % args.task,
#           'rb') as fp:  # 我们过滤的同义词表
#     args.word_candidate = pickle.load(fp)

with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/dataset_%d.pkl' % (args.task, args.max_vocab_size), 'rb') as f:
    args.datasets = pickle.load(f)

if args.sym:
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/word_candidates_sense_top5_sym.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
        args.word_candidate = pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/word_candidates_sense_top5.pkl' % args.task, 'rb') as fp:  # 我们过滤的同义词表
        args.word_candidate = pickle.load(fp)

if args.train_set:
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/pos_tags.pkl' % args.task, 'rb') as fp:
        args.pos_tags = pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/pos_tags_test.pkl' % args.task, 'rb') as fp:
        args.pos_tags = pickle.load(fp)
# load candidate for texts
if args.sym:
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/candidates_train_sym.pkl' % args.task, 'rb') as fp:
        candidate_bags_train = pickle.load(fp)
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/candidates_test_sym.pkl' % args.task, 'rb') as fp:
        candidate_bags_test = pickle.load(fp)
else:
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/candidates_train.pkl' % args.task, 'rb') as fp:
        candidate_bags_train = pickle.load(fp)
    with open('/pub/data/huangpei/TextFooler/data/adversary_training_corpora/%s/candidates_test.pkl' % args.task, 'rb') as fp:
        candidate_bags_test = pickle.load(fp)
args.candidate_bags = {**candidate_bags_train, **candidate_bags_test}

args.pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
args.inv_full_dict = args.datasets.inv_full_dict
args.full_dict = args.datasets.full_dict
args.full_dict['<oov>'] = len(args.full_dict.keys())
args.inv_full_dict[len(args.full_dict.keys())] = '<oov>'

if args.word_emb:
    args.glove_model = glove_utils.loadGloveModel(args.embedding)

