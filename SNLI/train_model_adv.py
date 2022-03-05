# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model_nli.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=int, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--sememe_dim", type=int, default=300, help="encoder sememe dimension")

# gpu
parser.add_argument("--gpu_id", type=int, default=5, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)
device = torch.device("cuda")
#print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
import pickle
f = open('dataset/all_seqs.pkl','rb')
train, valid, test = pickle.load(f)
word_vec = pickle.load(open('dataset/word_vec.pkl', 'rb'))

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder', 'LSTM_sememe', 'BILSTM_sememe']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model)
#nli_net.emb_sememe.weight.data.copy_(emb_s)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='sum')

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
loss_fn.cuda()


"""
TEST
"""
def test_model():
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    s1 = test['s1']
    s2 = test['s2']
    target = test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

    # save model
    test_acc = 100 * float(correct) / len(s1)

    print('Test accuracy:', test_acc)

    return test_acc


"""
EVALUATE
"""

"""向训练集中加入对抗样本"""
def get_adv_examples():
    f = open('dataset/AD_dpso_sem.pkl', 'rb')
    input_list, test_list, true_label_list, output_list, success, change_list, target_list = pickle.load(f)
    for i in range(len(input_list)):
        adv_s1 = input_list[i][0]
        train['s1'].append(adv_s1)
        adv_s2 = output_list[i]
        train['s2'].append(adv_s2)
        adv_true_label = input_list[i][2]
        train['label'].append(adv_true_label)
    print('Original train data size:', len(train['s1']))
    print('After adding adversial examples, train data size:', len(train['s1']))

    return None

"""
TRAIN
"""
def train_model():
    val_acc_best = -1e10
    adam_stop = False
    stop_training = False
    lr = optim_params['lr'] if 'sgd' in params.optimizer else None

    train['label'] = np.array(train['label'])
    valid['label'] = np.array(valid['label'])
    test['label'] = np.array(test['label'])

    """
    Train model on Natural Language Inference task
    """

    def trainepoch(epoch):
        print('\nTRAINING : Epoch ' + str(epoch))
        nli_net.train()
        all_costs = []
        logs = []
        words_count = 0

        last_time = time.time()
        correct = 0.
        # shuffle the data
        permutation = np.random.permutation(len(train['s1']))
        s1 = train['s1'][permutation]
        s2 = train['s2'][permutation]
        target = train['label'][permutation]

        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0][
                                              'lr'] * params.decay if epoch > 1 and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']
        print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

        for stidx in range(0, len(s1), params.batch_size):
            # prepare batch
            s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec, params.word_emb_dim)
            s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec, params.word_emb_dim)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
            k = s1_batch.size(1)  # actual batch size

            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()
            assert len(pred) == len(s1[stidx:stidx + params.batch_size])

            # loss
            loss = loss_fn(output, tgt_batch)
            all_costs.append(loss.item())
            words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping (off by default)
            shrink_factor = 1
            total_norm = 0

            for p in nli_net.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        p.grad.data.div_(k)  # divide by the actual batch size
                        total_norm += (p.grad.data.norm() ** 2).item()
            total_norm = np.sqrt(total_norm)

            if total_norm > params.max_norm:
                shrink_factor = params.max_norm / total_norm
            current_lr = optimizer.param_groups[0]['lr']  # current lr (no external "lr", for adam)
            optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

            # optimizer step
            optimizer.step()
            optimizer.param_groups[0]['lr'] = current_lr

            if len(all_costs) == 100:
                logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                    stidx, round(np.mean(all_costs), 2),
                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)),
                    100. * float(correct) / (stidx + k)))
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []
        train_acc = 100 * float(correct) / len(s1)
        print('results : epoch {0} ; mean accuracy train : {1}'
              .format(epoch, train_acc))
        return train_acc

    def val_model(epoch, val_acc_best):
        print('\nVALIDATION : Epoch {0}'.format(epoch))
        nli_net.eval()
        correct = 0.

        s1 = valid['s1']
        s2 = valid['s2']
        target = valid['label']

        for i in range(0, len(s1), params.batch_size):
            # prepare batch
            s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
            s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        val_acc = 100 * float(correct) / len(s1)
        print('Valid accuracy: ', val_acc)
        return val_acc

    epoch = 1
    while not stop_training and epoch <= params.n_epochs:
        train_acc = trainepoch(epoch)

        val_acc = val_model(epoch, val_acc_best)
        if val_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.outputmodelname))
            torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))

            val_acc_best = val_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print(
                    'Shrinking lr by : {0}. New lr = {1}'.format(params.lrshrink, optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
        epoch += 1

    # Save encoder instead of full model
    # torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))


if __name__ == '__main__':
    # tiz
    # load state dict (model before adv train)
    nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))
    # print()
    # test the original model
    test_model()  # Test accuracy: 84.42589576547232
    # get robust of original model

    # get_adv_examples()

    # # adv train
    train_model()
    #
    # # test the adv model
    # test_model()
    # # get robust of the adv model


