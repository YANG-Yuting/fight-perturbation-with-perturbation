import numpy as np

from SNLI_BERT import ModelTrainer
from SNLI_BERT import adjustBatchInputLen
from pytorch_transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule
from torch import nn
import torch
import config
from torch.autograd import Variable
import sys
sys.path.append('..')
from train_classifier import perturb_texts, gen_sample_multiTexts,perturb_FGWS



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.config = config.SNLIConfig()
        model = BertModel.from_pretrained(self.config.BERT_MODEL)
        self.model = ModelTrainer(model, 3)
        self.model.load_state_dict(torch.load(args.target_model_path,map_location='cuda:0'))
        # print(checkpoint)))
        self.model = self.model.cuda()
        self.inv_dict = args.inv_full_dict
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL)
        self.m = nn.Softmax(1)
        self.args = args


    def forward(self,input_x):
        assert len(input_x[0]) == len(input_x[1]), "premise and hypothesis should share the same batch lens!"
        num_instance = len(input_x[0])
        batch = dict()
        batch["inputs"] = []
        batch["labels"] = torch.zeros((num_instance,)).long()
        for i in range(len(input_x[0])):
            tokens = list()
            tokens.append(self.tokenizer.cls_token)
            for k in [0, 1]:
                add_sep = False
                if k == 0:
                    add_sep = True
                # tiz
                # for j in range(len(input_x[k][i])):
                #     #print(input_x[i], tokens)
                #     #print(type(input_x[i][j]))
                #     #print(self.dataset.inv_dict[0])
                #     # inv_dict has no padding, maybe because of keras setting
                #     if input_x[k][i][j] != 0:
                #         tokens.append(self.inv_dict[int(input_x[k][i][j])])
                # tokens = ['<s>'] + tokens
                input_x[k][i] = [w.lower() for w in input_x[k][i]] ###############
                tokens.extend(input_x[k][i])
                # tokens.extend(['</s>'])
                # tiz
                if add_sep:
                    tokens.append("[SEP]")
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            batch["inputs"].append(tokens)
        adjustBatchInputLen(batch)
        end_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        for i in range(len(input_x[0])):
            tokens = batch["inputs"][i]
            tokens.append(end_id)
        batch["inputs"] = torch.stack([torch.LongTensor(x) for x in batch['inputs']])

        loss, logits = self.model(batch)
        logits = self.m(logits[:,[1,0,2]])
        return logits

    def predict(self, input_x):
        # sess is of no use, just to tailor the ugly interface
        return self(input_x)

    def text_pred(self):
        """
        org: original predictor
        Enhance: Enhancement version
        adv:adv trained
        SAFER：SAFER
        FGWS：FGWS
        """
        call_func = getattr(self, 'text_pred_%s' % self.args.kind, 'Didn\'t find your text_pred_*.')
        return call_func

    def text_pred_org(self, ori_s2, text,batch_size=128):
        self.model.eval()
        s1 = text[0]
        s2 = text[1]
        with torch.no_grad():
            s1 = [s.tolist() if not isinstance(s, list) else s for s in s1]
            s2 = [s.tolist() if not isinstance(s, list) else s for s in s2]

            logits_all = []
            data_size = len(s1)
            for i in range(0, data_size, batch_size):
                # prepare batch
                s1_batch = s1[i: min(i + batch_size, data_size)]
                s2_batch = s2[i: min(i + batch_size, data_size)]
                # model forward
                logits = self.forward([s1_batch, s2_batch])
                logits_all.append(logits)
        return torch.cat(logits_all, dim=0)

    def text_pred_adv(self, ori_s2, text,batch_size=128):
        self.model.eval()
        s1 = text[0]
        s2 = text[1]
        with torch.no_grad():
            s1 = [s.tolist() if not isinstance(s, list) else s for s in s1]
            s2 = [s.tolist() if not isinstance(s, list) else s for s in s2]

            logits_all = []
            data_size = len(s1)
            for i in range(0, data_size, batch_size):
                # prepare batch
                s1_batch = s1[i: min(i + batch_size, data_size)]
                s2_batch = s2[i: min(i + batch_size, data_size)]
                # model forward
                logits = self.forward([s1_batch, s2_batch])
                logits_all.append(logits)
        return torch.cat(logits_all, dim=0)

    def text_pred_Enhance(self, orig_s2, text, sample_num=256, batch_size=128):
        self.model.eval()
        s1 = text[0]
        s2 = text[1]
        with torch.no_grad():
            perturbed_s2 = perturb_texts(self.args, orig_s2, s2, self.args.tf_vocabulary, change_ratio=0.2)
            Samples_s2 = gen_sample_multiTexts(self.args, orig_s2, perturbed_s2, sample_num, change_ratio=0.25)

            s1s = [s for s in s1 for i in range(sample_num)]

            Sample_probs = self.text_pred_org(orig_s2, (s1s, Samples_s2), batch_size)
            lable_mum = Sample_probs.shape[-1]
            Sample_probs = Sample_probs.view(len(s2), sample_num, lable_mum)
            probs_boost = []
            for l in range(lable_mum):
                num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
                prob = num.float() / float(sample_num)
                probs_boost.append(prob.view(len(s2), 1))
            probs_boost_all = torch.cat(probs_boost, dim=1)

        return probs_boost_all

    def text_pred_SAFER(self, orig_s2, text, sample_num=256, batch_size=128):
        self.model.eval()
        s1 = text[0]
        s2 = text[1]
        with torch.no_grad():
            Samples_s2 = gen_sample_multiTexts(self.args, orig_s2, s2, sample_num, change_ratio=1)
            s1s = [s for s in s1 for i in range(sample_num)]
            Sample_probs = self.text_pred_org(orig_s2, (s1s, Samples_s2), batch_size)
            lable_mum = Sample_probs.size()[-1]
            Sample_probs = Sample_probs.view(len(s2), sample_num, lable_mum)
            probs_boost = []
            for l in range(lable_mum):
                num = torch.sum(torch.eq(torch.argmax(Sample_probs, dim=2), l), dim=1)
                prob = num.float() / float(sample_num)
                probs_boost.append(prob.view(len(s2), 1))
            probs_boost_all = torch.cat(probs_boost, dim=1)
        return probs_boost_all

    def text_pred_FGWS(self, orig_s2, text, batch_size=128):
        self.model.eval()
        s1 = text[0]
        s2 = text[1]
        gamma1 = 0.5
        with torch.no_grad():
            perturbed_s2 = perturb_FGWS(self.args, orig_s2, s2, self.args.tf_vocabulary)
            pre_prob = self.text_pred_org(orig_s2, (s1, perturbed_s2), batch_size)
            ori_prob = self.text_pred_org(orig_s2, (s1, s2), batch_size)
            lable = torch.argmax(ori_prob, dim=1)
            index = torch.arange(len(text[1])).cuda()
            D = ori_prob.gather(1, lable.view(-1, 1)) - pre_prob.gather(1, lable.view(-1, 1)) - gamma1
            probs_boost_all = torch.where(D > 0, pre_prob, ori_prob)
            #D = D.view(-1)
            #probs_boost_all = torch.ones_like(ori_prob)
            #probs_boost_all=ori_prob.index_put((index, lable), - D)
            #probs_boost_all.index_put_((index, 1 - lable), 0.5 + D)
        return probs_boost_all

    def adjustBatchInputLen(self, batch):
        inputs = batch["inputs"]
        # length = 0
        # for item in inputs:
        #     length = max(length, len(item))
        # length = min(length, self.config.max_sent_lens)
        length = self.config.max_sent_lens

        num = len(inputs)
        for i in range(num):
            if length > len(inputs[i]):
                for j in range(length - len(inputs[i])):
                    inputs[i].append(self.tokenizer.pad_token_id)
            else:
                inputs[i] = inputs[i][:length]

    def eval_model(self, s1, s2, target):
        predictor = self.text_pred()
        ori_probs = predictor(s2,(s1,s2))
        ori_labels = torch.argmax(ori_probs, dim=1)
        input_y = torch.tensor(target).long().cuda()
        correct = ori_labels.eq(input_y).cpu().sum()
        # self.model.train()
        return correct.item() / float(len(s2))
