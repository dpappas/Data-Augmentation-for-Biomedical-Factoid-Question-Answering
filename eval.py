
from transformers import AutoTokenizer, AutoModel
import zipfile, json, pickle, random, os
from tqdm import tqdm
import numpy as np
from pprint import pprint
from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import logging, re, sys
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as sklearn_auc
from collections import Counter
from nltk.corpus import stopwords
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim",         type=int,   default=100,    nargs="?",  help="Hidden dimensions.",              required=False)
parser.add_argument("--seed",               type=int,   default=1,      nargs="?",  help="Random seed.",                    required=False)
parser.add_argument("--data_path",          type=str,   default='pubmed_factoid_extracted_data_test.p', help="Train path.", required=True)
parser.add_argument("--model_name",         type=str,   default="ktrapeznikov/albert-xlarge-v2-squad-v2", help="Prefix.",   required=False)
parser.add_argument("--transformer_size",   type=int,   default=2048,                 help="transformer_size.",             required=False)
parser.add_argument("--trained_model_path", type=str,   default='',                   help="trained_model_path.",           required=False)

args                = parser.parse_args()
my_seed             = args.seed
data_path           = args.data_path
model_name          = args.model_name
hidden_dim          = args.hidden_dim
transformer_size    = args.transformer_size
my_model_path       = args.trained_model_path

sws = stopwords.words('english')

def pre_rec_auc(target, preds):
    # Data to plot precision - recall curve
    precision, recall, thresholds = precision_recall_curve(target, preds)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = sklearn_auc(recall, precision)
    # print(auc_precision_recall)
    return auc_precision_recall

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(level=logging.ERROR)

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def prep_bpe_data_tokens(tokens):
    sent_ids    = []
    for token in tokens:
        token_ids = bert_tokenizer.encode(token)[1:-1]
        sent_ids.extend(token_ids)
    ###################################################################
    sent_ids    = [bert_tokenizer.cls_token_id] + sent_ids + [bert_tokenizer.sep_token_id]
    return sent_ids

def prep_bpe_data_text(text):
    sent_ids = bert_tokenizer.encode(text)
    return sent_ids

def rebuild_tokens_from_bpes(bert_bpes):
  _tokens = []
  for bpe in bert_bpes:
    if bpe.startswith('##') :
      _tokens[-1] = _tokens[-1]+bpe[2:]
    else:
      _tokens.append(bpe)
  return _tokens

def pull_per_tokens(bert_bpe_ids, vecs, tags):
  ################################################################
  bert_bpes = bert_tokenizer.convert_ids_to_tokens(bert_bpe_ids)
  first_sep = bert_bpes.index('[SEP]')
  ################################################################
  _tokens  = []
  _vecs    = []
  _tags    = []
  ################################################################
  for i in range(first_sep+1, len(bert_bpes)-1):
    bpe = bert_bpes[i]
    vec = vecs[i]
    tag = tags[i]
    if bpe.startswith('##') :
      _tokens[-1] = _tokens[-1]+bpe[2:]
    else:
      _tokens.append(bpe)
      _vecs.append(vec)
      _tags.append(tag)
  return _tokens, _vecs, _tags

def eval_one():
    gb = my_model.eval()
    with torch.no_grad():
        #########################
        aucs, prerec_aucs, overall_losses, f1s = [], [], [], {}
        pbar = tqdm(dev_data)
        for qtext, exact_answers, snip, _ in pbar:
            sent_ids = prep_bpe_data_text(snip.lower())[1:]
            quest_ids = prep_bpe_data_text(qtext.lower())
            ##########################################################
            if (len(quest_ids + sent_ids) > 512):
                continue
            ##########################################################
            bert_input = torch.tensor([quest_ids + sent_ids]).to(device)
            ##########################################################
            bert_out = bert_model(bert_input)[0]
            ##########################################################
            y = my_model(bert_out)
            y = torch.sigmoid(y)
            ##########################################################
            target_B = torch.FloatTensor([0] * len(sent_ids)).to(device)
            target_E = torch.FloatTensor([0] * len(sent_ids)).to(device)
            ##########################################################
            for ea in exact_answers:
                ea_ids = prep_bpe_data_text(ea.lower())[1:-1]
                for b, e in find_sub_list(ea_ids, sent_ids):
                    target_B[b] = 1
                    target_E[e] = 1
            if (sum(target_B) == 0):
                continue
            if (sum(target_E) == 0):
                continue
            ##########################################################
            begin_y = y[0, -len(sent_ids):, 0]
            end_y = y[0, -len(sent_ids):, 1]
            ##########################################################
            loss_begin = my_model.loss(begin_y, target_B)
            loss_end = my_model.loss(end_y, target_E)
            overall_loss = (loss_begin + loss_end) / 2.0
            overall_losses.append(overall_loss.cpu().item())
            ##########################################################
            auc = (roc_auc_score(target_B.tolist(), begin_y.tolist()) + roc_auc_score(target_E.tolist(), end_y.tolist())) / 2.0
            aucs.append(auc)
            prerec_aucs.append((pre_rec_auc(target_B.tolist(), begin_y.tolist()) + pre_rec_auc(target_E.tolist(), end_y.tolist())) / 2.0)
            ##########################################################
            for thresh in range(1, 10):
                thr = float(thresh) / 10.0
                by = [int(tt > thr) for tt in begin_y.tolist()]
                ey = [int(tt > thr) for tt in end_y.tolist()]
                f1_1 = f1_score(target_B.tolist(), by)
                f1_2 = f1_score(target_E.tolist(), ey)
                f1 = (f1_1 + f1_2) / 2.0
                try:
                    f1s[thresh].append(f1)
                except:
                    f1s[thresh] = [f1]
        ###########################################################################
        print(
            ' '.join(
                [
                    str(np.average(overall_losses)),
                    str(np.average(aucs)),
                    str(np.average(prerec_aucs))
                ] + [
                    str(np.average(f1s[thr])) for thr in f1s
                ]
            )
        )
        ###########################################################################
        return np.average(prerec_aucs)

def load_model_from_checkpoint(resume_from):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        # print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        my_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    else:
        print("=> could not find path !!! '{}'".format(resume_from))
        exit()

class Ontop_Modeler(nn.Module):
    def __init__(self, input_size, hidden_nodes):
        super(Ontop_Modeler, self).__init__()
        self.input_size             = input_size
        self.linear1                = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2                = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss                   = nn.BCELoss()
        self.tanh                   = nn.Tanh()
    def forward(self, input_xs):
        y = self.linear1(input_xs)
        y = self.tanh(y)
        y = self.linear2(y)
        return y

random.seed(my_seed)
torch.manual_seed(my_seed)

def load_data(train_path, keep_only):
    train_data    =  pickle.load(open(train_path,'rb'))
    # data    += pickle.load(open('/home/dpappas/bioasq_factoid/nq_factoid_extracted_train_data.p','rb'))
    # data    += pickle.load(open('/home/dpappas/bioasq_factoid/squad_factoid_extracted_train_data.p','rb'))
    ##################################################################################################
    print(len(train_data))
    ##################################################################################################
    data_ = []
    for (qq, anss, context, type) in train_data:
        anss_ = []
        if(len(context)>500):
            continue
        for ans in anss:
            if(len(ans.strip())==0):
                continue
            if(len(ans.split())>4):
                continue
            if(ans.lower() in sws):
                continue
            if(ans.split()[0].lower() in ['the', 'a']):
                ans = ' '.join(ans.split()[1:])
            anss_.append(ans)
        if(len(anss_)>0):
            data_.append((qq, anss_, context, type))
    ##################################################################################################
    train_data      = data_
    print(len(train_data))
    ##################################################################################################
    pprint(Counter([t[-1] for t in train_data]))
    ##################################################################################################
    if keep_only is not None:
        train_data      = [t for t in train_data if t[-1] in keep_only]
    ##################################################################################################
    return train_data

dev_data        = load_data(data_path, {'factoid_snippet'})

use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda") if(use_cuda) else torch.device("cpu")

bert_tokenizer 	= AutoTokenizer.from_pretrained(model_name)
pprint(bert_tokenizer.special_tokens_map)
bert_model 		= AutoModel.from_pretrained(model_name).to(device)
bert_model.eval()

my_model        = Ontop_Modeler(transformer_size, hidden_dim).to(device)

load_model_from_checkpoint(my_model_path)

eval_one()
