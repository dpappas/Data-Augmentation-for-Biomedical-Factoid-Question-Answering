
from transformers import AutoTokenizer, AutoModel
import zipfile, json, pickle, random, os
from tqdm import tqdm
from pprint import pprint
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as sklearn_auc
import numpy as np
import argparse
import xlrd

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

sws = stopwords.words('english')

import logging, re

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

def pre_rec_auc(target, preds):
    # Data to plot precision - recall curve
    precision, recall, thresholds = precision_recall_curve(target, preds)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = sklearn_auc(recall, precision)
    # print(auc_precision_recall)
    return auc_precision_recall

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

def save_checkpoint(epoch, model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch':            epoch,
        'state_dict':       model.state_dict(),
        'optimizer':        optimizer.state_dict(),
        'scheduler':        scheduler.state_dict(),
    }
    torch.save(state, filename)

def load_model_from_checkpoint(resume_from):
    global optimizer, lr_scheduler
    if os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        my_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    else:
        print("=> could not find path !!! '{}'".format(resume_from))

def load_excel_data(ret, excel_path):
    book    = xlrd.open_workbook(excel_path)
    sh      = book.sheet_by_index(0)
    for rx in range(sh.nrows):
        row_data    = [t.value for t in sh.row(rx)]
        question    = row_data[0]
        snippet     = row_data[1]
        exact_ans   = row_data[2]
        try:
            ret[(question, snippet)].append(exact_ans)
        except:
            ret[(question, snippet)] = [exact_ans]
    return ret

def load_data(train_path, keep_only, augment_with):
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
    augment_data    = []
    ##################################################################################################
    if keep_only is not None:
        augment_data        = [t for t in train_data if t[-1] != 'factoid_snippet' and t[-1] in keep_only]
        train_data          = [t for t in train_data if t[-1] == 'factoid_snippet']
    ##################################################################################################
    if augment_with is not None:
        if 't5_bioasq' in augment_with:
            aug_1   = {}
            load_excel_data(aug_1, '/home/dpappas/bioasq_factoid/t5_bioasq_snippets_only.xlsx').items()
            aug_1   = [(k[0], v, k[1], 't5_augment') for (k, v) in aug_1.items()]
            augment_data.extend(aug_1)
        if 't5' in augment_with:
            # aug_1 = dict(
            #     list(load_excel_data(augment_data, '/home/dpappas/t5_multitask_api/questions_3.xls').items())+
            #     list(load_excel_data(augment_data, '/home/dpappas/t5_multitask_api/questions_4.xls').items())+
            #     list(load_excel_data(augment_data, '/home/dpappas/t5_multitask_api/questions_5.xls').items())
            # )
            aug_1   = {}
            load_excel_data(aug_1, '/home/dpappas/t5_multitask_api/questions_3.xls').items()
            load_excel_data(aug_1, '/home/dpappas/t5_multitask_api/questions_4.xls').items()
            load_excel_data(aug_1, '/home/dpappas/t5_multitask_api/questions_5.xls').items()
            load_excel_data(aug_1, '/home/dpappas/t5_multitask_api/questions_6.xls').items()
            aug_1   = [(k[0], v, k[1], 't5_augment') for (k, v) in aug_1.items()]
            augment_data.extend(aug_1)
        if 'translation_l1' in augment_with:
            aug_1   = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_translated_train_data_new.p','rb'))
            aug_1   = [t for t in aug_1 if t[-1] in ['transl_q_fr', 'transl_c_fr']]
            print('Total translate: {}'.format(len(aug_1)))
            print('Keeping: {}'.format(len(aug_1)))
            augment_data.extend([list(t)[:4] for t in aug_1])
        if 'translation_l3' in augment_with:
            # aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_translated_train_data.p','rb'))
            # augment_data.extend([list(t)+['translate_augment'] for t in aug_1])
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_translate_all_ion.p','rb'))
            # aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_translated_train_data_new.p','rb'))
            print('Total translate: {}'.format(len(aug_1)))
            ttt     = set()
            kept    = []
            for t in aug_1:
                temp    = list(t)[:4]
                k       = str(temp[:3])
                if k not in ttt:
                    kept.append(temp)
                    ttt.add(k)
            print('Keeping: {} unique from {}'.format(len(kept), len(aug_1)))
            del(ttt)
            augment_data.extend(kept)
        if 'retrieval' in augment_with:
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_ir_train_data.p','rb'))
            augment_data.extend([list(t) for t in aug_1])
        if 'nli_w2v' in augment_with:
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_nli_train_data.p','rb'))
            augment_data.extend([list(t) for t in aug_1])
        if 'nli_phrase' in augment_with:
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_nli_phrase_train_data.p', 'rb'))
            augment_data.extend([list(t) for t in aug_1])
        if 'w2v_embed' in augment_with:
            # aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_w2v_embed_train_data.p', 'rb'))
            aug_1       = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_w2v_embed_train_data_new.p', 'rb'))
            print('Total w2v: {}'.format(len(aug_1)))
            data_aug    = [list(t) for t in aug_1]
            if 'best' in prefix:
                data_aug = data_aug[:10000]
            print('Keeping: {}'.format(len(data_aug)))
            augment_data.extend(data_aug)
        if 'phrase_embed' in augment_with:
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_phrase_train_data_all.p', 'rb'))
            augment_data.extend([list(t) for t in aug_1])
        if 'nli_bertLM' in augment_with:
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_nli_bertLM_train_data.p', 'rb'))
            augment_data.extend([list(t) for t in aug_1])
        if 'bert_LM' in augment_with:
            # aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_bertLM_train_data_all.p', 'rb'))
            aug_1 = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_bertLM_train_data_all_new.p', 'rb'))
            augment_data.extend([list(t) for t in aug_1])
        if 'biomrc' in augment_with:
            aug_1       = pickle.load(open('/home/dpappas/bioasq_factoid/pubmed_factoid_biomrc_train_data.p', 'rb'))
            print('Total biomrc: {}'.format(len(aug_1)))
            data_aug    = [list(t) for t in aug_1]
            if 'best' in prefix:
                data_aug = data_aug[:10000]
            print('Keeping: {}'.format(len(data_aug)))
            augment_data.extend(data_aug)
    return train_data, augment_data

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

parser = argparse.ArgumentParser()
parser.add_argument("--seed",               type=int,   default=1,      nargs="?",  help="Random seed.",            required=False)
parser.add_argument("--batch_size",         type=int,   default=16,     nargs="?",  help="Batch size.",             required=False)
parser.add_argument("--warmup",             type=int,   default=0,      nargs="?",  help="Warmup steps.",           required=False)
parser.add_argument("--total_epochs",       type=int,   default=50,     nargs="?",  help="Total epochs.",           required=False)
parser.add_argument("--patience",           type=int,   default=5,      nargs="?",  help="patience.",               required=False)
parser.add_argument("--hidden_dim",         type=int,   default=100,    nargs="?",  help="Hidden dimensions.",      required=False)
parser.add_argument("--lr",                 type=float, default=5e-5,   nargs="?",  help="Learning rate.",          required=False)
parser.add_argument("--train_path",         type=str,                               help="Train path.",             required=True)
parser.add_argument("--dev_path",           type=str,                               help="Dev path.",               required=True)
parser.add_argument("--keep_only",          type=str,                               help="Keep Only.",              required=True)
parser.add_argument("--augment_with",       type=str,   default=None,               help="Augment with.",           required=False)
parser.add_argument("--how_many_aug",       type=int,   default=10000,  nargs="?",  help="How many AUG instances.", required=False)
parser.add_argument("--augment_strategy",   type=str,   default="separate",         help="Augment Strategy.",       required=False)
parser.add_argument("--prefix",             type=str,                               help="Prefix.",                 required=False)
parser.add_argument("--model_name",         type=str, default="ktrapeznikov/albert-xlarge-v2-squad-v2", help="Prefix.",     required=False)
parser.add_argument("--transformer_size",   type=int, default=2048,                 help="Prefix.",                 required=False)
parser.add_argument("--monitor",            type=str, default='auc',                help="loss OR auc.",            required=False)

args                = parser.parse_args()

batch_size          = args.batch_size
lr                  = args.lr
hidden_nodes        = args.hidden_dim
warmup_steps        = args.warmup
patience            = args.patience
total_epochs        = args.total_epochs
my_seed             = args.seed
train_path          = args.train_path   # '/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data.p'
dev_path            = args.dev_path     #  '/home/dpappas/bioasq_factoid/pubmed_factoid_extracted_data_dev.p'
how_many_aug        = int(args.how_many_aug)
augment_strategy    = args.augment_strategy.lower().strip() # separate ORR combined
transformer_size    = int(args.transformer_size)
monitor             = args.monitor
if args.keep_only is None or len(args.keep_only.strip()) == 0:
    keep_only       = None
else:
    keep_only       = set([t.strip() for t in args.keep_only.split(',')]) # {'factoid_snippet'}
if args.augment_with is None or len(args.augment_with.strip()) == 0:
    augment_with    = None
else:
    augment_with    = set([t.strip() for t in args.augment_with.split(',')]) # {'factoid_snippet'}

print('augment_with:')
print(augment_with)

prefix              = args.prefix       # 'albert'

random.seed(my_seed)
torch.manual_seed(my_seed)

use_cuda            = torch.cuda.is_available()
# device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
all_devices         = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

if len(all_devices)>1:
    bert_device = torch.device("cuda:0")
    rest_device = torch.device("cuda:1")
elif len(all_devices)==1:
    bert_device = torch.device("cuda:0")
    rest_device = torch.device("cuda:0")
else:
    rest_device = torch.device("cpu")
    bert_device = torch.device("cpu")


print('DEVICE:')
print((use_cuda,bert_device))
print((use_cuda,rest_device))
pprint(all_devices)


model_name          = args.model_name
bert_tokenizer 	    = AutoTokenizer.from_pretrained(model_name)
pprint(bert_tokenizer.special_tokens_map)
bert_model 		    = AutoModel.from_pretrained(model_name).to(bert_device)
bert_model.eval()
for param in bert_model.parameters():
    param.requires_grad = False


train_data, augment_data    = load_data(train_path, keep_only, augment_with)
print('LEN OF AUG DATA : {}'.format(len(augment_data)))
dev_data, _                 = load_data(dev_path, {'factoid_snippet'}, None)

random.shuffle(train_data)
random.shuffle(augment_data)
if how_many_aug>0:
    augment_data            = augment_data[:how_many_aug]
    print('KEEPING ONLY {} OF AUG DATA'.format(len(augment_data)))
else:
    print('keeping all augmented : {}'.format(len(augment_data)))


num_training_steps          = total_epochs * (len(train_data+augment_data) // batch_size)

my_model                    = Ontop_Modeler(transformer_size, hidden_nodes).to(rest_device)
optimizer                   = optim.AdamW(my_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
lr_scheduler                = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

def train_one(the_data):
    my_model.train()
    optimizer.zero_grad()
    #########################
    overall_losses  = []
    steps           = 0
    pbar            = tqdm(the_data)
    for qtext, exact_answers, snip, _ in pbar:
        sent_ids        = prep_bpe_data_text(snip.lower())[1:]
        quest_ids       = prep_bpe_data_text(qtext.lower())
        ##########################################################
        if(len(quest_ids+sent_ids)>512):
            continue
        ##########################################################
        bert_input      = torch.tensor([quest_ids+sent_ids]).to(bert_device)
        ##########################################################
        bert_out        = bert_model(bert_input)[0]
        ##########################################################
        y               = my_model(bert_out.to(rest_device))
        y               = torch.sigmoid(y)
        ##########################################################
        target_B        = torch.FloatTensor([0] * len(sent_ids)).to(rest_device)
        target_E        = torch.FloatTensor([0] * len(sent_ids)).to(rest_device)
        ##########################################################
        for ea in exact_answers:
            if len(ea) == 0:
                continue
            ea_ids      = prep_bpe_data_text(ea.lower())[1:-1]
            for b, e in find_sub_list(ea_ids, sent_ids):
                target_B[b] = 1
                target_E[e] = 1
        if (sum(target_B) == 0):
            continue
        if (sum(target_E) == 1):
            continue
        ##########################################################
        begin_y     = y[0, -len(sent_ids):, 0]
        end_y       = y[0, -len(sent_ids):, 1]
        ##########################################################
        loss_begin  = my_model.loss(begin_y, target_B)
        loss_end    = my_model.loss(end_y, target_E)
        overall_loss = (loss_begin + loss_end) / 2.0
        overall_losses.append(overall_loss)
        steps       += 1
        ##########################################################
        if(len(overall_losses) >= batch_size):
            cost_ = sum(overall_losses) / float(len(overall_losses))
            cost_.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            overall_losses = []
            pbar.set_description('{}'.format(round(cost_.cpu().item(), 4)))
        if(steps%1000 == 0):
            print(40 * '=')
            print(qtext)
            print(snip)
            for i in range(len(sent_ids)):
                print(
                    '\t'.join(
                        [
                            str(t) for t in
                            [
                                '{:18s}'.format(bert_tokenizer.convert_ids_to_tokens(sent_ids[i])),
                                sent_ids[i],
                                int(target_B[i].cpu().item()),
                                round(begin_y[i].cpu().item(), 2),
                                int(target_E[i].cpu().item()),
                                round(end_y[i].cpu().item(), 2)
                            ]
                        ]
                    )
                )
            print(40 * '=')
        ##########################################################
    ################################################
    if (len(overall_losses) > 0):
        cost_ = sum(overall_losses) / float(len(overall_losses))
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description('{}'.format(round(cost_.cpu().item(), 4)))

def eval_one(the_data):
    gb = my_model.eval()
    with torch.no_grad():
        #########################
        aucs, prerec_aucs, overall_losses, f1s = [], [], [], {}
        pbar = tqdm(the_data)
        for qtext, exact_answers, snip, _ in pbar:
            sent_ids = prep_bpe_data_text(snip.lower())[1:]
            quest_ids = prep_bpe_data_text(qtext.lower())
            ##########################################################
            if (len(quest_ids + sent_ids) > 512):
                continue
            ##########################################################
            bert_input = torch.tensor([quest_ids + sent_ids]).to(bert_device)
            ##########################################################
            bert_out = bert_model(bert_input)[0]
            ##########################################################
            y = my_model(bert_out.to(rest_device))
            y = torch.sigmoid(y)
            ##########################################################
            target_B = torch.FloatTensor([0] * len(sent_ids)).to(rest_device)
            target_E = torch.FloatTensor([0] * len(sent_ids)).to(rest_device)
            ##########################################################
            for ea in exact_answers:
                ea_ids = prep_bpe_data_text(ea.lower())[1:-1]
                for b, e in find_sub_list(ea_ids, sent_ids):
                    target_B[b] = 1
                    target_E[e] = 1
            if (sum(target_B) == 0):
                continue
            if (sum(target_E) == 1):
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
           'DEV:' + ' '.join(
                [
                    str(np.average(overall_losses)),
                    str(np.average(aucs)),
                    str(np.average(prerec_aucs))
                ] + [str(np.average(f1s[thr])) for thr in f1s]
            )
        )
        ###########################################################################
        if monitor == 'auc':
            return np.average(prerec_aucs)
        else:
            return np.average(overall_losses)

if augment_strategy == 'separate':
    if len(augment_data)>0:
        best_dev = None
        patience_ = patience
        for epoch in range(0, total_epochs):
            train_one(augment_data)
            dev_score = eval_one(dev_data)
            if best_dev is None or (
                (monitor == 'auc' and dev_score > best_dev)
                or
                (monitor == 'loss' and dev_score < best_dev)
            ):
                save_checkpoint(
                    epoch, my_model, optimizer, lr_scheduler,
                    filename='{}_{}_MLP_{}_{}_{}_AUG.pth.tar'.format(
                        prefix,
                        model_name.replace(os.path.sep, '__'),
                        hidden_nodes,
                        epoch,
                        str(lr).replace('.','p')
                    )
                )
                best_dev = dev_score
                patience_ = patience
            else:
                patience_ -= 1
                if patience_ == 0:
                    break
    #######################################
    best_dev = None
    patience_ = patience
    for epoch in range(0, total_epochs):
        train_one(train_data)
        dev_score = eval_one(dev_data)
        if best_dev is None or (
                (monitor == 'auc' and dev_score > best_dev)
                or
                (monitor == 'loss' and dev_score < best_dev)
            ):
            save_checkpoint(
                epoch, my_model, optimizer, lr_scheduler,
                filename='{}_{}_MLP_{}_{}_{}_AFTERAUG.pth.tar'.format(
                    prefix,
                    model_name.replace(os.path.sep, '__'),
                    hidden_nodes,
                    epoch,
                    str(lr).replace('.','p')
                )
            )
            best_dev = dev_score
            patience_ = patience
        else:
            patience_ -= 1
            if patience_ == 0:
                break
else:
    augment_data = augment_data+train_data
    random.shuffle(augment_data)
    #######################################
    best_dev = None
    patience_ = patience
    for epoch in range(0, total_epochs):
        train_one(augment_data)
        dev_score = eval_one(dev_data)
        if best_dev is None or (
                (monitor == 'auc' and dev_score > best_dev)
                or
                (monitor == 'loss' and dev_score < best_dev)
            ):
            save_checkpoint(
                epoch, my_model, optimizer, lr_scheduler,
                filename='{}_{}_MLP_{}_{}_{}_AUG.pth.tar'.format(
                    prefix,
                    model_name.replace(os.path.sep, '__'),
                    hidden_nodes,
                    epoch,
                    str(lr).replace('.','p')
                )
            )
            best_dev = dev_score
            patience_ = patience
        else:
            patience_ -= 1
            if patience_ == 0:
                break
    #######################################

