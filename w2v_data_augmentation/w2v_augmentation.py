
from nltk.corpus import stopwords
from gensim.models import FastText
from pprint import pprint
import os, pickle, re, json, random, sys
from tqdm import tqdm
from gensim.models.keyedvectors  import KeyedVectors

bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()

w2v_bin_path    = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
sws_path 		= '/home/dpappas/bioasq_all/stopwords.pkl'
inpath  		= sys.argv[1]
opath   		= sys.argv[2]


sws         = set(stopwords.words('english'))
with open(sws_path, 'rb') as f:
	stopwords = pickle.load(f)

d = pickle.load(open(inpath,'rb'))

d = [t for t in d if t[-1]=='factoid_snippet']

print(len(d))

wv = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)

w2v_aug_data        = []
already_known_sim   = {}
for qtext, answers, snip, dtype in tqdm(d):
    clean_answers   = ' '.join(bioclean_mod(' '.join(answers)))
    #####################################################################################################
    for tok in bioclean_mod(snip):
        if tok.isnumeric():
            continue
        elif tok in sws:
            continue
        elif tok in clean_answers:
            continue
        else:
            try:
                tt = already_known_sim[tok]
            except:
                try:
                    already_known_sim[tok] = [t[0] for t in wv.most_similar(tok, topn=10) if t[1] > 0.95]
                except:
                    already_known_sim[tok] = []
    #####################################################################################################
    clean_toks = bioclean_mod(snip)
    clean_snip = ' '.join(clean_toks)
    sents = [clean_snip]
    for _ in range(1000):
        s = ''
        for tok in clean_toks:
            if tok in clean_answers:
                s += tok+' '
            else:
                try:
                    cands = already_known_sim[tok] + [tok]
                    s += random.choice(cands) + ' '
                except:
                    s += tok + ' '
        s = s.strip()
        if s != clean_snip and s not in sents:
            sents.append(s)
    #####################################################################################################
    for s in sents:
        w2v_aug_data.append((qtext, answers, s, 'w2v_embed', snip))

pickle.dump(w2v_aug_data, open(opath,'wb'))


