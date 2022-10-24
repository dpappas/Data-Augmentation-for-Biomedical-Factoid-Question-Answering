
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pickle, re, json, os, sys
from pprint import pprint
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

sws = set(stopwords.words('english'))

with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()

def first_alpha_is_upper(sent):
    specials = [
        '__EU__','__SU__','__EMS__','__SMS__','__SI__',
        '__ESB','__SSB__','__EB__','__SB__','__EI__',
        '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
        '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
        '__FIG','__AWID','__FUNDS'
    ]
    for special in specials:
        sent = sent.replace(special,'')
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.|[\W\s]min.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def starts_with_special(sent):
    sent    = sent.strip().lower()
    chars   = ':%@#$^&*()\\,<>?/=+-_'
    for c in chars:
        if(sent.startswith(c)):
            return True
    return False

def split_sentences2(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent) or
                not first_alpha_is_upper(sents[i+1]) or
                starts_with_special(sents[i + 1])
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n', ' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    sents = []
    for subtext in ntext.split('\n'):
        subtext = re.sub('\s+', ' ', subtext.replace('\n',' ')).strip()
        if (len(subtext) > 0):
            ss = split_sentences2(subtext)
            sents.extend([ s for s in ss if(len(s.strip())>0)])
    if(len(sents)>0 and len(sents[-1]) == 0 ):
        sents = sents[:-1]
    return sents

def fix2(qtext):
    qtext = qtext.lower()
    if(qtext.startswith('can ')):
        qtext = qtext[4:]
    if(qtext.startswith('list the ')):
        qtext = qtext[9:]
    if(qtext.startswith('list ')):
        qtext = qtext[5:]
    if(qtext.startswith('describe the ')):
        qtext = qtext[13:]
    if(qtext.startswith('describe ')):
        qtext = qtext[9:]
    if('list as many ' in qtext and 'as possible' in qtext):
        qtext = qtext.replace('list as many ', '')
        qtext = qtext.replace('as possible', '')
    if('yes or no' in qtext):
        qtext = qtext.replace('yes or no', '')
    if('also known as' in qtext):
        qtext = qtext.replace('also known as', '')
    if('is used to ' in qtext):
        qtext = qtext.replace('is used to ', '')
    if('are used to ' in qtext):
        qtext = qtext.replace('are used to ', '')
    tokenized_body  = [t for t in qtext.split() if t not in stopwords]
    tokenized_body  = bioclean_mod(' '.join(tokenized_body))
    question        = ' '.join(tokenized_body)
    return question

def fix1(qtext):
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    return question

def get_first_n_1(qtext, anss, n, max_year=2022):
    # tokenized_body  = bioclean_mod(qtext)
    # tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    # question        = ' '.join(tokenized_body)
    question = fix2(qtext)
    # print(question)
    ################################################
    bod             = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"range": {"DateCompleted": {"gte": "1900", "lte": str(max_year), "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "30%"
                            }
                        }
                    },
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "50%"
                            }
                        }
                    },
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "70%"
                            }
                        }
                    },
                    {"match_phrase": {"joint_text": {"query": question, "boost": 1}}}
                ],
                "minimum_should_match": 1,
            }
        }
    }
    #############
    bod['query']['bool']['should'] += [{"match_phrase": {"joint_text": {"query": ans, "boost": 1}}} for ans in anss]
    bod['query']['bool']['minimum_should_match'] = 2
    #############
    res             = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

data_path           = sys.argv[1]
ofpath              = sys.argv[2]

filter_keep_these   = ['factoid_snippet']
data                = pickle.load(open(data_path, 'rb'))
data                = [item for item in data if item[-1] in filter_keep_these]
q2snip              = {}
q2ans               = {}
for q, anss, snip, type in data:
    if q in q2snip:
        q2ans[q].extend([t.lower() for t in anss])
    else:
        q2ans[q] = [t.lower() for t in anss]
    q2ans[q] = list(set(q2ans[q]))
    #
    if q in q2snip:
        q2snip[q].add(snip)
    else:
        q2snip[q] = set([snip])

cluster_ips         = [
    '192.168.188.95:9200',
    '192.168.188.86:9200',
    '192.168.188.79:9201',
    '192.168.188.55:9200',
    '192.168.188.108:9200',
    '192.168.188.109:9200',
    '192.168.188.110:9200'
]

doc_index           = 'pubmed_abstracts_joint_0_1'
mapping             = "abstract_map_joint_0_1"
es                  = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

more_data           = []
pbar                = tqdm(q2ans)
for q in pbar:
    r       = get_first_n_1(q, q2ans[q], 500, max_year=2022)
    all_sents = []
    for item in r :
        title   = item['_source']['joint_text'].split('--------------------')[0].strip()
        abs     = [t.strip() for t in item['_source']['joint_text'].split('--------------------')[1].strip().split('\n\n')]
        all_sents = get_sents(title)
        for p in abs:
            all_sents.extend(get_sents(p))
    for sent in all_sents:
        if sent in q2snip[q]:
            continue
        if any(ans.lower() in sent.lower() for ans in q2ans[q]):
            more_data.append((q, q2ans[q], sent, 'ir_aug'))
    pbar.set_description(str(len(more_data)))

print(len(more_data))

pickle.dump(more_data, open(ofpath,'wb'))










